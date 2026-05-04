"""
MultimodalMambaDINOv2: three-branch multimodal CGM forecaster.

Branches
--------
1. CGM sequence branch   : MambaFormer encoder → sequence tokens (B, L, d_model)
2. Image branch          : DINOv2 (frozen) → patch tokens per selected image(s) →
                           optional token pooling → project + concat image context
3. Time-of-day branch    : sin/cos cyclic encoding → small MLP → tod feature (B, d_tod)

Fusion
------
  cross_attn:
    Q = sequence tokens, K = V = image context
    → image-aware sequence representation → mean-pool
    → concat with tod feature → ForecastHead

  gated_residual:
    base = ForecastHead(mean MambaFormer tokens + ToD)
    delta = ForecastHead(image-aware MambaFormer tokens + ToD)
    pred = base + sigmoid(gate) * delta

image_type : 'rp' | 'spectrogram' | 'gaf' | 'mtf' | 'all'
  Dataset image order: [RP(0), Spectrogram(1), GAF(2), MTF(3)]
"""
import sys, os
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.mambaformer_model import MambaBlock

IMAGE_IDX = {'rp': 0, 'spectrogram': 1, 'gaf': 2, 'mtf': 3}


# ─── 1. CGM Sequence Encoder ─────────────────────────────────────────────────

class MambaFormerSequenceEncoder(nn.Module):
    """
    MambaFormer that returns intermediate token representations.
    Reuses MambaBlock from mambaformer_model.py – same weights/architecture.
    Returns (B, seq_len, d_model) tokens instead of a pooled prediction.
    """

    def __init__(self, seq_len=96, d_model=128, n_heads=4,
                 num_mamba_layers=2, num_attn_layers=2,
                 dim_feedforward=256, dropout=0.1,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_proj  = nn.Linear(1, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_mamba_layers)
        ])
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.attn_encoder = nn.TransformerEncoder(attn_layer,
                                                   num_layers=num_attn_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, seq_len) → tokens: (B, seq_len, d_model)
        x = x.unsqueeze(-1)          # (B, L, 1)
        x = self.input_proj(x)       # (B, L, d_model)
        for blk in self.mamba_layers:
            x = blk(x)
        x = self.attn_encoder(x)     # (B, L, d_model)
        return self.norm(x)


# ─── 2. DINOv2 Image Encoder ─────────────────────────────────────────────────

class DINOv2ImageEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 + learnable linear projection.

    Supports a subset of images via img_indices (e.g. [0] for RP only).
    Returns (B, n * n_patches, d_out) where n = len(img_indices).

    Parameters
    ----------
    d_out     : target projection dimension (should == d_model of sequence encoder)
    freeze    : if True (default), DINOv2 parameters are frozen
    """

    DINO_EMBED = 384   # ViT-S/14 hidden dimension

    def __init__(self, d_out=128, freeze=True, pool='none'):
        super().__init__()
        self.freeze = freeze
        self.pool = pool
        if pool not in ('none', 'mean', 'cls'):
            raise ValueError(f'Unsupported DINOv2 pool mode: {pool}')
        print('[DINOv2] Loading dinov2_vits14 from torch.hub ...', flush=True)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                   pretrained=True, verbose=False)
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()

        self.proj = nn.Linear(self.DINO_EMBED, d_out)

    def forward(self, images, img_indices=None):
        """
        images      : (B, 4, 3, H, W)  – full stack of 4 image types
        img_indices : list of ints, e.g. [0] or [0,1,2,3]; None = all 4
        returns     : (B, n * n_tokens, d_out), where n_tokens is either
                      all patch tokens or one pooled token per image.
        """
        B = images.shape[0]
        if img_indices is None:
            img_indices = list(range(images.shape[1]))
        selected = images[:, img_indices]              # (B, n, 3, H, W)
        n_img = len(img_indices)
        C, H, W = selected.shape[2], selected.shape[3], selected.shape[4]
        imgs_flat = selected.reshape(B * n_img, C, H, W)

        if self.freeze:
            with torch.no_grad():
                feats = self.dino.forward_features(imgs_flat)
        else:
            feats = self.dino.forward_features(imgs_flat)

        if self.pool == 'cls':
            tokens = feats['x_norm_clstoken'].unsqueeze(1)   # (B*n, 1, 384)
        else:
            tokens = feats['x_norm_patchtokens']             # (B*n, n_patches, 384)
            if self.pool == 'mean':
                tokens = tokens.mean(dim=1, keepdim=True)    # (B*n, 1, 384)

        n_tokens = tokens.shape[1]
        tokens = self.proj(tokens)                           # (B*n, n_tokens, d_out)
        tokens = tokens.view(B, n_img * n_tokens, -1)

        return tokens


class CNNImageEncoder(nn.Module):
    """
    Lightweight image encoder for testing whether a generated image
    representation is useful without relying on DINOv2.

    Returns one image token per selected image: (B, n_img, d_out).
    """

    def __init__(self, d_out=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, d_out)

    def forward(self, images, img_indices=None):
        B = images.shape[0]
        if img_indices is None:
            img_indices = list(range(images.shape[1]))
        selected = images[:, img_indices]
        n_img = len(img_indices)
        C, H, W = selected.shape[2], selected.shape[3], selected.shape[4]
        imgs_flat = selected.reshape(B * n_img, C, H, W)
        feats = self.encoder(imgs_flat).flatten(1)
        tokens = self.proj(feats).view(B, n_img, -1)
        return tokens


# ─── 3. Time-of-Day Encoder ──────────────────────────────────────────────────

class TimeOfDayEncoder(nn.Module):
    """Maps (sin, cos) cyclic time-of-day encoding → dense feature."""

    def __init__(self, d_out=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2,    64),
            nn.GELU(),
            nn.Linear(64,   d_out),
        )

    def forward(self, tod):
        # tod: (B, 2)  → (B, d_out)
        return self.mlp(tod)


# ─── 4. Cross-Attention Fusion ───────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Single-layer cross-attention:
      Q  = sequence tokens (B, L, d_model)
      KV = image tokens    (B, N_img, d_model)
    → image-aware sequence (B, L, d_model)
    """

    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, seq_tokens, img_tokens):
        # Cross-attention: Q=seq, KV=img
        attn_out, _ = self.attn(query=seq_tokens,
                                 key=img_tokens,
                                 value=img_tokens)
        seq_tokens = self.norm1(seq_tokens + attn_out)   # residual
        seq_tokens = self.norm2(seq_tokens + self.ff(seq_tokens))
        return seq_tokens  # (B, L, d_model)


class ModalityAttentionFusion(nn.Module):
    """
    Adaptive image-modality fusion over pooled image tokens.

    Query is generated from the temporal feature and optional ToD feature;
    keys/values are the selected image tokens. With image_type='all' and
    dino_pool='mean', the attention is over [RP, Spectrogram, GAF, MTF].
    """

    def __init__(self, d_model=128, d_tod=32):
        super().__init__()
        self.q_proj = nn.Linear(d_model + d_tod, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, seq_feat, tod_feat, img_tokens):
        query_in = torch.cat([seq_feat, tod_feat], dim=-1)
        q = self.q_proj(query_in).unsqueeze(1)       # (B, 1, d_model)
        k = self.k_proj(img_tokens)                  # (B, M, d_model)
        v = self.v_proj(img_tokens)                  # (B, M, d_model)
        scores = (q * k).sum(dim=-1) * self.scale    # (B, M)
        alpha = torch.softmax(scores, dim=-1)
        fused = torch.bmm(alpha.unsqueeze(1), v)     # (B, 1, d_model)
        return fused, alpha


# ─── 5. Forecast Head ────────────────────────────────────────────────────────

class ForecastHead(nn.Module):
    """(d_model + d_tod) → single glucose prediction."""

    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # (B,)


class ResidualGate(nn.Module):
    def __init__(self, d_model, d_tod):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2 + d_tod, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, seq_feat, fused_feat, tod_feat):
        return torch.sigmoid(self.net(torch.cat([seq_feat, fused_feat, tod_feat], dim=-1))).squeeze(-1)


# ─── 6. Full Multimodal Model ────────────────────────────────────────────────

class MultimodalMambaDINOv2(nn.Module):
    """
    Three-branch multimodal glucose forecaster.

    Parameters
    ----------
    seq_len           : input CGM window length (default 96)
    d_model           : shared feature dimension (default 128)
    n_heads           : attention heads in MambaFormer + cross-attention
    num_mamba_layers  : MambaFormer Mamba blocks
    num_attn_layers   : MambaFormer transformer encoder layers
    dim_feedforward   : FFN width in MambaFormer
    dropout           : dropout rate
    d_tod             : time-of-day MLP output dimension
    freeze_dinov2     : whether to freeze DINOv2 (default True)
    """

    def __init__(self, seq_len=96, d_model=128, n_heads=4,
                 num_mamba_layers=2, num_attn_layers=2,
                 dim_feedforward=256, dropout=0.1,
                 d_tod=32, freeze_dinov2=True, image_type='all',
                 use_image=True, use_tod=True,
                 fusion_mode='cross_attn', dino_pool='none',
                 modality_fusion='none', image_encoder='dino'):
        super().__init__()
        assert use_image or use_tod, 'at least one of use_image / use_tod must be True'
        if fusion_mode not in ('cross_attn', 'gated_residual', 'simple_concat'):
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')
        if modality_fusion not in ('none', 'attention', 'uniform'):
            raise ValueError(f'Unsupported modality_fusion: {modality_fusion}')
        if image_encoder not in ('dino', 'cnn'):
            raise ValueError(f'Unsupported image_encoder: {image_encoder}')
        self.use_image = use_image
        self.use_tod   = use_tod
        self.fusion_mode = fusion_mode
        self.modality_fusion = modality_fusion
        self.image_encoder = image_encoder

        if image_type == 'all':
            self.img_indices = [0, 1, 2, 3]
        else:
            self.img_indices = [IMAGE_IDX[image_type]]

        self.seq_enc = MambaFormerSequenceEncoder(
            seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            num_mamba_layers=num_mamba_layers, num_attn_layers=num_attn_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

        if use_image:
            if image_encoder == 'dino':
                self.img_enc = DINOv2ImageEncoder(d_out=d_model, freeze=freeze_dinov2, pool=dino_pool)
            else:
                self.img_enc = CNNImageEncoder(d_out=d_model)
            if modality_fusion == 'attention':
                self.modality_attn = ModalityAttentionFusion(
                    d_model=d_model, d_tod=d_tod if use_tod else 0)
            if fusion_mode != 'simple_concat':
                self.fusion  = CrossAttentionFusion(d_model=d_model,
                                                    n_heads=n_heads, dropout=dropout)

        if use_tod:
            self.tod_enc = TimeOfDayEncoder(d_out=d_tod)

        d_head = d_model + (d_tod if use_tod else 0)
        if fusion_mode == 'simple_concat' and use_image:
            d_head = d_model * 2 + (d_tod if use_tod else 0)
        if fusion_mode == 'gated_residual' and use_image:
            self.base_head = ForecastHead(d_in=d_head)
            self.delta_head = ForecastHead(d_in=d_head)
            nn.init.zeros_(self.delta_head.head[-1].weight)
            nn.init.zeros_(self.delta_head.head[-1].bias)
            self.gate = ResidualGate(d_model=d_model, d_tod=d_tod if use_tod else 0)
        else:
            self.head = ForecastHead(d_in=d_head)

    def forward(self, cgm_seq, images, tod_enc, debug=False, return_aux=False):
        """
        cgm_seq : (B, seq_len)
        images  : (B, 4, 3, 224, 224)
        tod_enc : (B, 2)
        """
        seq_tokens = self.seq_enc(cgm_seq)           # (B, L, d_model)
        base_tokens = seq_tokens

        seq_feat = base_tokens.mean(dim=1)           # (B, d_model)

        if self.use_tod:
            tod_feat = self.tod_enc(tod_enc)         # (B, d_tod)
        else:
            tod_feat = seq_feat.new_zeros(seq_feat.shape[0], 0)

        aux = {}
        if self.use_image:
            img_tokens = self.img_enc(images, self.img_indices)
            if self.modality_fusion == 'attention':
                img_tokens, alpha = self.modality_attn(seq_feat, tod_feat, img_tokens)
                aux['modality_alpha'] = alpha
            elif self.modality_fusion == 'uniform':
                n_tokens = img_tokens.shape[1]
                alpha = img_tokens.new_full(
                    (img_tokens.shape[0], n_tokens), 1.0 / n_tokens)
                img_tokens = img_tokens.mean(dim=1, keepdim=True)
                aux['modality_alpha'] = alpha
            if self.fusion_mode != 'simple_concat':
                seq_tokens = self.fusion(seq_tokens, img_tokens)

        fused_feat = seq_tokens.mean(dim=1)          # (B, d_model)

        if self.fusion_mode == 'simple_concat' and self.use_image:
            img_feat = img_tokens.mean(dim=1)
            fused_feat = torch.cat([fused_feat, img_feat], dim=-1)

        if self.use_tod:
            base_combined = torch.cat([seq_feat, tod_feat], dim=-1)
            fused_combined = torch.cat([fused_feat, tod_feat], dim=-1)
        else:
            base_combined = seq_feat
            fused_combined = fused_feat

        if self.fusion_mode == 'gated_residual' and self.use_image:
            base = self.base_head(base_combined)
            delta = self.delta_head(fused_combined)
            gate = self.gate(seq_feat, fused_feat, tod_feat)
            pred = base + gate * delta
            if return_aux:
                aux['gate'] = gate
                return pred, aux
            return pred

        pred = self.head(fused_combined)
        if return_aux:
            return pred, aux
        return pred
