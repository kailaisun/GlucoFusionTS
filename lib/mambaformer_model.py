"""
MambaFormer: Mamba SSM encoder + Transformer decoder for CGM forecasting.
Uses mamba-ssm (selective state space model) for long-range sequence modeling.
Falls back to a pure-Transformer if mamba_ssm is unavailable.
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class MambaBlock(nn.Module):
    """One Mamba layer with residual + LayerNorm."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        else:
            # Fallback: 1-D depthwise conv with padding='same' for exact length preservation
            self.mamba = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3,
                          padding=1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 1))

    def forward(self, x):
        # x: (B, L, d_model)
        res = x
        x = self.norm(x)
        if HAS_MAMBA:
            x = self.mamba(x)
        else:
            x = x.transpose(1, 2)
            x = self.mamba(x)
            x = x.transpose(1, 2)
        return x + res


class MambaFormer(nn.Module):
    def __init__(self, seq_len=96, pred_len=12, d_model=128, n_heads=4,
                 num_mamba_layers=2, num_attn_layers=2,
                 dim_feedforward=256, dropout=0.1,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_mamba_layers)
        ])
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, num_layers=num_attn_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (B, seq_len)
        x = x.unsqueeze(-1)                 # (B, L, 1)
        x = self.input_proj(x)              # (B, L, d_model)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.attn_encoder(x)            # (B, L, d_model)
        x = self.norm(x)
        x = x.transpose(1, 2)              # (B, d_model, L)
        x = self.pool(x).squeeze(-1)        # (B, d_model)
        return self.head(x)                 # (B, pred_len)
