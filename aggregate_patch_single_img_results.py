#!/usr/bin/env python3
"""Aggregate single-image patch-token experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("results/mambaformer_win96_single_img_patch_tod_selected")
SPEC_PATH = Path("results/main_patch_tod_tuned_final_summary/best_by_horizon_final.json")
OUT_DIR = ROOT
HORIZONS = [15, 30, 45, 60, 75, 90]

BASELINE = {
    15: {"MAE": 9.33, "RMSE": 13.66},
    30: {"MAE": 15.28, "RMSE": 21.83},
    45: {"MAE": 19.75, "RMSE": 29.40},
    60: {"MAE": 24.11, "RMSE": 35.68},
    75: {"MAE": 28.67, "RMSE": 41.88},
    90: {"MAE": 31.67, "RMSE": 46.41},
}


def read_json_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct_improvement(value: float, base: float) -> float:
    return (base - value) / base * 100.0


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    baseline_avg_mae = sum(BASELINE[h]["MAE"] for h in HORIZONS) / len(HORIZONS)
    baseline_avg_rmse = sum(BASELINE[h]["RMSE"] for h in HORIZONS) / len(HORIZONS)
    rows.append(
        {
            "model": "MambaFormer-96",
            "image_type": "none",
            "metric": "MAE",
            **{f"{h}min": BASELINE[h]["MAE"] for h in HORIZONS},
            "Avg": baseline_avg_mae,
        }
    )
    rows.append(
        {
            "model": "MambaFormer-96",
            "image_type": "none",
            "metric": "RMSE",
            **{f"{h}min": BASELINE[h]["RMSE"] for h in HORIZONS},
            "Avg": baseline_avg_rmse,
        }
    )

    if SPEC_PATH.exists():
        spec_rows = {int(r["horizon_min"]): r for r in read_json_rows(SPEC_PATH)}
        rows.extend(make_metric_rows("+ Spectrogram", "spectrogram", spec_rows))

    for image_type, label in [("rp", "+ RP"), ("gaf", "+ GAF"), ("mtf", "+ MTF")]:
        image_rows: dict[int, dict] = {}
        for path in sorted((ROOT / image_type).glob("*/results_*.json")):
            for row in read_json_rows(path):
                image_rows[int(row["horizon_min"])] = row
        rows.extend(make_metric_rows(label, image_type, image_rows))

    return rows


def make_metric_rows(model: str, image_type: str, by_horizon: dict[int, dict]) -> list[dict]:
    out = []
    for metric in ["MAE", "RMSE"]:
        values = {f"{h}min": float(by_horizon[h][metric]) for h in HORIZONS}
        avg = sum(values[f"{h}min"] for h in HORIZONS) / len(HORIZONS)
        out.append(
            {
                "model": model,
                "image_type": image_type,
                "metric": metric,
                **values,
                "Avg": avg,
            }
        )
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["model", "image_type", "metric", *[f"{h}min" for h in HORIZONS], "Avg"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_md(rows: list[dict], path: Path) -> None:
    headers = ["Model", "Metric", *[f"{h} min" for h in HORIZONS], "Avg"]
    lines = [
        "# Single-Image Patch-Token Results",
        "",
        "All image variants use MambaFormer-96 with DINOv2 patch tokens, gated residual fusion, and time-of-day features. Lower MAE/RMSE is better.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [row["model"], row["metric"]]
        vals.extend(fmt(row[f"{h}min"]) for h in HORIZONS)
        vals.append(fmt(row["Avg"]))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Improvement vs. MambaFormer-96 Baseline",
            "",
            "Positive values indicate lower error than the sequence-only MambaFormer-96 baseline.",
            "",
            "| Model | Metric | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | Avg |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    baseline_by_metric = {row["metric"]: row for row in rows if row["model"] == "MambaFormer-96"}
    for row in rows:
        if row["model"] == "MambaFormer-96":
            continue
        base = baseline_by_metric[row["metric"]]
        vals = [row["model"], row["metric"]]
        vals.extend(f"{pct_improvement(row[f'{h}min'], base[f'{h}min']):+.2f}%" for h in HORIZONS)
        vals.append(f"{pct_improvement(row['Avg'], base['Avg']):+.2f}%")
        lines.append("| " + " | ".join(vals) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "summary_patch_single_img_with_spec.csv")
    write_json(rows, OUT_DIR / "summary_patch_single_img_with_spec.json")
    write_md(rows, OUT_DIR / "summary_patch_single_img_with_spec.md")
    print(f"Wrote {OUT_DIR / 'summary_patch_single_img_with_spec.md'}")


if __name__ == "__main__":
    main()
