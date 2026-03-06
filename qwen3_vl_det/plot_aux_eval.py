"""Plot analysis charts from eval_split_aux JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LM/DETR metrics from aux eval JSON.")
    p.add_argument(
        "--input-json",
        action="append",
        required=True,
        help="Path(s) to eval_split_aux JSON. Repeat for multi-run comparison.",
    )
    p.add_argument("--labels", default="", help="Comma-separated labels matching --input-json order.")
    p.add_argument("--output-dir", default="qwen3_vl_det/analysis_plots")
    p.add_argument("--title", default="Aux Eval Analysis")
    return p.parse_args()


def _safe_float(d: dict[str, Any], k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(d: dict[str, Any], k: str, default: int = 0) -> int:
    v = d.get(k, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_label(path: str, idx: int) -> str:
    stem = Path(path).stem
    return f"{idx+1}:{stem}"


def _plot_single(run: dict[str, Any], label: str, output_dir: str, title: str) -> None:
    import matplotlib.pyplot as plt

    metrics_detr = run.get("metrics_detr", run.get("metrics", {}))
    metrics_lm_text = run.get("metrics_lm_text", {})
    metrics_lm_boxes = run.get("metrics_lm_boxes", {})
    bucket_metrics = run.get("bucket_metrics", {})

    # Overall summary bars
    names = [
        "DETR MAE",
        "DETR SoftMAE",
        "LM-Text MAE",
        "LM-Text Acc",
        "LM-Box MAE",
    ]
    values = [
        _safe_float(metrics_detr, "count_mae"),
        _safe_float(metrics_detr, "count_soft_mae"),
        _safe_float(metrics_lm_text, "count_mae"),
        _safe_float(metrics_lm_text, "count_accuracy"),
        _safe_float(metrics_lm_boxes, "count_mae"),
    ]

    plt.figure(figsize=(10, 4.8))
    bars = plt.bar(names, values)
    plt.title(f"{title}\n{label} - Overall Metrics")
    plt.ylabel("Value")
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out1 = os.path.join(output_dir, f"{label}_overall.png")
    plt.savefig(out1, dpi=180)
    plt.close()

    # Bucket LM text MAE/Acc
    buckets = ["easy", "medium", "hard", "extreme"]
    lm_text_mae = []
    lm_text_acc = []
    n_samples = []
    for b in buckets:
        bm = bucket_metrics.get(b, {})
        lm_t = bm.get("lm_text", {})
        lm_text_mae.append(_safe_float(lm_t, "count_mae"))
        lm_text_acc.append(_safe_float(lm_t, "count_accuracy"))
        n_samples.append(_safe_int(bm, "num_samples"))

    x = list(range(len(buckets)))
    plt.figure(figsize=(10, 4.8))
    width = 0.4
    plt.bar([i - width / 2 for i in x], lm_text_mae, width=width, label="LM Text MAE")
    plt.bar([i + width / 2 for i in x], lm_text_acc, width=width, label="LM Text Acc")
    plt.xticks(x, [f"{b}\n(n={n})" for b, n in zip(buckets, n_samples)])
    plt.title(f"{title}\n{label} - LM Text by Bucket")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(output_dir, f"{label}_lm_text_bucket.png")
    plt.savefig(out2, dpi=180)
    plt.close()

    # Bucket DETR hard/soft MAE
    detr_mae = []
    detr_soft = []
    for b in buckets:
        bm = bucket_metrics.get(b, {})
        detr_mae.append(_safe_float(bm, "count_mae"))
        detr_soft.append(_safe_float(bm, "count_soft_mae"))
    plt.figure(figsize=(10, 4.8))
    plt.plot(buckets, detr_mae, marker="o", label="DETR MAE")
    plt.plot(buckets, detr_soft, marker="o", label="DETR Soft MAE")
    plt.title(f"{title}\n{label} - DETR Count Error by Bucket")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    out3 = os.path.join(output_dir, f"{label}_detr_bucket.png")
    plt.savefig(out3, dpi=180)
    plt.close()


def _plot_compare(runs: list[dict[str, Any]], labels: list[str], output_dir: str, title: str) -> None:
    import matplotlib.pyplot as plt

    if len(runs) < 2:
        return
    keys = ["detr_mae", "detr_soft_mae", "lm_text_mae", "lm_text_acc", "lm_boxes_mae"]
    vals = {k: [] for k in keys}
    for run in runs:
        md = run.get("metrics_detr", run.get("metrics", {}))
        mt = run.get("metrics_lm_text", {})
        mb = run.get("metrics_lm_boxes", {})
        vals["detr_mae"].append(_safe_float(md, "count_mae"))
        vals["detr_soft_mae"].append(_safe_float(md, "count_soft_mae"))
        vals["lm_text_mae"].append(_safe_float(mt, "count_mae"))
        vals["lm_text_acc"].append(_safe_float(mt, "count_accuracy"))
        vals["lm_boxes_mae"].append(_safe_float(mb, "count_mae"))

    x = list(range(len(labels)))
    plt.figure(figsize=(11, 5.2))
    plt.plot(x, vals["lm_text_mae"], marker="o", label="LM Text MAE")
    plt.plot(x, vals["lm_text_acc"], marker="o", label="LM Text Acc")
    plt.plot(x, vals["detr_soft_mae"], marker="o", label="DETR Soft MAE")
    plt.plot(x, vals["detr_mae"], marker="o", label="DETR MAE")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title(f"{title}\nCross-run Comparison")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_runs.png")
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels.strip() else []
    if labels and len(labels) != len(args.input_json):
        raise ValueError("If provided, --labels must have same length as --input-json count.")
    if not labels:
        labels = [_derive_label(p, i) for i, p in enumerate(args.input_json)]

    runs = [_load_json(p) for p in args.input_json]

    # Save compact summary table
    summary_rows: list[dict[str, Any]] = []
    for lab, run in zip(labels, runs):
        md = run.get("metrics_detr", run.get("metrics", {}))
        mt = run.get("metrics_lm_text", {})
        mb = run.get("metrics_lm_boxes", {})
        summary_rows.append(
            {
                "label": lab,
                "detr_mae": _safe_float(md, "count_mae"),
                "detr_soft_mae": _safe_float(md, "count_soft_mae"),
                "lm_text_mae": _safe_float(mt, "count_mae"),
                "lm_text_acc": _safe_float(mt, "count_accuracy"),
                "lm_text_parse_rate": _safe_float(mt, "parse_rate"),
                "lm_boxes_mae": _safe_float(mb, "count_mae"),
            }
        )
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    try:
        for lab, run in zip(labels, runs):
            _plot_single(run=run, label=lab, output_dir=args.output_dir, title=args.title)
        _plot_compare(runs=runs, labels=labels, output_dir=args.output_dir, title=args.title)
    except Exception as e:
        print(f"WARNING: plotting failed ({type(e).__name__}: {e}).")
        print("summary.json was still written.")
        return

    print(f"Wrote plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
