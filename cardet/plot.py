"""Plot training curves from Ultralytics runs.

python3 -m cardet.plot out.png run_dir [run_dir ...]      (each run_dir holds results.csv, e.g. runs/detect/runs/ycA)
Left: training loss per epoch (log scale), the usual loss curve. Right: held-out precision, recall and mAP50 per epoch.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(run):
    with open(Path(run) / "results.csv") as f:
        rows = [{k.strip(): float(v) for k, v in r.items() if v.strip()} for r in csv.DictReader(f)]
    return rows


def plot(out, runs):
    fig, (loss, val) = plt.subplots(1, 2, figsize=(13, 5))
    for i, run in enumerate(runs):
        rows, name = read(run), Path(run).name
        colour = f"C{i}"
        epochs = [r["epoch"] for r in rows]
        total = [r["train/box_loss"] + r["train/cls_loss"] + r["train/dfl_loss"] for r in rows]
        loss.plot(epochs, total, color=colour, label=f"{name} train loss")
        if "val/box_loss" in rows[0]:
            vtotal = [r["val/box_loss"] + r["val/cls_loss"] + r["val/dfl_loss"] for r in rows]
            loss.plot(epochs, vtotal, color=colour, linestyle="--", label=f"{name} held-out loss")
        val.plot(epochs, [r["metrics/mAP50(B)"] for r in rows], color=colour, label=f"{name} mAP50")
        val.plot(epochs, [r["metrics/precision(B)"] for r in rows], color=colour, linestyle=":", label=f"{name} precision")
        val.plot(epochs, [r["metrics/recall(B)"] for r in rows], color=colour, linestyle="-.", label=f"{name} recall")
    loss.set(yscale="log", xlabel="epoch", ylabel="box + class + dfl loss (log scale)", title="Loss curve")
    val.set(xlabel="epoch", ylabel="score on held-out tiles", ylim=(0, 1), title="Held-out accuracy")
    for ax in (loss, val):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130)


if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2:])
