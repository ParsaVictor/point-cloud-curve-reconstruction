"""Generate the pipeline architecture diagram saved to results/architecture.png."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "architecture.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ── colour palette ──────────────────────────────────────────────────────────
C_IN    = "#4A90D9"   # input / output — blue
C_FILT  = "#5BAD6F"   # filter stage — green
C_CURVE = "#E8852B"   # curve stage — orange
C_OUT   = "#7B5EA7"   # output — purple
C_PAPER = "#555555"   # paper label — grey

def box(ax, x, y, w, h, label, sub, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
            fontsize=7.5, color="white", alpha=0.88)

def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))

# ── boxes ────────────────────────────────────────────────────────────────────
BH, BY = 1.6, 2.7   # box height, vertical centre

box(ax,  0.3, BY, 1.7, BH, "Raw Point\nCloud", "3D scan (N pts)", C_IN)
arrow(ax, 2.0, 2.3, BY + BH / 2)
box(ax,  2.3, BY, 2.0, BH, "kNN Outlier\nFilter", "MAD-based\n(Lee 2000)", C_FILT)
arrow(ax, 4.3, 4.6, BY + BH / 2)
box(ax,  4.6, BY, 2.0, BH, "Density\nFilter", "inv-distance\ndensity", C_FILT)
arrow(ax, 6.6, 6.9, BY + BH / 2)
box(ax,  6.9, BY, 2.0, BH, "LOF Filter", "Local Outlier\nFactor", C_FILT)
arrow(ax, 8.9, 9.2, BY + BH / 2)
box(ax,  9.2, BY, 2.3, BH, "Curve-Point\nRecovery", "PCA + poly fit\n(Peng 2022)", C_CURVE)
arrow(ax, 11.5, 11.8, BY + BH / 2)
box(ax, 11.8, BY, 1.9, BH, "Clean\nPoint Cloud", "M pts retained", C_OUT)

# ── paper labels (above each stage) ──────────────────────────────────────────
papers = [
    (3.3,  "Lee 2000\nMST + kNN"),
    (5.6,  "Liu 2006\nParam. search"),
    (7.9,  "Fugacci 2024\nFeature curves"),
    (10.35,"Peng 2022\nClust. + PCA"),
]
for xc, lbl in papers:
    ax.text(xc, BY + BH + 0.35, lbl, ha="center", va="bottom",
            fontsize=7.5, color=C_PAPER, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="#eeeeee", ec="#aaaaaa", alpha=0.7))

# ── Javidrad 2011 — B-spline contour (separate lane) ─────────────────────────
box(ax, 2.3, 0.6, 4.3, 1.2, "B-spline Contour Fitting  (Javidrad 2011)",
    "Adaptive slicing + control-point extraction", C_CURVE, fontsize=8.5)
ax.annotate("", xy=(4.45, 2.7), xytext=(4.45, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2, linestyle="dashed"))
ax.text(4.65, 2.25, "feeds\ndiagnosis", fontsize=7, color="#888888", style="italic")

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7, 6.55, "3D Point Cloud Curve Reconstruction — Pipeline Overview",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#222222")
ax.text(7, 6.1, "Five peer-reviewed algorithms benchmarked on industrial 3D-printing scan data",
        ha="center", va="center", fontsize=9, color="#555555")

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_FILT,  label="Statistical filter"),
    mpatches.Patch(color=C_CURVE, label="Curve-fitting stage"),
    mpatches.Patch(color=C_IN,    label="Input / Output"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8,
          framealpha=0.85, edgecolor="#aaaaaa")

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(OUT)}")
