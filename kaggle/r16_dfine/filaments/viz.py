"""Plots shared by the notebook and the report figures."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

from .metrics import Report

# tab20 without its two greys, which vanish against the solar disk
PALETTE = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)]


def overlay(img_bgr: np.ndarray, rles: list[dict], alpha: float = 0.45) -> np.ndarray:
    """RGB image with each instance filled in its own colour and outlined."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = rgb.copy()
    for i, rle in enumerate(rles):
        m = mask_utils.decode(rle).astype(bool)
        c = np.array(PALETTE[i % len(PALETTE)]) * 255
        out[m] = (1 - alpha) * out[m] + alpha * c
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, tuple(float(v) for v in c), 2)
    return out.clip(0, 255).astype(np.uint8)


def show_row(panels: list[tuple[str, np.ndarray]], crop=None, size: float = 3.2, cmap=None):
    """One row of titled panels, optionally cropped to (y0, y1, x0, x1)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(size * len(panels), size))
    for ax, (title, im) in zip(np.atleast_1d(axes), panels):
        if crop is not None:
            y0, y1, x0, x1 = crop
            im = im[y0:y1, x0:x1]
        ax.imshow(im, cmap=cmap if im.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def distributions(model: Report, annot: Report, figsize=(7.0, 2.1)):
    """Model vs inter-annotator histograms of the rubric's per-item scores."""
    items = [
        ("Dice of matched filaments", model.matched_dice, annot.matched_dice, (0.5, 1)),
        ("best IoU per GT filament", np.asarray(model.gt_best_iou), np.asarray(annot.gt_best_iou), (0, 1)),
        ("image Dice (per view)", np.asarray(model.image_dice), np.asarray(annot.image_dice), (0, 1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (title, a, b, rng) in zip(axes, items):
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        bins = np.linspace(*rng, 21)
        ax.hist(b, bins, density=True, color="0.6", alpha=0.6, label=f"annotators ({np.mean(b):.2f})")
        ax.hist(a, bins, density=True, histtype="step", lw=1.6, color="C0", label=f"model ({np.mean(a):.2f})")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    return fig
