"""Image -> filament probability map -> filament instances (COCO RLEs).

The network is a D-FINE-seg-S semantic segmentation model run on the native 2048 x 2048
frame. Instances come from the probability map:

1. threshold p(filament) at `thresh`;
2. morphological closing with a disk of radius `close` bridges the 1-3 px gaps that split
   one filament into several pieces;
3. 8-connected components are the instances (`min_area` drops specks);
4. each component is scored by its mean probability and kept if the score >= `min_score`.
   An emitted component adds 1/2 to PQ's denominator whether it matches or not, and its
   IoU to the numerator only if it matches, so it pays off only if P(match) > PQ / (2 SQ),
   about 1/3 here. Dropping uncertain components adds +0.010 PQ on validation.

Components never overlap, so the output is a valid Kaggle submission as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

SIZE = 2048


@dataclass(frozen=True)
class PostProcess:
    """Settings of the selected submission (tuned on validation, see README)."""

    thresh: float = 0.4
    close: int = 4
    min_area: int = 200
    min_score: float = 0.90


# The 8 flips / 90-degree rotations of a square image (optional test-time augmentation).
_VIEWS = [(k, f) for f in (False, True) for k in range(4)]


def _dihedral(x: torch.Tensor, k: int, flip: bool, inverse: bool = False) -> torch.Tensor:
    if not inverse:
        return torch.rot90(x.flip(-1) if flip else x, k, (-2, -1))
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if flip else x


class FilamentSegmenter:
    """D-FINE-seg sem_seg model + component post-processing.

    >>> seg = FilamentSegmenter("weights/sem_s_2048.pt")
    >>> rles = seg(cv2.imread("20150125172714Mh.jpeg"))   # list of COCO RLEs
    """

    def __init__(self, weights: str | Path, post: PostProcess = PostProcess(), tta: bool = False,
                 device: str | None = None):
        from dfine_seg import load_model  # heavy import, keep module import cheap

        kw = {"device": device} if device else {}
        self.net = load_model(str(weights), input_height=SIZE, input_width=SIZE, **kw)
        self.post = post
        self.tta = tta

    @torch.no_grad()
    def probabilities(self, img_bgr: np.ndarray) -> np.ndarray:
        """uint8 map of round(255 * p(filament)) at the input resolution."""
        x, _, (orig,) = self.net._prepare_inputs(img_bgr)
        views = _VIEWS if self.tta else _VIEWS[:1]
        prob = 0
        for k, f in views:
            logits = self.net._predict(_dihedral(x, k, f))["sem_seg_logits"].float()
            prob = prob + _dihedral(logits.softmax(1)[:, 1:2], k, f, inverse=True)
        prob = torch.nn.functional.interpolate(prob / len(views), size=orig, mode="bilinear")
        return (prob[0, 0] * 255).round().byte().cpu().numpy()

    def __call__(self, img_bgr: np.ndarray) -> list[dict]:
        return [rle for _, rle in instances(self.probabilities(img_bgr), self.post)]


def instances(prob: np.ndarray, post: PostProcess = PostProcess()) -> list[tuple[float, dict]]:
    """[(score, COCO RLE)] for the kept components of a uint8 probability map, in scan order."""
    fg = (prob >= post.thresh * 255).view(np.uint8)
    if post.close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * post.close + 1, 2 * post.close + 1))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < post.min_area:
            continue
        x, y, w, h = stats[i, :4]
        crop = lbl[y : y + h, x : x + w] == i
        score = float(prob[y : y + h, x : x + w][crop].astype(np.float32).mean()) / 255
        if score < post.min_score:
            continue
        out.append((score, _crop_rle(crop, x, y, prob.shape)))
    return out


def _crop_rle(crop: np.ndarray, x: int, y: int, shape: tuple[int, int]) -> dict:
    """COCO RLE of a full-frame mask that is `crop` at (x, y) and 0 elsewhere.

    Same result as mask_utils.encode on the full frame, without materialising it: COCO RLE
    runs over pixels in column-major order, so the mask's pixels map to sorted flat indices.
    """
    h, _ = shape
    cols, rows = np.nonzero(crop.T)  # column-major scan order
    idx = (cols + x).astype(np.int64) * h + rows + y
    brk = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]] + 1
    gaps = starts - np.r_[0, ends[:-1]]
    counts = np.column_stack([gaps, ends - starts]).ravel().tolist() + [int(h * shape[1] - ends[-1])]
    return mask_utils.frPyObjects({"size": list(shape), "counts": counts}, *shape)


def write_submission(preds: dict[str, list[dict]], path: str | Path) -> None:
    """Kaggle CSV: filament_id = <observation>_<k>, segmentation_rle = RLE counts (2048 x 2048)."""
    rows = ["filament_id,segmentation_rle"]
    for stem in sorted(preds):
        for i, rle in enumerate(preds[stem], 1):
            rows.append(f"{stem}_{i},{rle['counts'].decode('ascii')}")
    Path(path).write_text("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} filaments over {len(preds)} images -> {path}")
