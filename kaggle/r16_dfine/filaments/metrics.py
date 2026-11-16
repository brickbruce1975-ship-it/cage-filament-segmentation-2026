"""Competition metrics, computed the way the organisers' self-evaluation notebook does.

Predictions of an observation are compared with every annotator view of it separately and
the counts are pooled over all (observation, view) pairs:

* Panoptic Quality: a prediction and a ground-truth filament match when IoU > 0.5
  (such matches are unique, so no assignment solver is needed);
  PQ = sum(IoU of matches) / (TP + FP/2 + FN/2) = SQ * RQ.
* Instance Dice / IoU distributions: of every matched pair, and the best IoU of every
  ground-truth filament (0 when nothing overlaps it).
* Image Dice: binary Dice of the union of predicted masks vs the union of the view's masks
  (the filament class; one value per view, views where both are empty are skipped).
* One-to-many: ground-truth filaments overlapped (IoU > 0) by more than one prediction
  (fragmentation). Many-to-one: predictions overlapping more than one ground-truth filament
  (over-merging).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class Report:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    one_to_many: int = 0  # GT split across > 1 prediction
    many_to_one: int = 0  # prediction spanning > 1 GT
    n_gt: int = 0
    n_pred: int = 0
    matched_iou: list = field(default_factory=list)
    gt_best_iou: list = field(default_factory=list)
    image_dice: list = field(default_factory=list)

    @property
    def pq(self) -> float:
        return self.iou_sum / max(self.tp + 0.5 * self.fp + 0.5 * self.fn, 1e-9)

    @property
    def sq(self) -> float:
        return self.iou_sum / max(self.tp, 1)

    @property
    def rq(self) -> float:
        return self.tp / max(self.tp + 0.5 * self.fp + 0.5 * self.fn, 1e-9)

    @property
    def matched_dice(self) -> np.ndarray:
        iou = np.asarray(self.matched_iou)
        return 2 * iou / (1 + iou)

    def summary(self) -> dict:
        return {
            "PQ": self.pq, "SQ": self.sq, "RQ": self.rq,
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "mean image Dice": float(np.nanmean(self.image_dice)),
            "mean matched Dice": float(self.matched_dice.mean()) if self.matched_iou else 0.0,
            "one-to-many (% of GT)": 100 * self.one_to_many / max(self.n_gt, 1),
            "many-to-one (% of pred)": 100 * self.many_to_one / max(self.n_pred, 1),
        }


def _union_dice(gt: list[dict], pred: list[dict]) -> float:
    a = mask_utils.area(mask_utils.merge(gt)) if gt else 0
    b = mask_utils.area(mask_utils.merge(pred)) if pred else 0
    if a + b == 0:
        return float("nan")
    inter = 0
    if gt and pred:
        inter = mask_utils.area(mask_utils.merge([mask_utils.merge(gt), mask_utils.merge(pred)], intersect=True))
    return 2 * float(inter) / float(a + b)


def add_view(r: Report, gt: list[dict], pred: list[dict]) -> None:
    """Accumulate one (annotator view, prediction set) pair into the report."""
    r.n_gt += len(gt)
    r.n_pred += len(pred)
    r.image_dice.append(_union_dice(gt, pred))
    if not gt or not pred:
        r.fp += len(pred)
        r.fn += len(gt)
        r.gt_best_iou.extend([0.0] * len(gt))
        return
    iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)))  # (n_pred, n_gt)
    hits = iou > 0.5
    tp = int(hits.sum())
    r.tp += tp
    r.fp += len(pred) - tp
    r.fn += len(gt) - tp
    r.iou_sum += float(iou[hits].sum())
    r.matched_iou.extend(iou[hits].tolist())
    r.gt_best_iou.extend(iou.max(0).tolist())
    touch = iou > 0
    r.one_to_many += int((touch.sum(0) > 1).sum())
    r.many_to_one += int((touch.sum(1) > 1).sum())


def evaluate(gt: dict[str, list[list[dict]]], preds: dict[str, list[dict]]) -> Report:
    """gt: {observation: [view rles, ...]}, preds: {observation: rles}."""
    r = Report()
    for stem, views in gt.items():
        for view in views:
            add_view(r, view, preds.get(stem, []))
    return r


def inter_annotator(gt: dict[str, list[list[dict]]]) -> Report:
    """Each annotator scored as a 'prediction' against every other annotator of the same
    observation: the agreement ceiling a model trained on these labels is measured against."""
    r = Report()
    for views in gt.values():
        for i, pred in enumerate(views):
            for j, view in enumerate(views):
                if i != j:
                    add_view(r, view, pred)
    return r
