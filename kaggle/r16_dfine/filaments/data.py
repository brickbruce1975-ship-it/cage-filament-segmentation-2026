"""Competition data: COCO annotations -> per-annotator masks, date-grouped split, D-FINE-seg dataset.

Every observation (one H-alpha JPEG) is annotated by 1-3 annotators independently, and the
organisers score each annotator view separately. We keep that structure end to end:

* training: one sample per annotator view (the JPEG paired with that annotator's mask), so
  the model sees every annotator's style instead of a consensus nobody drew;
* validation: all views of an observation stay in the same split (split by observation
  *date*, so near-duplicate frames of the same day never leak across the split);
* scoring: predictions for an observation are matched against each of its views.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

SIZE = 2048  # every GONG frame in the competition is 2048 x 2048
ANN_NAME = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


@dataclass(frozen=True)
class View:
    """One annotator's drawing of one observation."""

    view_id: str  # e.g. "010101-20110114105034Ch" (annotator batch - observation stem)
    stem: str  # observation stem, e.g. "20110114105034Ch"
    rles: tuple  # one COCO RLE per annotated filament


def load_views(raw_dir: Path) -> list[View]:
    """All annotator views of the training set, polygons rasterised exactly like the scorer."""
    coco = json.loads((Path(raw_dir) / ANN_NAME).read_text())
    polys = defaultdict(list)
    for a in coco["annotations"]:
        polys[a["image_id"]].append(a["segmentation"])
    views = []
    for im in coco["images"]:
        rles = tuple(mask_utils.merge(mask_utils.frPyObjects(p, SIZE, SIZE)) for p in polys[im["id"]])
        views.append(View(im["id"], Path(im["file_name"]).stem, rles))
    return views


def split_dates(stems: list[str], val_frac: float = 0.2, seed: int = 0) -> set[str]:
    """Observation dates (YYYYMMDD) held out for validation."""
    dates = sorted({s[:8] for s in stems})
    random.Random(seed).shuffle(dates)
    return set(dates[: round(len(dates) * val_frac)])


def group_by_observation(views: list[View]) -> dict[str, list[tuple]]:
    """{observation stem: [view0 rles, view1 rles, ...]} - the layout the PQ scorer expects."""
    gt = defaultdict(list)
    for v in views:
        gt[v.stem].append(list(v.rles))
    return dict(gt)


def val_ground_truth(raw_dir: Path, val_frac: float = 0.2, seed: int = 0) -> dict[str, list]:
    views = load_views(raw_dir)
    val = split_dates([v.stem for v in views], val_frac, seed)
    return group_by_observation([v for v in views if v.stem[:8] in val])


def prepare_sem_dataset(raw_dir: Path, out: Path, val_frac: float = 0.2, seed: int = 0) -> None:
    """Write the D-FINE-seg `sem_seg` dataset layout.

    out/images/<view_id>.jpeg  symlink to the shared observation JPEG
    out/labels/<view_id>.png   uint8, 0 = background, 1 = filament (union of the view's polygons)
    out/train.csv, out/val.csv image file names per split
    """
    raw_dir, out = Path(raw_dir).resolve(), Path(out)
    views = load_views(raw_dir)
    val = split_dates([v.stem for v in views], val_frac, seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for v in views:
        img = out / "images" / f"{v.view_id}.jpeg"
        if not img.exists():
            img.symlink_to(raw_dir / "train/train_images" / f"{v.stem}.jpeg")
        mask = mask_utils.decode(mask_utils.merge(list(v.rles))) if v.rles else None
        if mask is None:
            mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.imwrite(str(out / "labels" / f"{v.view_id}.png"), mask.astype(np.uint8))
        splits["val" if v.stem[:8] in val else "train"].append(img.name)
    for name, files in splits.items():
        (out / f"{name}.csv").write_text("\n".join(sorted(files)) + "\n")
    n_obs = len({v.stem for v in views if v.stem[:8] in val})
    print(f"train views {len(splits['train'])}, val views {len(splits['val'])} ({n_obs} observations)")
