"""Command line entry point.

  python -m filaments prepare  --data DATA --out data/dataset_sem
  python -m filaments evaluate --data DATA [--weights weights/sem_s_2048.pt]
  python -m filaments predict  --data DATA --out submission.csv [--tta]

DATA is the unzipped competition folder `MAGFiLO_1.0_Kaggle_2026` (train/ and test/ inside).
Training itself is D-FINE-seg's own trainer, see README.md.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from .data import prepare_sem_dataset, val_ground_truth
from .inference import FilamentSegmenter, PostProcess, write_submission
from .metrics import evaluate, inter_annotator

WEIGHTS = Path(__file__).resolve().parents[1] / "weights/sem_s_2048.pt"


def _predict_dir(seg: FilamentSegmenter, image_dir: Path, stems=None) -> dict[str, list]:
    paths = sorted(image_dir.glob("*.jpeg"))
    if stems is not None:
        paths = [p for p in paths if p.stem in stems]
    t = time.perf_counter()
    preds = {p.stem: seg(cv2.imread(str(p), cv2.IMREAD_COLOR)) for p in tqdm(paths)}
    print(f"{1000 * (time.perf_counter() - t) / len(paths):.0f} ms/image end-to-end")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m filaments")
    ap.add_argument("command", choices=["prepare", "evaluate", "predict"])
    ap.add_argument("--data", type=Path, required=True, help="MAGFiLO_1.0_Kaggle_2026 folder")
    ap.add_argument("--out", type=Path, help="dataset dir (prepare) or submission CSV (predict)")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--tta", action="store_true", help="average 8 flips/rotations (slower)")
    ap.add_argument("--min-score", type=float, default=PostProcess.min_score)
    a = ap.parse_args()

    if a.command == "prepare":
        prepare_sem_dataset(a.data, a.out or Path("data/dataset_sem"))
        return
    seg = FilamentSegmenter(a.weights, PostProcess(min_score=a.min_score), tta=a.tta)
    if a.command == "predict":
        preds = _predict_dir(seg, a.data / "test/test_images")
        write_submission(preds, a.out or Path("submission.csv"))
        return
    gt = val_ground_truth(a.data)
    preds = _predict_dir(seg, a.data / "train/train_images", set(gt))
    for name, rep in [("model", evaluate(gt, preds)), ("annotators", inter_annotator(gt))]:
        print(name, json.dumps({k: round(v, 4) for k, v in rep.summary().items()}))


if __name__ == "__main__":
    main()
