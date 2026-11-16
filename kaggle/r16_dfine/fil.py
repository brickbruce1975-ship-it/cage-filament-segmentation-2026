#!/usr/bin/env python3
"""Filament r16 inference kernel (PREREG_r16_2026-11-16.md + A1).

Frozen ArgoHA D-FINE-seg-S weights + frozen PostProcess. No training, no labels,
no metric. Writes holdout_inst_r16.pkl, submission.csv, done_r16.json.
A1: sibling kernel files are not copied to /kaggle/src; stems are embedded and
weights/package come from dataset jamesmills1975/cage-filament-r16-dfine-w.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Frozen 25 holdout stems (holdout46_files.json pretty-file sha256
# 9b0a5f1aa5632979afa1c4a96cb30f14b25013e3982eda648c697381a8221962).
FILES = ['20110523063154Uh', '20120331142954Bh', '20120910074014Th', '20121022145554Bh', '20130218220314Mh', '20130221063134Lh', '20140111224134Lh', '20140205180154Bh', '20140218230234Lh', '20140509071314Th', '20140603070414Th', '20140615195814Mh', '20141216222634Lh', '20141225172214Mh', '20151106171514Mh', '20151116201114Mh', '20160229151254Bh', '20160606163514Mh', '20160925165814Mh', '20170310173450Bh', '20180306081810Th', '20181101230010Mh', '20190925165810Mh', '20210626133950Bh', '20210814044930Lh']
EXPECTED_FILES_SHA = "7378afd5668b877e0e5f49f3cb8adbf51bd8bc4609074b79c22aa6d850fb096b"  # sha256(json.dumps(FILES))
WEIGHTS_SHA = "11958712586f000a164c07bf77164ee39ed281bbb99b65a581e2da04a548cff8"

# Box has a non-empty /kaggle/input from other agents. Kaggle = /kaggle/input
# exists AND the box MAGFiLO path is absent (same rule as r15).
BOX_DATA = Path("/workspace/fil/MAGFiLO_1.0_Kaggle_2026")
ON_KAGGLE = Path("/kaggle/input").exists() and not BOX_DATA.exists()
if ON_KAGGLE:
    hits = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026/", recursive=True)
    DATA = Path(hits[0]) if hits else Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
    OUT = Path("/kaggle/working")
else:
    DATA = BOX_DATA
    OUT = Path("/workspace/fil/r16/run_box")
OUT.mkdir(parents=True, exist_ok=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_weights() -> Path:
    cands = []
    for root in [Path("/kaggle/input"), HERE, Path("/workspace/fil/r16/dataset"), Path("/workspace/fil/r16/kernel")]:
        if root.exists():
            cands.extend(Path(p) for p in glob.glob(str(root / "**/sem_s_2048.pt"), recursive=True))
    if not cands:
        raise FileNotFoundError("sem_s_2048.pt not found")
    return cands[0]


def _add_package_path() -> None:
    cands = []
    for root in [Path("/kaggle/input"), HERE, Path("/workspace/fil/r16/dataset"), Path("/workspace/fil/r16/kernel")]:
        if root.exists():
            cands.extend(Path(p) for p in glob.glob(str(root / "**/filaments/__init__.py"), recursive=True))
    if cands:
        sys.path.insert(0, str(cands[0].parent.parent))


def main() -> None:
    t0 = time.time()
    done = {
        "round": "r16",
        "on_kaggle": ON_KAGGLE,
        "data": str(DATA),
        "out": str(OUT),
        "weights_path": None,
        "weights_sha256": None,
        "n_holdout_files": len(FILES),
        "n_holdout_predicted": 0,
        "n_holdout_nonempty": 0,
        "n_test_images": 0,
        "n_test_instances": 0,
        "device": None,
        "ok": False,
        "error": None,
        "true_rsi": False,
        "true_agi": False,
    }
    try:
        assert hashlib.sha256(json.dumps(FILES).encode()).hexdigest() == EXPECTED_FILES_SHA, "FILES_SHA_MISMATCH"
        weights = _find_weights()
        done["weights_path"] = str(weights)
        wsha = _sha256(weights)
        done["weights_sha256"] = wsha
        assert wsha == WEIGHTS_SHA, ("WEIGHTS_SHA_MISMATCH", wsha)
        _add_package_path()
        if ON_KAGGLE:
            rc = os.system("pip -q install 'dfine-seg==0.7.1'")
            if rc != 0:
                raise RuntimeError(f"pip dfine-seg failed rc={rc}")
        from filaments.inference import FilamentSegmenter, PostProcess, write_submission
        import cv2
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        done["device"] = device
        post = PostProcess(thresh=0.4, close=4, min_area=200, min_score=0.90)
        seg = FilamentSegmenter(weights, post=post, tta=False, device=device)

        train_dir = DATA / "train/train_images"
        hold = {}
        missing = []
        for stem in FILES:
            p = train_dir / f"{stem}.jpeg"
            if not p.exists():
                missing.append(stem)
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            hold[stem] = seg(img)
        if missing:
            raise FileNotFoundError(f"missing holdout jpegs: {missing}")
        done["n_holdout_predicted"] = len(hold)
        done["n_holdout_nonempty"] = sum(1 for s in FILES if len(hold[s]) > 0)
        with (OUT / "holdout_inst_r16.pkl").open("wb") as f:
            pickle.dump(hold, f, protocol=4)

        test_dir = DATA / "test/test_images"
        test_paths = sorted(test_dir.glob("*.jpeg"))
        assert len(test_paths) == 180, len(test_paths)
        preds = {p.stem: seg(cv2.imread(str(p), cv2.IMREAD_COLOR)) for p in test_paths}
        done["n_test_images"] = len(preds)
        done["n_test_instances"] = int(sum(len(v) for v in preds.values()))
        write_submission(preds, OUT / "submission.csv")
        done["ok"] = (
            done["n_holdout_predicted"] == 25
            and done["n_holdout_nonempty"] >= 20
            and done["n_test_images"] == 180
        )
    except Exception as e:
        done["error"] = f"{type(e).__name__}: {e}"
        done["traceback"] = traceback.format_exc()
        print("R16_FAIL", done["error"], flush=True)
        print(done.get("traceback"), flush=True)
    done["wall_s"] = round(time.time() - t0, 2)
    (OUT / "done_r16.json").write_text(json.dumps(done, indent=2) + "\n")
    print("R16DONE", json.dumps({k: v for k, v in done.items() if k != "traceback"}), flush=True)
    if not done.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
