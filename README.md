# CAGE — Solar Filament Segmentation Challenge 2026 (r13 final submission)

Code for our final Kaggle entry in the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(team: James Mills, Kaggle `jamesmills1975`). Final submission **58336268 ("r13")**, **public PQ 0.35**.

Method: a U-Net semantic pipeline (mask net + "hull" grouping net, r5 recipe) fused with a YOLO11s-seg instance model
(r10 recipe). U-Net instances are kept where YOLO agrees (agreement threshold 0.2, YOLO conf >= 0.25), and small YOLO-only
instances (conf >= 0.15, area < 5,000 px) are added (frozen r12 fusion recipe). Details are in our technical report.

## Layout

| path | what |
|---|---|
| `kaggle/full_unet_r8/` | Kaggle GPU script that trains the U-Nets used for the **test** predictions: `base_mask.pt` (r5 mask recipe, 22 ep, seed 5) and `hull.pt` (16 ep, seed 6), 1024 px whole image. (It also trains an `r8_mask` tile model that the final pipeline does not use.) |
| `kaggle/full_yolo_r10/` | Kaggle GPU script that trains YOLO11s-seg (imgsz 2048, 40 ep, batch 8 DDP on 2×T4, seed 0) -> `runs/r10/weights/best.pt`, used for the **test** predictions. |
| `kaggle/r13_unet_holdout/` | r13 confirmatory run: same U-Net recipe retrained without the fresh holdout; writes raw holdout probabilities `holdout_probs_r13.npz`. |
| `kaggle/r13_yolo_holdout/` | r13 confirmatory run: same YOLO recipe retrained without the fresh holdout; writes `holdout_inst_r13.pkl`. |
| `pipeline/common.py` | data loading and the file-grouped splits (sha256-asserted). |
| `pipeline/analyze.py` | `labels_B`: U-Net "config B" instance grouping (th 0.5, hull th 0.3, close 51, minA 250). |
| `pipeline/r12_fuse.py` | U-Net + YOLO fusion (`prep_file`, `fuse`). |
| `pipeline/selfeval.py` | port of the organizers' self-evaluation PQ (`get_pq_score`, IoU > 0.5). |
| `pipeline/cache_test.py` | inference on the 180 test images with the full models (U-Net probs with hflip TTA, YOLO conf >= 0.05, NMS 0.7, max_det 100, retina masks). |
| `r13/split13.py`, `r13/fresh13_holdout200.json` | the fresh file-grouped holdout (201 ids on 118 files; image ids only, no data). |
| `r13/score_r13.py` | scores incumbent (U-Net only) vs. fused candidate once on the fresh holdout. |
| `r13/make_sub_r13.py` | builds `submission.csv` from the cached test outputs with `fuse(P, 0.2, None, 0.15, 5000)`. |

No data, weights, or Kaggle credentials are in this repo.

## How to reproduce

The scripts were run as-is with absolute paths from our working machine. To rerun them, recreate that layout (or edit the path constants at the top of each script):

```
/workspace/fil/MAGFiLO_1.0_Kaggle_2026/   <- competition data (unzipped filament-segmentation-2026.zip)
/workspace/fil/r11/  <- pipeline/common.py, analyze.py, selfeval.py, cache_test.py
/workspace/fil/r12/  <- pipeline/r12_fuse.py
/workspace/fil/r13/  <- r13/*
/workspace/fil/r8/out_f/base_mask.pt, hull.pt          <- outputs of kaggle/full_unet_r8
/workspace/fil/r10/out_f/runs/r10/weights/best.pt      <- output of kaggle/full_yolo_r10
```

1. **Train the full models (Kaggle, free T4 GPUs).** Push each folder with `kaggle kernels push -p kaggle/full_unet_r8` and
   `kaggle kernels push -p kaggle/full_yolo_r10` (change the `id` in `kernel-metadata.json` to your own user). Download the outputs.
2. **Test inference (CPU is fine).** `python pipeline/cache_test.py` writes one cache file per test image to `/dev/shm/fil_r11/test/`.
3. **Fuse and write the submission.** `python r13/make_sub_r13.py` -> `/workspace/fil/r13/submission.csv` (180 test images, one COCO RLE row per instance).
4. **Optional: the r13 validation.** Push `kaggle/r13_unet_holdout` and `kaggle/r13_yolo_holdout`, download their outputs, then
   `python r13/score_r13.py <unet_out_dir> <yolo_out_dir>`.

Environment used: Python 3, torch 2.x, segmentation-models-pytorch 0.5.0, timm 1.0.x, ultralytics 8.4.212, pycocotools 2.0.x,
opencv-python, numpy, pandas, scikit-image. Seeds: U-Nets 5 (mask) and 6 (hull), YOLO 0; splits use RandomState 0/8/10/13.
GPU training is not bit-for-bit deterministic, so retrained models can give slightly different numbers.

## Results

Fresh file-grouped holdout (201 ids, never used for selection), scored once with the self-evaluation port:

| method | PQ |
|---|---|
| U-Net only (r5 recipe, config B) | 0.3846 |
| **U-Net + YOLO fusion (r13)** | **0.4092** |

Gain +0.0246 vs. a pre-registered bar of +0.015; paired per-image bootstrap 95% lower bound +0.020.
Public leaderboard: **0.35** for r13 (our earlier U-Net-only entries scored 0.33-0.34).

## Later rounds (honest NO_PASS notes)

We kept trying after r13. Neither later round passed its pre-registered bar, so **nothing after r13 was submitted** and r13 is our final pick.

- **r14 (NO_PASS):** YOLO11m-seg in place of YOLO11s-seg inside the same fusion. Holdout PQ 0.4061 vs. 0.4092 for r13 (gain -0.0031, bar +0.015). Not submitted.
- **r15 (NO_PASS):** a crop U-Net refiner on the r13 instances. Holdout PQ 0.4140 vs. 0.4092 (gain +0.0048, bar +0.015; paired bootstrap lower bound below 0). Not submitted.

Other things that did not help in earlier rounds: Mask R-CNN (below the U-Net), marker/watershed splitting (over-splitting hurts PQ), and a
shared 3-head U-Net (worse masks than separate nets).

## Licence

This code is released under the **GNU Affero General Public License v3.0** (see `LICENSE`), because it uses
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Other dependencies: segmentation-models-pytorch (MIT), timm (Apache-2.0).
The competition data (MAGFiLO, doi:10.1038/s41597-024-03876-y) is CC BY-NC 4.0 and is not included here.
