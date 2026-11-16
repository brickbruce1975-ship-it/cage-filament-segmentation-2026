# CAGE — Solar Filament Segmentation Challenge 2026

Code for our Kaggle entries in the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(team: James Mills, Kaggle `jamesmills1975`).

The competition closed Mon Nov 16, 2026, 5:59:59 AM CT (11:59:59 UTC). Final standing: **391 of 1,567 teams**.

We submitted two files:

| entry | method | Kaggle ref | public PQ | private PQ |
|---|---|---|---|---|
| **r16** | frozen public D-FINE-seg-S (Apache-2.0) | **58424634** | **0.38** | **0.36** |
| r13 | U-Net + YOLO11s-seg fusion | 58336268 | 0.35 | 0.33 |

The public +0.03 of r16 over r13 held as +0.03 on the private test. r16 was submitted only after James overrode a pre-registered Gate 2 (margin) BLOCK. See [Results](#results) for the honest holdout numbers. The technical report is in [`report/`](report/) (`main.tex` and the figure PNGs).

## Layout

| path | what |
|---|---|
| `kaggle/full_unet_r8/` | Kaggle GPU script that trains the U-Nets used for the **test** predictions: `base_mask.pt` (r5 mask recipe, 22 ep, seed 5) and `hull.pt` (16 ep, seed 6), 1024 px whole image. (It also trains an `r8_mask` tile model that the final pipeline does not use.) |
| `kaggle/full_yolo_r10/` | Kaggle GPU script that trains YOLO11s-seg (imgsz 2048, 40 ep, batch 8 DDP on 2×T4, seed 0) -> `runs/r10/weights/best.pt`, used for the **test** predictions. |
| `kaggle/r13_unet_holdout/` | r13 confirmatory run: same U-Net recipe retrained without the fresh holdout; writes raw holdout probabilities `holdout_probs_r13.npz`. |
| `kaggle/r13_yolo_holdout/` | r13 confirmatory run: same YOLO recipe retrained without the fresh holdout; writes `holdout_inst_r13.pkl`. |
| `kaggle/r16_dfine/` | r16 inference-only Kaggle script (`jamesmills1975/cage-filament-r16-dfine` v2): frozen ArgoHA D-FINE-seg-S + frozen post-processing. Includes an unmodified copy of that repo's `filaments` package (Apache-2.0). Weights are linked, not committed. |
| `pipeline/common.py` | data loading and the file-grouped splits (sha256-asserted). |
| `pipeline/analyze.py` | `labels_B`: U-Net "config B" instance grouping (th 0.5, hull th 0.3, close 51, minA 250). |
| `pipeline/r12_fuse.py` | U-Net + YOLO fusion (`prep_file`, `fuse`). |
| `pipeline/selfeval.py` | port of the organizers' self-evaluation PQ (`get_pq_score`, IoU > 0.5). |
| `pipeline/cache_test.py` | inference on the 180 test images with the full models (U-Net probs with hflip TTA, YOLO conf >= 0.05, NMS 0.7, max_det 100, retina masks). |
| `r13/split13.py`, `r13/fresh13_holdout200.json` | the fresh file-grouped holdout (201 ids on 118 files; image ids only, no data). |
| `r13/score_r13.py` | scores incumbent (U-Net only) vs. fused candidate once on the fresh holdout. |
| `r13/make_sub_r13.py` | builds `submission.csv` from the cached test outputs with `fuse(P, 0.2, None, 0.15, 5000)`. |
| `r16/holdout46_files.json`, `r16/README.md` | the 25 leak-free file stems used to score r16 (ids only), and how to download the frozen weights. |
| `report/` | team report in the official ACM template: `main.tex` plus `fig0_pipeline.png`, `fig1_iou_dice_hist.png`, `fig2_overlay_median.png`, `fig3_pq_ci.png`. |

No data, weights, or Kaggle credentials are in this repo.

## How to reproduce r13 (U-Net + YOLO fusion)

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

## How to reproduce r16 (frozen D-FINE-seg-S)

This entry did not train anything. It ran the published ArgoHA weights once.

1. Download `sem_s_2048.pt` from [ArgoHA/solar-filament-dfine-seg](https://github.com/ArgoHA/solar-filament-dfine-seg) and check sha256 `11958712586f000a164c07bf77164ee39ed281bbb99b65a581e2da04a548cff8`. Steps are in [`kaggle/r16_dfine/WEIGHTS.md`](kaggle/r16_dfine/WEIGHTS.md).
2. `pip install 'dfine-seg==0.7.1'` (and torch, opencv-python, pycocotools).
3. Point `kaggle/r16_dfine/fil.py` at the unzipped competition data (or keep the `/workspace/fil/MAGFiLO_1.0_Kaggle_2026` layout) and run `python kaggle/r16_dfine/fil.py`.

Post-processing is frozen as published: threshold `p >= 0.4`, close with a disk of radius 4, drop components `< 200` px, keep a component only if its mean probability is `>= 0.90`. Details: [`r16/README.md`](r16/README.md).

The submitted CSV sha256 is `7566598e23a7394ab0daf39e32dcf390b89c61f814b027ba780efd57b95e166a` (the file is not in this repo).

## Results

### r13 — U-Net + YOLO fusion

Method: a U-Net semantic pipeline (mask net + "hull" grouping net, r5 recipe) fused with a YOLO11s-seg instance model
(r10 recipe). U-Net instances are kept where YOLO agrees (agreement threshold 0.2, YOLO conf >= 0.25), and small YOLO-only
instances (conf >= 0.15, area < 5,000 px) are added (frozen r12 fusion recipe).

Fresh file-grouped holdout (201 ids, never used for selection), scored once with the self-evaluation port:

| method | PQ |
|---|---|
| U-Net only (r5 recipe, config B) | 0.3846 |
| **U-Net + YOLO fusion (r13)** | **0.4092** |

Gain +0.0246 vs. a pre-registered bar of +0.015; paired per-image bootstrap 95% lower bound +0.020.
Public leaderboard: **0.35** for r13 (our earlier U-Net-only entries scored 0.33-0.34).

### r16 — frozen public D-FINE-seg-S (override submit)

Family switch after r14 and r15 missed their bars. Inference only, one T4 session, kernel `jamesmills1975/cage-filament-r16-dfine` v2, frozen weights `sem_s_2048.pt` (sha256 `11958712586f000a164c07bf77164ee39ed281bbb99b65a581e2da04a548cff8`).

The published weights were trained on 80% of observation dates, so they cannot be scored on all 201 r13 holdout ids without leakage. r16 was scored once on the leak-free intersection of that holdout with the authors' held-out dates: **46 ids / 25 files**.

| method (same 46 ids) | PQ |
|---|---|
| r13 recipe | 0.40384 |
| **frozen D-FINE-seg-S (r16)** | **0.43205** |

- Gate 1 (point-estimate) **PASS**: gain +0.02821 vs. a bar of +0.015 (TP 201→205, FP 157→123, FN 112→108).
- Gate 2 (margin, inherited k=4) **BLOCK**: mean paired delta +0.0325, 98.75% bootstrap lower bound −0.0118 (below 0). n=46 is too small at this reuse level.
- James overrode the Gate 2 block (card t3577s17648) and the file was submitted through `submit_once.sh`. Public PQ **0.38** (Kaggle ref **58424634**), vs. 0.35 for r13. Submission CSV sha256: `7566598e23a7394ab0daf39e32dcf390b89c61f814b027ba780efd57b95e166a`.
- After close (Mon Nov 16, 2026, 5:59:59 AM CT / 11:59:59 UTC), private PQ was **0.36** for r16 vs. **0.33** for r13 (ref 58336268). The public +0.03 held as +0.03 private. Final rank: **391 of 1,567 teams**.
- That override did not change our two-pick final-selection settings.

## Later rounds (honest NO_PASS notes)

We kept trying after r13. r14 and r15 did not pass their pre-registered bars and were **not** submitted. r16 is the only later file that was submitted, and only under the override above.

- **r14 (NO_PASS):** YOLO11m-seg in place of YOLO11s-seg inside the same fusion. Holdout PQ 0.4061 vs. 0.4092 for r13 (gain -0.0031, bar +0.015). Not submitted.
- **r15 (NO_PASS):** a crop U-Net refiner on the r13 instances. Holdout PQ 0.4140 vs. 0.4092 (gain +0.0048, bar +0.015; paired bootstrap lower bound below 0). Not submitted.
- **r16 (Gate 1 PASS, Gate 2 BLOCK, submitted under override):** frozen public D-FINE-seg-S, as above.

Other things that did not help in earlier rounds: Mask R-CNN (below the U-Net), marker/watershed splitting (over-splitting hurts PQ), and a
shared 3-head U-Net (worse masks than separate nets).

## Licence

The r13 fusion code in this repository is released under the **GNU Affero General Public License v3.0** (see `LICENSE`), because it uses
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Other r13 dependencies: segmentation-models-pytorch (MIT), timm (Apache-2.0).

The r16 inference path includes an unmodified copy of the `filaments` package and Apache-2.0 license text from
[ArgoHA/solar-filament-dfine-seg](https://github.com/ArgoHA/solar-filament-dfine-seg) (see `kaggle/r16_dfine/LICENSE.solar-filament-dfine-seg`).
Those files remain Apache-2.0. The frozen weights are not in this repo; download them from that project and verify the sha256 in
`kaggle/r16_dfine/WEIGHTS.md`. The D-FINE-seg framework is also Apache-2.0 ([ArgoHA/D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg)).

The competition data (MAGFiLO, doi:10.1038/s41597-024-03876-y) is CC BY-NC 4.0 and is not included here.
