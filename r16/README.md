# r16 — frozen public D-FINE-seg-S

Kaggle kernel `jamesmills1975/cage-filament-r16-dfine` version 2 (private script,
one T4, internet on to install `dfine-seg==0.7.1`). Weights and the `filaments`
package were attached as dataset `jamesmills1975/cage-filament-r16-dfine-w`.

This was an **inference-only** run of frozen public weights. No training, no
threshold retuning, no test-time augmentation.

## Credit

Frozen weights and the `filaments` inference package come from
[ArgoHA/solar-filament-dfine-seg](https://github.com/ArgoHA/solar-filament-dfine-seg)
(Apache-2.0, public 14 Nov 2026). See `kaggle/r16_dfine/WEIGHTS.md` and
`kaggle/r16_dfine/LICENSE.solar-filament-dfine-seg`. The framework is
[D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg).

## Post-processing (frozen, as published)

1. Threshold `p(filament) >= 0.4`
2. Morphological closing with a disk of radius 4
3. Drop 8-connected components smaller than 200 px
4. Keep a component only if its mean probability is `>= 0.90`

Set in the kernel as `PostProcess(thresh=0.4, close=4, min_area=200, min_score=0.90)`.

## Result

| | PQ |
|---|---|
| r13 recipe on the same 46 leak-free ids | 0.40384 |
| r16 frozen D-FINE-seg-S on those 46 ids | 0.43205 |
| Point-estimate gain (Gate 1 bar +0.015) | +0.02821 **PASS** |
| Margin gate k=4, 98.75% bootstrap lower bound | −0.0118 **BLOCK** |
| Public leaderboard (Kaggle ref **58424634**) | **0.38** (r13 was 0.35) |

Submission CSV sha256: `7566598e23a7394ab0daf39e32dcf390b89c61f814b027ba780efd57b95e166a`.

The 46-id slice (`r16/holdout46_files.json`, 25 file stems) is the leak-free
intersection of the r13 file-grouped holdout with the dates held out of the
published D-FINE training. Gate 2 blocked the submit because n=46 is too small
at inherited reuse k=4. James overrode that block (card t3577s17648: "Submit
Filament r16 D-FINE through submit_once.sh anyway, overriding the noise block").
The file was submitted through `submit_once.sh`, not the raw Kaggle CLI. Final
two-pick selection was not changed by that submit.

The CSV itself is not in this repo (same policy as r13: no data or weights).

## How to run

1. Download and hash-check the weights (`kaggle/r16_dfine/WEIGHTS.md`).
2. Unzip the competition data so `MAGFiLO_1.0_Kaggle_2026/` is on the path the
   kernel expects, or edit the path constants at the top of
   `kaggle/r16_dfine/fil.py` (same convention as the r13 scripts).
3. `pip install 'dfine-seg==0.7.1' pycocotools opencv-python torch` (plus the
   rest of `dfine-seg`'s deps).
4. `python kaggle/r16_dfine/fil.py` writes `submission.csv`. A GPU is optional
   but was used on Kaggle (T4, 409 s for 25 holdout files + 180 test images).

`kaggle/r16_dfine/fil.py` is the kernel source that produced the submitted file.
