# Frozen public D-FINE-seg-S weights (not in this repo)

The r16 entry used the frozen public checkpoint `sem_s_2048.pt` from
[ArgoHA/solar-filament-dfine-seg](https://github.com/ArgoHA/solar-filament-dfine-seg)
(Apache-2.0). Do not commit the file here.

- File: `weights/sem_s_2048.pt` (32,302,363 bytes)
- sha256: `11958712586f000a164c07bf77164ee39ed281bbb99b65a581e2da04a548cff8`
- Source repo: https://github.com/ArgoHA/solar-filament-dfine-seg
- Direct file: https://github.com/ArgoHA/solar-filament-dfine-seg/raw/main/weights/sem_s_2048.pt
- Framework: [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) (`dfine-seg==0.7.1`)

Download and verify:

```bash
mkdir -p kaggle/r16_dfine/weights
curl -L -o kaggle/r16_dfine/weights/sem_s_2048.pt \
  https://github.com/ArgoHA/solar-filament-dfine-seg/raw/main/weights/sem_s_2048.pt
sha256sum kaggle/r16_dfine/weights/sem_s_2048.pt
# must be 11958712586f000a164c07bf77164ee39ed281bbb99b65a581e2da04a548cff8
```

The `filaments/` package in this folder is an unmodified copy of the upstream
package (same sha256 as the 2026-11-14 public tree). Its Apache-2.0 license is
in `LICENSE.solar-filament-dfine-seg`.
