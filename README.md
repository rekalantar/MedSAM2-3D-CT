# MedSAM2 for 3D CT

Segment a whole CT volume by prompting **one slice**.

SAM 2 tracks objects across video frames using memory attention. A CT volume has the
same structure — consecutive slices differ only slightly, exactly like consecutive
frames — so a single bounding box on one slice can be propagated through the entire
stack. [MedSAM2](https://github.com/bowang-lab/MedSAM2) is SAM 2.1 fine-tuned on
medical data so this works on 15-Hounsfield-unit soft tissue contrast rather than
natural-image contrast.

That takes a 94-slice lesion annotation from 94 prompts to one.

![propagation](assets/propagation.gif)

---

## Install

```bash
pip install git+https://github.com/rekalantar/medsam2-3d-ct.git
```

Volume loading needs SimpleITK, kept as an optional extra:

```bash
pip install "medsam2-ct[io] @ git+https://github.com/rekalantar/medsam2-3d-ct.git"
```

MedSAM2 itself and its weights are **not** bundled — they carry their own license:

```bash
git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2
pip install -e ".[dev]"
bash download.sh
```

## Use

```python
from medsam2_ct import load_volume, window_hu, build_predictor, segment_volume, save_gif

volume_hu, spacing = load_volume("scan.nii.gz")        # (z, y, x), mm
volume = window_hu(volume_hu, width=400, level=40)     # HU -> uint8, abdominal window

predictor = build_predictor()
masks = segment_volume(predictor, volume, box=[180, 180, 260, 260], key_slice=47)

save_gif(volume, masks, "propagation.gif")
```

## Tutorial

[`tutorial/medsam2_3d_ct.ipynb`](tutorial/medsam2_3d_ct.ipynb) runs the whole method on a
single case from the MedSAM2 demo dataset — a few MB, fetched automatically. Because that
dataset ships ground-truth masks, the notebook places the prompt from the label and scores
the result, then tests three claims the method rests on:

1. **Windowing changes the result.** Reruns on a naively min-maxed volume and scores the difference.
2. **Backward propagation matters.** Runs forward-only and reports it as a fraction of the bidirectional result.
3. **Accuracy decays with distance from the prompt.** Plots per-slice mask area against distance from the prompted slice.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rekalantar/medsam2-3d-ct/blob/main/tutorial/medsam2_3d_ct.ipynb)

Needs a GPU runtime.

## API

| Function | Does |
|---|---|
| `window_hu(volume, width, level)` | Hounsfield windowing to uint8 |
| `window_preset(volume, preset)` | `abdomen`, `lung`, `brain`, `bone`, `mediastinum` |
| `load_volume(path)` | NIfTI/DICOM → array + spacing, both `(z, y, x)` |
| `build_predictor(config, checkpoint)` | MedSAM2 video predictor |
| `segment_volume(predictor, volume, box, key_slice)` | one box → 3D mask, both directions |
| `largest_component(mask)` | drop propagation leakage |
| `overlay_mask(slice, mask)` | RGB overlay |
| `save_gif(volume, masks, path)` | scrolling animation |

## Two things that silently go wrong

**Windowing.** CT spans several thousand HU; the network takes 8-bit. Min-maxing the
full range leaves soft tissue with almost no dynamic range, and the model then appears
to fail for reasons that have nothing to do with the model. There's a test for this.

**Spacing axis order.** `load_volume` returns `(z, y, x)`. SimpleITK's `GetSpacing()`
returns `(x, y, z)`. Mix them and every physical distance is wrong on anisotropic
scans — no error, perfectly plausible numbers.

## What this does not solve

- **It still needs a human.** One box instead of ninety-four, but finding the lesion was often the hard part.
- **Accuracy decays away from the prompt**, which is exactly where boundary definition matters most for a treatment margin.
- **Diffuse boundaries remain unsolved.** Infiltrative disease gets a confident-looking mask over a region where confidence isn't warranted.
- **Weights are research and education only.** Not cleared for clinical or commercial use.

## Development

```bash
git clone https://github.com/rekalantar/medsam2-3d-ct.git
cd medsam2-3d-ct
pip install -e ".[dev]"
pytest
```

## Reference

Ma, Yang, Kim, Chen, Baharoon, Fallahpour, Asakereh, Lyu & Wang.
[MedSAM2: Segment Anything in 3D Medical Images and Videos](https://arxiv.org/abs/2504.03600), 2025.
Weights: [huggingface.co/wanglab/MedSAM2](https://huggingface.co/wanglab/MedSAM2).

Write-up: **ARTICLE_URL**

MIT licensed. No model weights or patient data are redistributed here.
