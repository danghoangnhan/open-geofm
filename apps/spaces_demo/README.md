---
title: open-geofm Demo
emoji: "\U0001F4D0"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# open-geofm — Hugging Face Spaces demo

Gradio app for the [open-geofm](https://github.com/CallMeDaniel/open-geofm)
educational reproduction of *GeoFM* (Zhang et al., 2025, arXiv:2510.27448).

* Upload (or pick from the example carousel) a geometric figure.
* Type a problem statement.
* Get the model's response.

## Configuration (Space → Settings → Variables)

| Env var | Default | Notes |
|---|---|---|
| `OPEN_GEOFM_BASE_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | Any Qwen2-VL / Qwen2.5-VL checkpoint. |
| `OPEN_GEOFM_LORA_ID` | *(unset)* | Hub repo id for an `open-geofm` LoRA adapter. When set the app applies the adapter via `PeftModel.from_pretrained`. |
| `OPEN_GEOFM_MAX_NEW_TOKENS` | `512` | Generation cap. |

The app degrades gracefully when CUDA isn't available — it surfaces a banner
and the UI still loads, useful for local previews on a CPU box.

## Hardware

* **Required:** free A10G tier (24 GB VRAM) is enough for Qwen2-VL-2B fp16 with
  the LoRA delta merged. Qwen2-VL-7B needs the paid L40s tier.
* **Cold-start:** ~30 s to load weights, ~3 s per inference at 512 max tokens.

## Local preview

```bash
cd apps/spaces_demo
uv pip install -r requirements.txt
python app.py
```

In preview mode (CPU only) the inference function returns a placeholder; the
UI is fully interactive so you can iterate on layout without GPU time.

## Deploy

```bash
# One-time: create the Space.
gh repo create CallMeDaniel/open-geofm-demo --public --description "open-geofm Gradio demo"

# Push.
huggingface-cli repo create open-geofm-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/CallMeDaniel/open-geofm-demo
cp app.py requirements.txt README.md examples/* open-geofm-demo/
cd open-geofm-demo && git add . && git commit -m "deploy" && git push
```

License: Apache-2.0 (inherits from the main repo).
