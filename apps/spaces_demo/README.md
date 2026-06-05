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

Gradio app for the [open-geofm](https://github.com/danghoangnhan/open-geofm)
educational reproduction of *GeoFM* (arXiv:2510.27448).

> **This file is the Hugging Face Space config card** — the YAML frontmatter
> above is required for the Space to build. Full demo documentation
> (configuration, hardware, local preview, deploy) lives in the wiki:
> [**Spaces-Demo**](https://github.com/danghoangnhan/open-geofm/wiki/Spaces-Demo)
> (source: [`wiki/Spaces-Demo.md`](../../wiki/Spaces-Demo.md)).

## Quick start

```bash
cd apps/spaces_demo
uv pip install -r requirements.txt
python app.py
```

Configure via the Space's environment variables: `OPEN_GEOFM_BASE_MODEL`
(default `Qwen/Qwen2-VL-2B-Instruct`), `OPEN_GEOFM_LORA_ID`, and
`OPEN_GEOFM_MAX_NEW_TOKENS` (default `512`). The app degrades gracefully on CPU.
