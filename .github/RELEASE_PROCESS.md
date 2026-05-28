# Release process

This repo uses [GitHub releases](https://github.com/danghoangnhan/geofm/releases) +
[Zenodo](https://zenodo.org) to mint a citable DOI per tag. Pretty much
follows the Citation File Format (CFF) recipe — but documented in one place so
the next release isn't a treasure hunt through six tabs.

## One-time setup (skip if `https://zenodo.org/account/settings/github/` already
lists the repo with the toggle ON)

1. Sign into Zenodo with the GitHub identity that owns the repo.
2. Go to **Settings → GitHub** and flip the **danghoangnhan/geofm** toggle ON.
3. Trigger a release (next section). Zenodo will mint a DOI within a couple of
   minutes.

## Per-release checklist

1. **Bump the version** in three places (search-and-replace):
   * `pyproject.toml::project.version`
   * `CITATION.cff::version` (add the field if it isn't there yet)
   * `README.md` — the Zenodo DOI badge (auto-updated after step 5 — only edit
     if it gets stale).
2. **Tag the SHA**:
   ```bash
   git tag -a v0.2.0 -m "v0.2.0 — add Phase 9 ablation harness + 7 notebooks"
   git push origin v0.2.0
   ```
   Use semver. Patch bumps for docs / notebook re-runs; minor bumps for new
   features (new notebooks, new ablation axes); major bumps for breaking
   changes to the CLI / dataset schema.
3. **Create the GitHub release** from the tag:
   ```bash
   gh release create v0.2.0 \
       --title "v0.2.0 — Phase 9 ablation + 7 notebooks" \
       --notes-file .github/release-notes/v0.2.0.md
   ```
   Or use the web UI: **Releases → Draft a new release → choose tag**. Keep
   release notes short (5-15 bullets) and link to the CHANGELOG entry if
   one exists.
4. **Wait ~2 minutes** for Zenodo to pick the release up and mint the DOI.
5. **Copy the DOI badge** Zenodo gives you into `README.md` (right under the
   Apache-2.0 / Python / arXiv badges). The badge URL is stable across
   versions — you only edit it on the first release.
6. **Update the wiki** if any user-facing surface changed (CLI flags, dataset
   schema fields, supported configs).

## Release notes — what to include

* **Headline** — one-line summary.
* **What's new** — bullets, oriented around user-visible surface: new CLI
  commands, new configs, new notebooks, new tests.
* **Eval deltas** — if a real VLMEvalKit run was reproduced, link the
  ablation table (`outputs/ablate/scale_*.md`) so readers can see the +pp
  movement.
* **Blackwell stack snapshot** — pin the working
  PyTorch / FlashAttention / bnb / vLLM versions at the time of the release,
  matching `wiki/07-Blackwell-Setup-Log.md`.
* **Breaking changes** — flag them with ⚠️. The point of semver is the user
  shouldn't have to read the diff.
* **Acknowledgements** — name the contributors and the upstream packages.

## How tags compose with the dataset releases

Dataset versions on the Hub (`open-geofm-mini-5k`, `-10k`, `-20k`) are
**independent** of code tags — each dataset repo carries its own version
field. Convention: bump the dataset version whenever any of
`{algorithm1 swap RNG seed, gather_metric_info depth, renderer, NLG rewriter
backend}` changes, *not* on every code release. Document the binding in the
dataset card.

## Adapter releases

LoRA adapters land at:
* `danghoangnhan/Qwen2-VL-2B-OpenGeoFM-LoRA` (headline)
* `danghoangnhan/Qwen2-VL-7B-OpenGeoFM-LoRA`
* `danghoangnhan/Qwen2.5-VL-7B-OpenGeoFM-LoRA`

Each Hub release should pin the matching open-geofm code tag in the model
card's `model_creator` / `base_model` fields so users can reproduce the LoRA
delta deterministically.

## CI gate

`.github/workflows/ci.yml` runs ruff + pytest on every push to `main` and
every PR. A release should only be cut when CI is green on the SHA being
tagged. Don't push the tag before the green check — Zenodo archives the
*exact* SHA, so a yellow / red tagged commit produces a bad citation.
