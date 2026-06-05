# open-geofm

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![Paper: arXiv:2510.27448](https://img.shields.io/badge/arXiv-2510.27448-b31b1b.svg)](https://arxiv.org/abs/2510.27448)

> **Educational reproduction — not the official code.**
> A single-GPU, LoRA-only re-implementation of the data-generation methodology of *GeoFM*. Not affiliated with the authors; the original paper has no public code release.

---

## Abstract

> *From the original paper — [**GeoFM: Enhancing Geometric Reasoning of MLLMs via Synthetic Data Generation through Formal Language**](https://arxiv.org/abs/2510.27448) (Zhang, Hu, Yu, Liu, and Liu, 2025, arXiv:2510.27448):*

> Multi-modal Large Language Models (MLLMs) have gained significant attention in both academia and industry for their capabilities in handling multi-modal tasks. However, these models face challenges in mathematical geometric reasoning due to the scarcity of high-quality geometric data. To address this issue, synthetic geometric data has become an essential strategy. Current methods for generating synthetic geometric data involve rephrasing or expanding existing problems and utilizing predefined rules and templates to create geometric images and problems. However, these approaches often produce data that lacks diversity or is prone to noise. Additionally, the geometric images synthesized by existing methods tend to exhibit limited variation and deviate significantly from authentic geometric diagrams. To overcome these limitations, we propose GeoFM, a novel method for synthesizing geometric data. GeoFM uses formal languages to explore combinations of conditions within metric space, generating high-fidelity geometric problems that differ from the originals while ensuring correctness through a symbolic engine. Experimental results show that our synthetic data significantly outperforms existing methods. The model trained with our data surpass the proprietary GPT-4o model by 18.7% on geometry problem-solving tasks in MathVista and by 16.5% on GeoQA. Additionally, it exceeds the performance of a leading open-source model by 5.7% on MathVista and by 2.7% on GeoQA.

---

## Documentation

Full project documentation lives in the **[wiki](https://github.com/danghoangnhan/open-geofm/wiki)** (mirrored in [`wiki/`](wiki/)):

- [Project README / quickstart](wiki/README.md)
- [Overview](wiki/00-Overview.md) · [Formal Language](wiki/01-Formal-Language.md) · [Condition Sampling](wiki/02-Condition-Sampling.md) · [Symbolic Verification](wiki/03-Symbolic-Verification.md)
- [Diagram Rendering](wiki/04-Diagram-Rendering.md) · [Qwen2-VL Fine-tuning](wiki/05-Qwen2VL-Finetuning.md) · [Evaluation](wiki/06-Evaluation.md) · [Blackwell Setup Log](wiki/07-Blackwell-Setup-Log.md)
- [Contributing](wiki/CONTRIBUTING.md)

---

## Citation

Please cite **both** the original paper and this repository.

```bibtex
@article{zhang2025geofm,
  title  = {GeoFM: Enhancing Geometric Reasoning of MLLMs via Synthetic Data Generation through Formal Language},
  author = {Zhang, Yuhao and Hu, Dingxin and Yu, Tinghao and Liu, Hao and Liu, Yiting},
  journal= {arXiv:2510.27448},
  year   = {2025}
}

@software{open_geofm,
  title  = {open-geofm: An educational reproduction of GeoFM on a single RTX 5090},
  author = {open-geofm contributors},
  year   = {2026},
  url    = {https://github.com/danghoangnhan/open-geofm}
}
```

License: Apache-2.0 (see [`LICENSE`](LICENSE)).
