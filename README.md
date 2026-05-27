[![PyPI version](https://img.shields.io/pypi/v/granuscore.svg)](https://pypi.org/project/granuscore/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/lukasellinger/granuscore)
# Granuscore

**Granuscore** is a Python library for measuring the *semantic granularity* of natural language text.

It provides an end-to-end pipeline that:

1. splits text into referential units,
2. assigns continuous granularity scores to each unit,
3. aggregates these scores into document-level estimates.

Granuscore is designed for analyzing how *fine-grained* or *coarse-grained* textual expressions are in applications such as question answering, educational dialogue, summarization, and scientific writing.

---

## Installation

Install from PyPI:

```bash
pip install granuscore
```

Or install the latest development version locally:

```bash
git clone https://github.com/lukasellinger/granuscore.git
cd granuscore
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from granuscore import GranuScore

scorer = GranuScore()

text = """
Tony Hawk was born in San Diego.
"""

score = scorer(text)

print(score)
```

By default, Granuscore returns percentile scores, where higher values correspond to coarser-grained expressions.

---

## Default Configuration

The default configuration reproduces the setup used in the paper.

```python
scorer = GranuScore()
```

Equivalent to:

```python
scorer = GranuScore(
    predictor_type="hit",
)
```

Default settings:

- `predictor_type="hit"`
- `model_name="Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"`
- `search_method="random_anchors"`
- `random_anchors_k=999`

Required artifacts such as:
- FAISS indices,
- anchor vectors,
- LightGBM models,
- and reference percentile distributions

are automatically downloaded and cached on first use.

---

## Important Compatibility Note

The default configuration works out of the box and is the recommended setup.

If you customize components such as:
- the embedding model,
- search method,
- FAISS index,
- anchor vectors,
- or LightGBM model,

you must ensure that all resources are compatible with each other.

For example, a LightGBM model trained using:

```python
search_method="random_anchors"
```

should not be combined with:

```python
search_method="nearest_neighbor"
```

Similarly, FAISS indices, anchor vectors, percentile reference distributions, and LightGBM models must originate from the same embedding space and training configuration.

Compatibility between custom resources is not validated automatically.

---

## Notebook Tutorial

An interactive introduction is available in:

```text
notebooks/getting_started.ipynb
```

---

## Repository Structure

```text
granuscore/
├── src/
│   └── granuscore/
│       ├── pipeline.py
│       ├── granularity_predictor.py
│       ├── claim_splitter.py
│       ├── bucket_output.py
│       ├── cache.py
│       └── artifacts.py
├── notebooks/
│   ├── build_granola_dataset.ipynb
│   └── getting_started.ipynb
├── training_scripts/
├── evaluation/
├── assets/
├── data/ (needs to be externally downloaded)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Reproducing Paper Experiments

The datasets and precomputed resources required to reproduce the experiments from the paper are available here:

https://drive.google.com/drive/folders/1mJdUENOxHEiuYn-_f1KRQ1PZggXJDnb4?usp=sharing

Download the archive and extract it into the repository root:

```bash
unzip data.zip
```

This will create the expected directory structure used by the training and evaluation scripts.

---

---

## GRANOLA Dataset Construction

We use a processed version of the [GRANOLA-EQ Dataset](https://huggingface.co/datasets/google/granola-entity-questions) introduced by [Yona et al. (2024)](https://aclanthology.org/2024.acl-long.365/). Due to licensing restrictions of upstream resources, the processed dataset version used in this work is not redistributed directly in this repository or on external hosting platforms.

Instead, the dataset can be reconstructed locally using the notebook:

```text
notebooks/build_granola_dataset.ipynb
```

---

## Training Pipeline

Training uses precomputed `.pkl` feature files.

1. Generate precomputed datasets:

```text
training_scripts/build_precalc_data/
```

2. Train LightGBM models:

```bash
python training_scripts/train_lgb_models.py
```

---

## Citation

```bibtex
@misc{ellinger2026granuscorereferencefreemeasuregranularity,
      title={Granuscore: A Reference-Free Measure of Granularity for Text Analysis and Question Answering}, 
      author={Lukas Ellinger and Alexander Fichtl and Miriam Anschütz and Georg Groh},
      year={2026},
      eprint={2605.26620},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.26620}, 
}
```