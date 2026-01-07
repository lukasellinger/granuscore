# Granuscore

**Granuscore** is a Python library for measuring the *granularity* of natural language text.

It provides an end-to-end pipeline that:
1. splits text into referential units,
2. assigns a continuous granularity score to each unit, and
3. aggregates these scores to obtain a document-level measure.

---

## Installation

Clone the repository and install the package locally:

```bash
pip install -e .
```

### Optional Dependencies

Additional dependencies for experiments and development can be installed via:

```bash
pip install -e ".[dev]"
```

---

## Example Usage

A hands-on introduction to Granuscore is provided in  
[`getting_started.ipynb`](./notebooks/getting_started.ipynb), which demonstrates the main pipeline and expected outputs.

---

## Repository Structure

The repository structure below highlights the most important components of the project.

```
granuscore/
├── src/
│   └── granuscore/                   # Core Python package
│       ├── assets/                   # Reference distributions, anchors, and FAISS index
│       ├── pipeline.py               # Main Granuscore pipeline
│       └── granularity_predictor.py  # Granularity predictors (e.g., HiT, SentenceTransformer, LLM)
├── training_scripts/                 # Training scripts for the LGB model and predictor evaluation on Granola-EQ
├── evaluation/                       # Experiments and ablations (QA analysis, sentence specificity, paper sections)
├── notebooks/
│   └── getting_started.ipynb         # Interactive introduction to Granuscore
├── pyproject.toml                    # Package configuration
└── README.md
```

## Reproducing Paper Experiments

To reproduce the experiments reported in the paper, the data must be placed in the expected location.

Unzip the provided archive in the repository root:

```bash
unzip data.zip
```

## Training Pipeline

Training uses precomputed `.pkl` files for the train, validation, and test splits.

To train a model:

1. Generate the precomputed datasets using one of the scripts in  
   `training_scripts/build_precalc_data/`.
2. Train the model by running `training_scripts/train_lgb_models.py`.
