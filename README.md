# Legal Document Simplification Dataset (LegalSim-PT)

This repository contains code and resources for building and evaluating a large-scale dataset for **legal document simplification in Portuguese**.

**GitHub:** [https://github.com/scalercio/legal-doc-simplification-data](https://github.com/scalercio/legal-doc-simplification-data)

**Hugging Face Dataset:** [https://huggingface.co/datasets/melll-uff/legal-simplification-pt](https://huggingface.co/datasets/melll-uff/legal-simplification-pt)

---

## Overview

This project introduces **LegalSim-PT**, a large-scale dataset for **automatic simplification of legal documents**, containing nearly **1 million document pairs**.

The repository includes:

* Code for generating simplified legal documents using LLMs
* Scripts for fine-tuning models
* Tools for linguistic annotation and evaluation
* Dataset organization and statistics

---

## Dataset

The dataset is publicly available on Hugging Face under:

**`melll-uff/legal-simplification-pt`**

It is released in two configurations:

### 1. `random_split`

Standard machine learning split:

* `train`
* `validation`
* `test`

### 2. `by_source`

Grouped by document origin:

* TCU rulings (*acórdãos*)
* STF decisions
* STF votes and reports
* TJ-SP jurisprudence
* TRF-5 jurisprudence
* Ulysses Tesemõ dataset (subset)

---

## How to Download the Dataset

### Option 1 — Using Hugging Face datasets

```python
from datasets import load_dataset

# Random split version
dataset = load_dataset("melll-uff/legal-simplification-pt", "random_split")

# By-source version
dataset_by_source = load_dataset("melll-uff/legal-simplification-pt", "by_source")
```

### Option 2 — Download manually

You can also download the files directly from the dataset page:

[https://huggingface.co/datasets/melll-uff/legal-simplification-pt](https://huggingface.co/datasets/melll-uff/legal-simplification-pt)

---

## Local Data Organization

The data used in this repository should be placed under:

```text
data/legal-simplification-pt/
├── random_split/
│   ├── train_random.parquet
│   ├── val_random.parquet
│   └── test_random.parquet
└── by_source/
    ├── acordaos_tcu.parquet
    ├── stf_decisions.parquet
    ├── stf_votes.parquet
    ├── tjsp.parquet
    ├── trf5.parquet
    └── ulysses_tesemo.parquet
```

---

## Code Structure

The main scripts are located in the `src/` directory.

### Data Generation

* `src/gen_paraphrases.py`
  Generates simplified legal documents using LLMs.

* `src/gen_paraphrases_gguf.py`
  Generates simplified legal documents using GGUF-based LLMs.

### Model Fine-tuning

* `src/finetune_qwen3.py`
  Fine-tunes the **Qwen3-1.7B** model.

* `src/finetune_chat.py`
  Fine-tunes the **Qwen2.5-7B** model.

### Linguistic Annotation

* `src/annotate_parquet.py`
  Performs morphosyntactic annotation of documents.

### Inference

* `src/eval_qwen2_5.py`
  Runs inference with **Qwen2.5-7B**.

* `src/eval_qwen3.py`
  Runs inference with **Qwen3-1.7B**.

* `src/generate_bode.py`
  Runs inference with **Bode** (`recogna-nlp/bode-7b-alpaca-pt-br`).

* `src/generate_tucano.py`
  Runs inference with **Tucano** (`TucanoBR/Tucano-2b4-Instruct`).

### LLM-based Evaluation

* `src/groq-judge.py`
  Used for evaluation with an **LLM-as-a-judge** approach.

### Statistics

* `statistics.ipynb`
  Notebook used to generate dataset statistics and analyses.

---

## Research Focus

This project explores:

* **Document-level simplification**
* The **Portuguese legal domain**, a relatively low-resource setting
* Trade-offs between **readability** and **content preservation**
* The use of **LLMs for large-scale data generation**
* Linguistic evaluation using **morphosyntactic metrics**

---

## Citation

If you use this dataset or repository, please cite:

```bibtex
@inproceedings{scalercio2026legalsim,
  title={LegalSim-PT: Building a Dataset for Legal Document Simplification in Portuguese Leveraging Linguistic Metrics},
  author={Scalercio, Arthur and others},
  booktitle={Proceedings of the International Conference on Computational Processing of the Portuguese Language (PROPOR)},
  year={2026}
}
```

---

## Notes

* The dataset contains automatically generated simplifications filtered using linguistic metrics.
* It is intended for research in NLP, legal text processing, and text simplification.


---

## Contact

Arthur Scalercio
[arthurscalercio@id.uff.br](mailto:arthurscalercio@id.uff.br)
