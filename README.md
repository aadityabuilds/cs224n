# Recursive Self-Improvement for Continual Adaptation in Code Generation

CS224N Final Project — Stanford University, Spring 2025

**Authors:** Aaditya Nalawade, Chandra Suda, Ethan Goodhart

A self-improving agent that uses Self-Distillation with Policy Optimization (SDPO) and Retrieval-Augmented Generation (RAG) to continually improve on code generation tasks from LiveCodeBench.

## Project Structure

```
├── agent/                  # Core agent implementation
│   ├── config.py           # Configuration dataclasses
│   ├── model.py            # 1.5B model management (student + teacher + reference)
│   ├── model_7b.py         # 7B model management (memory-optimized)
│   ├── rag.py              # FAISS-based RAG vector database
│   ├── router.py           # LLM-based adaptive routing (SDPO/RAG/pass)
│   ├── verification.py     # Code verification wrapper
│   ├── sdpo_update.py      # SDPO update step (1.5B)
│   ├── sdpo_update_7b.py   # SDPO update step (7B, 4 attempts)
│   ├── sdpo_update_7b_v2.py # SDPO update step (7B, 16 attempts)
│   └── sdpo_update_7b_v3.py # SDPO update step (7B, 32 attempts)
├── training/               # Training scripts
│   ├── main.py             # 1.5B local training loop
│   ├── main_7b.py          # 7B v1 training loop
│   ├── main_7b_v2.py       # 7B v2 training loop (16 attempts)
│   ├── main_7b_v3.py       # 7B v3 training loop (32 attempts)
│   ├── modal_app.py        # 1.5B Modal cloud deployment
│   ├── modal_app_7b.py     # 7B v1 Modal deployment
│   ├── modal_app_7b_v2.py  # 7B v2 Modal deployment
│   └── modal_app_7b_v3.py  # 7B v3 Modal deployment
├── eval/                   # Evaluation and analysis
│   ├── eval.py             # 1.5B evaluation with ablations
│   ├── eval_7b.py          # 7B evaluation
│   ├── eval_sweep_7b.py    # 7B v1 checkpoint sweep
│   ├── eval_sweep_7b_v2.py # 7B v2 checkpoint sweep
│   ├── eval_sweep_7b_v3.py # 7B v3 checkpoint sweep
│   ├── analyze_modal_run_metrics.py  # Live training metrics
│   └── export_rag_chunks_md.py       # RAG database export
├── SDPO/                   # SDPO framework (git submodule)
├── report.tex              # Project report
└── references.bib          # Bibliography
```

## Setup

```bash
git clone --recurse-submodules <repo-url>
pip install -e .
```

## Models

- **Qwen2.5-1.5B-Instruct** — baseline model with full reference model comparison
- **Qwen2.5-7B-Instruct** — primary model with memory-optimized training

## Acknowledgments

This project uses the [SDPO framework](https://github.com/lasgroup/SDPO) by Hübotter et al., licensed under the Apache License 2.0.
