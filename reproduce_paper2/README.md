# Paper 2 reproduction scaffold (Phase A + B)

Reproduce experiments from:

> *Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis*

## What is implemented

| Phase | Content | Status |
|-------|---------|--------|
| **A** | IMDb-2 binary, Paper 2 HPs, OP Alg 2 | ✅ |
| **B** | SST-2, MR, Amazon-2/5, IMDb-3/4, SST-5, multi-class train, OP Alg 1–4 | ✅ |
| **C1** | `run_all_ours` + `export_tables` → Table II–VI CSV/MD | ✅ |
| **C2** | `table_i` stats vs Paper Table I; stratified MR/Amazon splits | ✅ |
| **C3** | BERT-[CLS] SMOTE; NLPAUG = contextual emb + synonym + abstractive-lite | ✅ |

## Setup

```bash
pip install -r reproduce_paper2/requirements.txt
```

Stack: **PyTorch + HuggingFace**. GPU recommended for full runs.

IMDb cache: `~/.cache/reproduce_paper2/datasets/`  
HF datasets: HuggingFace cache.

## List datasets

```bash
python -m reproduce_paper2.train --list-datasets
```

| Key | Classes | Paper acc | OP mode | Notes |
|-----|---------|-----------|---------|-------|
| `imdb2` | 2 | 97.67% | binary | Official ACL IMDb |
| `mr` | 2 | 97.88% | binary | HF `rotten_tomatoes` |
| `sst2` | 2 | 97.62% | binary | GLUE SST-2 (val as test) |
| `amazon2` | 2 | 98.76% | binary | Video reviews if available |
| `imdb3` | 3 | 81.87% | three | From IMDb ratings |
| `imdb4` | 4 | 70.75% | four | Binary-tree split |
| `sst5` | 5 | 60.48% | five | + `--augment nlpaug` |
| `amazon5` | 5 | 69.68% | five | 1–5 stars |

Hyperparams auto-selected:

- **Binary:** max_len=128, lr=3e-5, batch=32, epochs=10, BCE  
- **Fine-grained:** max_len=256, lr=1e-4, batch=64, epochs=15, CE + weight_decay=1e-5  

## Commands

### Phase C1 — run all “Ours” + export tables

```bash
# Fast pipeline check (1 epoch, small data)
python -m reproduce_paper2.run_all_ours --smoke

# Full runs (GPU strongly recommended)
python -m reproduce_paper2.run_all_ours --device cuda --skip-existing

# Subset only
python -m reproduce_paper2.run_all_ours --only imdb2,sst5,sst5_nlpaug --smoke

# Rebuild Table II–VI from existing results.json (no training)
python -m reproduce_paper2.export_tables --results-root outputs --out-dir outputs/tables
```

Outputs: `outputs/tables/table_ii.csv` … `table_vi.csv`, `tables.md`  
Baseline rows are **copied from the paper**; only **Ours (repro)** is filled from local runs.

### Phase C2 — Table I dataset stats

```bash
python -m reproduce_paper2.table_i --out outputs/tables/table_i.csv
```

Compares local train/test class counts vs Paper Table I (`PAPER_TABLE_I` in `config.py`).

### Single-dataset train

```bash
# Unit tests (no GPU)
python -m reproduce_paper2.test_polarity

# Smoke: IMDb-2
python -m reproduce_paper2.train --dataset imdb2 --epochs 1 --limit-train 256 --limit-test 128 --output-dir outputs/smoke_imdb2

# Full binary / fine-grained
python -m reproduce_paper2.train --dataset imdb2 --output-dir outputs/imdb2
python -m reproduce_paper2.train --dataset imdb3
python -m reproduce_paper2.train --dataset sst5 --augment nlpaug --output-dir outputs/sst5_nlpaug
python -m reproduce_paper2.train --dataset sst5 --augment smote  --output-dir outputs/sst5_smote

# Amazon (large)
python -m reproduce_paper2.train --dataset amazon2 --limit-train 50000 --limit-test 10000
```

### Phase C3 — augmentation notes

| Flag | Implementation |
|------|----------------|
| `--augment smote` | Frozen BERT **[CLS]** features → SMOTE → map synthetic vectors to nearest original texts |
| `--augment nlpaug` | ContextualWordEmbsAug (BERT) + WordNet synonym + AbstSummAug(t5) when available, else extractive-lite |

Legacy full-sequence BiLSTM:

```bash
python -m reproduce_paper2.train --dataset imdb2 --bilstm-input sequence
```

## Outputs

Under `--output-dir`:

| File | Content |
|------|---------|
| `config.json` | Run config |
| `history.csv` | Epoch metrics |
| `results.json` | Test acc, OP, CM, paper target |
| `best_model.pt` | Best val checkpoint |

## Overall polarity

| Classes | Algorithm | coef notes |
|---------|-----------|------------|
| 2 | Alg 2 | 1.2 |
| 3 | Alg 1 | 85% neutral, then 1.5 |
| 4 | Alg 3 | super 1.2 / sub 1.5 |
| 5 | Alg 4 | 85% neutral + Alg 3 |

Table VI expectation is printed when known (e.g. IMDb-2 → Neutral).

## Layout

```text
reproduce_paper2/
  config.py         # Registry, Paper Table I–V constants
  data.py           # Loaders + stratified splits + stats
  augment.py        # C3: BERT-SMOTE + NLPAUG
  experiment.py     # Shared train/eval loop
  dataset_torch.py
  model.py
  polarity.py
  train.py          # Single-dataset CLI
  run_all_ours.py   # C1: multi-job runner
  export_tables.py  # C1: Table II–VI export
  table_i.py        # C2: Table I comparison
  test_polarity.py
```

## Notes

1. Exact % match is not guaranteed (seed / framework / HF split proxies).  
2. `mr` uses `rotten_tomatoes` (same research lineage as classic MR).  
3. `amazon_*` needs `datasets` access to `amazon_us_reviews` (or falls back to `amazon_polarity` for binary only).  
4. SST-5 paper gain is from **NLPAUG on raw text**, not SMOTE.  
5. CPU full runs are very slow — use CUDA when possible.
