# Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis

Fine-tune BERT with BiLSTM for movie-review sentiment analysis and compute **overall polarity** from model predictions.

This repository contains:

| Path | Description |
|------|-------------|
| `BERT+BiLSTM-SA.py` | Original TensorFlow notebook-style script (binary IMDb, legacy hyperparams) |
| `Utilities.py` | Partial helpers for SST / Amazon binary loading |
| `paper/` | Source PDFs + extracted Markdown of Paper 1 & Paper 2 |
| **`reproduce_paper2/`** | **Scaffold to reproduce Paper 2 (start: IMDb-2 binary)** |

## Papers

1. **Paper 1** — *Sentiment Analysis of Movie Reviews Using BERT* (binary SA + overall polarity)
2. **Paper 2** — *Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis* (extends Paper 1 with multi-class, SMOTE/NLPAUG, four OP algorithms)

## Reproduce Paper 2 (recommended)

```bash
pip install -r reproduce_paper2/requirements.txt

# List all Paper 2 datasets (Phase A+B)
python -m reproduce_paper2.train --list-datasets

# Smoke: IMDb-2 binary
python -m reproduce_paper2.train --dataset imdb2 --epochs 1 --limit-train 512 --limit-test 256 --output-dir outputs/smoke_imdb2

# Full IMDb-2 binary (Paper 2 defaults)
python -m reproduce_paper2.train --dataset imdb2 --output-dir outputs/imdb2

# Fine-grained / other sets
python -m reproduce_paper2.train --dataset imdb3
python -m reproduce_paper2.train --dataset imdb4
python -m reproduce_paper2.train --dataset sst2
python -m reproduce_paper2.train --dataset sst5 --augment nlpaug
python -m reproduce_paper2.train --dataset mr
```

### Phase C (tables + stats + aug)

```bash
# Table I: local counts vs paper
python -m reproduce_paper2.table_i --out outputs/tables/table_i.csv

# Run all Ours jobs (smoke) then export Table II–VI
python -m reproduce_paper2.run_all_ours --smoke

# Full paper-length runs (GPU)
python -m reproduce_paper2.run_all_ours --device cuda --skip-existing

# Export tables only from existing results
python -m reproduce_paper2.export_tables --results-root outputs --out-dir outputs/tables
```

See [`reproduce_paper2/README.md`](reproduce_paper2/README.md) for Phases A–C.

### Polarity unit tests (no GPU)

```bash
python -m reproduce_paper2.test_polarity
```

## Legacy script

```bash
# Original entrypoint (re-splits data; hyperparams differ from Paper 2)
python BERT+BiLSTM-SA.py
```

For other datasets, see comments in `Utilities.py` (incomplete vs. papers).
