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

# Smoke test
python -m reproduce_paper2.train --epochs 1 --limit-train 512 --limit-test 256 --output-dir outputs/smoke_imdb2

# Full IMDb-2 binary (Paper 2 defaults: max_len=128, lr=3e-5, batch=32, epochs=10)
python -m reproduce_paper2.train --output-dir outputs/imdb2_binary
```

See [`reproduce_paper2/README.md`](reproduce_paper2/README.md) for settings, outputs, and limitations.

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
