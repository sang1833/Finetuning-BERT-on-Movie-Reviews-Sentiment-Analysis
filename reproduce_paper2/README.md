# Paper 2 reproduction scaffold

Reproduce experiments from:

> *Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis*

This package currently implements **Phase A**:

- Paper 2 **binary** hyperparameters
- Official **IMDb-2** train/test split (25k / 25k)
- BERT-base-uncased + BiLSTM + classification head
- Overall polarity **Algorithm 2** (coef `1.2`)
- Configurable BiLSTM input: `cls` (Paper 2 text) or `sequence` (legacy repo)

Not yet implemented (next phases): MR / SST / Amazon, IMDb-3/4, SST-5 + NLPAUG/SMOTE, OP Alg 3–4 training loops.

## Setup

From the repository root:

```bash
pip install -r reproduce_paper2/requirements.txt
```

Implementation uses **PyTorch + HuggingFace** (the original repo script is TensorFlow; Paper 2 does not require a specific framework). GPU is strongly recommended for full 10-epoch runs.

IMDb is downloaded automatically to `~/.cache/reproduce_paper2/datasets/`.

## Quick smoke test (minutes)

Uses a small subset to verify download, tokenize, train, eval:

```bash
python -m reproduce_paper2.train --epochs 1 --limit-train 512 --limit-test 256 --output-dir outputs/smoke_imdb2
```

## Full IMDb-2 binary run (Paper 2 settings)

Default config:

| Setting        | Value              |
|----------------|--------------------|
| Model          | `bert-base-uncased`|
| Max length     | 128                |
| Batch size     | 32                 |
| Learning rate  | 3e-5               |
| Epsilon        | 1e-8               |
| Loss           | Binary cross-entropy (sigmoid head) |
| Epochs         | 10                 |
| BiLSTM input   | `[CLS]` only       |
| Freeze BERT    | yes                |
| Target accuracy| **97.67%** (paper) |

```bash
python -m reproduce_paper2.train --output-dir outputs/imdb2_binary
```

Optional: legacy full-sequence BiLSTM (closer to original `BERT+BiLSTM-SA.py`):

```bash
python -m reproduce_paper2.train --bilstm-input sequence --output-dir outputs/imdb2_seq
```

Unfreeze BERT (slower, may help if frozen run underfits):

```bash
python -m reproduce_paper2.train --no-freeze-bert
```

## Outputs

Under `--output-dir`:

| File              | Content                                      |
|-------------------|----------------------------------------------|
| `config.json`     | Full run configuration                       |
| `history.csv`     | Epoch metrics                                |
| `results.json`    | Test accuracy, OP, confusion matrix          |
| `best_model.pt`   | Best val-accuracy checkpoint                 |
| `final.weights.pt`| Final weights                                |

## Overall polarity check

Paper 2 Table VI (IMDb-2): Original OP = Neutral, Computed OP = Neutral  
(balanced pos/neg → neither side exceeds `1.2×` the other).

## Layout

```text
reproduce_paper2/
  config.py         # Paper 2 hyperparams
  data.py           # IMDb download + official split
  dataset_torch.py  # DataLoader
  model.py          # BERT + BiLSTM (PyTorch)
  polarity.py       # Algorithms 1–4 (2 used in train)
  train.py          # CLI entrypoint
  test_polarity.py  # Unit tests (no GPU)
```

## Notes on matching 97.67%

Exact reproduction is not guaranteed: the paper does not publish random seed, full layer-freeze policy, or library versions. This scaffold matches the **published hyperparameter table** and **official IMDb split**. Expect variance; treat ±1% as a reasonable first check if training is stable.
