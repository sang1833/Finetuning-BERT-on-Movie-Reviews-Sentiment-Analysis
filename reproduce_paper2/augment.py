"""Paper 2 accuracy-improvement: BERT-feature SMOTE + NLPAUG (C3).

Paper findings on SST-5:
  - bare BERT+BiLSTM ≈ 58.44%
  - +SMOTE ≈ 58.36% (no gain / slight drop)
  - +NLPAUG ≈ 60.48% (gain)

SMOTE (paper): convert minority reviews → numerical features → SMOTE →
combine with original data. Here numerical features = frozen BERT [CLS].

NLPAUG (paper): abstractive-summarization-style ops + synonym replacement
by word-embedding proximity. We chain:
  1) ContextualWordEmbsAug (BERT substitute) when available
  2) WordNet synonym replacement
  3) Extractive/abstractive-lite compression (sentence keep / AbstSummAug)
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .data import TextClassificationSplit


def _minority_labels(labels: np.ndarray) -> List[int]:
    counts = Counter(labels.tolist())
    if not counts:
        return []
    max_c = max(counts.values())
    return [lab for lab, c in counts.items() if c < max_c]


# ---------------------------------------------------------------------------
# BERT [CLS] encoding (for SMOTE feature space)
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_bert_cls(
    texts: Sequence[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode texts to frozen BERT [CLS] vectors (N, 768)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev)
    model.eval()

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        enc = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        vectors.append(cls)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 768), dtype=np.float32)


def oversample_smote_bert(
    texts: List[str],
    labels: np.ndarray,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    seed: int = 42,
    device: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """SMOTE in BERT [CLS] space; map synthetic points to nearest original texts.

    This follows paper intent (SMOTE on transformed BERT features) while
    keeping a text pipeline for end-to-end BERT+BiLSTM training. Synthetic
    samples become additional copies of nearest minority texts in CLS space
    (oversampling guided by BERT geometry). Often does **not** improve
    accuracy — consistent with paper Table V.
    """
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print(
            "[augment] imbalanced-learn missing — skipping BERT-SMOTE. "
            "Install: pip install imbalanced-learn"
        )
        return texts, labels

    if len(texts) == 0 or len(set(labels.tolist())) < 2:
        return texts, labels

    print(f"[augment] Encoding {len(texts)} texts with frozen {model_name} [CLS]...")
    X = encode_bert_cls(
        texts,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    y = labels.astype(np.int64)

    counts = Counter(y.tolist())
    min_c = min(counts.values())
    k = max(1, min(5, min_c - 1))
    if min_c < 2:
        print("[augment] BERT-SMOTE skipped: minority class too small")
        return texts, labels

    try:
        sm = SMOTE(random_state=seed, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
    except ValueError as exc:
        print(f"[augment] BERT-SMOTE failed ({exc}); skipping")
        return texts, labels

    # Map every resampled vector (incl. originals + synthetic) to nearest train text
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X_res)
    idx = idx.reshape(-1)
    new_texts = [texts[i] for i in idx]
    print(
        f"[augment] BERT-SMOTE: {len(texts)} → {len(new_texts)} samples "
        f"(class counts {dict(Counter(y_res.tolist()))})"
    )
    return new_texts, np.asarray(y_res, dtype=np.int64)


# ---------------------------------------------------------------------------
# NLPAUG
# ---------------------------------------------------------------------------


def _ensure_nltk_wordnet() -> None:
    import nltk

    for pkg in ("wordnet", "omw-1.4", "averaged_perceptron_tagger", "punkt"):
        try:
            if pkg.startswith("punkt"):
                nltk.data.find(f"tokenizers/{pkg}")
            elif "tagger" in pkg:
                nltk.data.find(f"taggers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _build_nlpaug_pipeline(model_name: str = "bert-base-uncased"):
    """Build ordered list of augmenters (embedding synonym + abstractive-lite)."""
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

    augs = []
    # 1) Contextual embedding substitution (word proximity in BERT space)
    try:
        augs.append(
            naw.ContextualWordEmbsAug(
                model_path=model_name,
                action="substitute",
                top_k=20,
                aug_p=0.3,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        print("[augment] NLPAUG: ContextualWordEmbsAug enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"[augment] ContextualWordEmbsAug unavailable ({exc})")

    # 2) WordNet synonym
    try:
        _ensure_nltk_wordnet()
        augs.append(naw.SynonymAug(aug_src="wordnet", aug_p=0.3))
        print("[augment] NLPAUG: SynonymAug(wordnet) enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"[augment] SynonymAug unavailable ({exc})")

    # 3) Abstractive / extractive compression
    try:
        augs.append(
            nas.AbstSummAug(
                model_path="t5-small",
                max_length=64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        print("[augment] NLPAUG: AbstSummAug(t5-small) enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"[augment] AbstSummAug unavailable ({exc}); using extractive lite")

    return augs


def _extractive_lite(text: str, rng: np.random.Generator) -> str:
    """Cheap abstractive-style compression: keep a subset of sentences."""
    parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()]
    if len(parts) <= 1:
        words = text.split()
        if len(words) <= 8:
            return text
        keep = max(4, int(len(words) * 0.7))
        start = int(rng.integers(0, max(1, len(words) - keep)))
        return " ".join(words[start : start + keep])
    k = max(1, int(round(len(parts) * 0.6)))
    chosen = sorted(rng.choice(len(parts), size=min(k, len(parts)), replace=False))
    return ". ".join(parts[i] for i in chosen) + "."


def augment_texts_nlpaug(
    texts: List[str],
    labels: np.ndarray,
    factor: float = 0.5,
    seed: int = 42,
    model_name: str = "bert-base-uncased",
) -> Tuple[List[str], np.ndarray]:
    """NLPAUG on minority classes: embedding synonym + abstractive-style ops."""
    try:
        import nlpaug  # noqa: F401
    except ImportError:
        print(
            "[augment] nlpaug not installed — skipping. "
            "Install: pip install nlpaug nltk"
        )
        return texts, labels

    rng = np.random.default_rng(seed)
    augs = _build_nlpaug_pipeline(model_name=model_name)
    minorities = set(_minority_labels(labels))
    if not minorities:
        return texts, labels

    new_texts: List[str] = list(texts)
    new_labels: List[int] = labels.tolist()

    for lab in minorities:
        idxs = [i for i, y in enumerate(labels) if int(y) == lab]
        n_aug = max(1, int(len(idxs) * factor))
        chosen = rng.choice(idxs, size=min(n_aug, len(idxs)), replace=False)
        for i in chosen:
            src = texts[i]
            out = src
            # Randomly pick one available augmenter, fallback extractive lite
            if augs:
                aug = augs[int(rng.integers(0, len(augs)))]
                try:
                    cand = aug.augment(src)
                    if isinstance(cand, list):
                        cand = cand[0] if cand else src
                    if cand and isinstance(cand, str) and cand.strip():
                        out = cand
                except Exception:  # noqa: BLE001
                    out = _extractive_lite(src, rng)
            else:
                out = _extractive_lite(src, rng)

            # Optional second pass: light extractive on long texts
            if len(out.split()) > 40 and float(rng.random()) < 0.3:
                out = _extractive_lite(out, rng)

            new_texts.append(out)
            new_labels.append(int(lab))

    print(
        f"[augment] NLPAUG: {len(texts)} → {len(new_texts)} train samples "
        f"(minorities={sorted(minorities)}, n_augs={len(augs)})"
    )
    return new_texts, np.asarray(new_labels, dtype=np.int64)


def apply_augment(
    split: TextClassificationSplit,
    mode: str,
    factor: float = 0.5,
    seed: int = 42,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    device: Optional[str] = None,
) -> TextClassificationSplit:
    if mode in (None, "none", ""):
        return split
    if mode == "nlpaug":
        texts, labels = augment_texts_nlpaug(
            split.texts,
            split.labels,
            factor=factor,
            seed=seed,
            model_name=model_name,
        )
    elif mode in ("smote", "smote_bert"):
        texts, labels = oversample_smote_bert(
            split.texts,
            split.labels,
            model_name=model_name,
            max_length=max_length,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"Unknown augment mode: {mode}")
    return TextClassificationSplit(
        texts=texts, labels=labels, name=split.name + f"_{mode}"
    )
