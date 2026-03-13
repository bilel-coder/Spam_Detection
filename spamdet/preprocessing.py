"""
Text preprocessing pipeline.

Steps
─────
1. Lower-case
2. Remove URLs, emails, phone numbers
3. Remove punctuation & special characters
4. Tokenise
5. Remove stop-words
6. Lemmatise

The module also encodes the binary label (ham → 0, spam → 1).
"""

import re
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from spamdet.config import TEXT_COLUMN, LABEL_COLUMN
from spamdet.data import load_raw, save_processed

logger = logging.getLogger(__name__)

# ── NLTK downloads (idempotent) ───────────────────────────────────────────────
for _resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()

# ── Regex patterns ────────────────────────────────────────────────────────────
_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL   = re.compile(r"\S+@\S+")
_RE_PHONE   = re.compile(r"\b\d[\d\s\-().]{6,}\d\b")
_RE_SPECIAL = re.compile(r"[^a-z\s]")
_RE_SPACES  = re.compile(r"\s+")


# ── Core cleaning function ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Return a normalised, lemmatised version of *text*."""
    text = str(text).lower()
    text = _RE_URL.sub(" url ", text)
    text = _RE_EMAIL.sub(" email ", text)
    text = _RE_PHONE.sub(" phone ", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()

    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in _STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)


# ── DataFrame-level helpers ───────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'ham' → 0, 'spam' → 1 in-place (returns modified copy)."""
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map({"ham": 0, "spam": 1})
    if df[LABEL_COLUMN].isna().any():
        raise ValueError("Unknown label values found. Expected 'ham' or 'spam'.")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hand-crafted numeric features (length, caps ratio, etc.)."""
    df = df.copy()
    df["msg_len"]      = df[TEXT_COLUMN].apply(len)
    df["num_words"]    = df[TEXT_COLUMN].apply(lambda x: len(str(x).split()))
    df["caps_ratio"]   = df[TEXT_COLUMN].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    df["num_digits"]   = df[TEXT_COLUMN].apply(
        lambda x: sum(c.isdigit() for c in str(x))
    )
    df["has_url"]      = df[TEXT_COLUMN].apply(
        lambda x: int(bool(_RE_URL.search(str(x))))
    )
    df["has_phone"]    = df[TEXT_COLUMN].apply(
        lambda x: int(bool(_RE_PHONE.search(str(x))))
    )
    df["clean_text"]   = df[TEXT_COLUMN].apply(clean_text)
    return df


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline: load raw → clean → save."""
    df = load_raw()
    df = encode_labels(df)
    df = add_features(df)

    # Drop duplicates after cleaning
    before = len(df)
    df.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)
    logger.info("Dropped %d duplicates. Remaining: %d rows", before - len(df), len(df))

    if save:
        save_processed(df)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_preprocessing()
    print("Preprocessing complete.")
