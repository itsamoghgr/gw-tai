"""
cleaning.py
-----------
Transform raw, HTML-laden job description text into clean plain text
suitable for downstream NLP processing.

Design decisions & bias notes
-------------------------------
- HTML tags are stripped via a regex fallback (no external dependency);
  for production use, ``BeautifulSoup`` (optional) gives better results on
  malformed HTML and is used when available.
- We preserve sentence boundaries (periods, question marks, exclamation
  marks) because downstream extractors rely on sentence-level patterns.
- Bullet-point markers (•, *, +, -, –, —) are replaced with newlines so
  lists survive as discrete items.
- Unicode normalization (NFKC) collapses decorative whitespace, non-breaking
  spaces, and lookalike characters that frequently appear in copy-pasted
  job descriptions.
- Telephone numbers and URLs are replaced with sentinel tokens
  (<PHONE> / <URL>) rather than deleted, preserving sentence structure.
- No stop-word removal or lemmatisation is performed here; that is left to
  skill extraction so that the cleaned text is still human-readable.

Known limitations
-----------------
- Very long descriptions (>20 k characters) are truncated at 20 k to prevent
  runaway regex backtracking.  The threshold is configurable.
- Descriptions written in non-English languages are NOT detected or flagged;
  downstream extractors will silently produce empty results for them.
"""

import html
import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DESC_LENGTH = 20_000  # characters; prevents catastrophic regex backtracking

# HTML tag pattern (fast fallback when bs4 is not installed)
_RE_HTML_TAG = re.compile(r"<[^>]{0,200}>", re.DOTALL)

# HTML entity pattern (e.g. &amp; &nbsp; &#39;)
_RE_HTML_ENTITY = re.compile(r"&[a-zA-Z0-9#]{1,10};")

# Bullet / list markers → newline
_RE_BULLET = re.compile(r"(?m)^\s*[•·▪▸►◆*+\-–—]\s+")

# Consecutive whitespace (not newlines) → single space
_RE_HORIZ_SPACE = re.compile(r"[ \t\r\f\v]+")

# More than 2 consecutive newlines → 2 newlines
_RE_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# URL pattern
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

# Phone number pattern (US-centric)
_RE_PHONE = re.compile(
    r"""
    (?:
        \(?\d{3}\)?        # area code
        [\s.\-/]?
        \d{3}
        [\s.\-/]?
        \d{4}
    )
    """,
    re.VERBOSE,
)

# Email address
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")

# Try to import BeautifulSoup; fall back gracefully
try:
    from bs4 import BeautifulSoup as _BS

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.debug("BeautifulSoup not found; using regex HTML stripper (install beautifulsoup4 for better results).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_description(text: Optional[str], max_length: int = MAX_DESC_LENGTH) -> str:
    """Clean a single job description string.

    Steps (in order):
    1. Guard against None / NaN.
    2. Truncate to ``max_length`` characters.
    3. Decode HTML entities (``&amp;`` → ``&``).
    4. Strip HTML tags.
    5. Normalise Unicode (NFKC).
    6. Replace URLs, phone numbers, and emails with sentinel tokens.
    7. Normalise bullet markers.
    8. Collapse excess whitespace.

    Parameters
    ----------
    text:
        Raw description string (may contain HTML).
    max_length:
        Maximum character length before truncation.

    Returns
    -------
    str
        Cleaned plain-text description.  Empty string if input is null.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- 1. Truncate early to cap regex complexity ---
    if len(text) > max_length:
        text = text[:max_length]

    # --- 2. Decode HTML entities first (so tags become parseable) ---
    text = html.unescape(text)

    # --- 3. Strip HTML tags ---
    if _BS4_AVAILABLE:
        soup = _BS(text, "html.parser")
        text = soup.get_text(separator="\n")
    else:
        text = _RE_HTML_TAG.sub(" ", text)
        text = _RE_HTML_ENTITY.sub(" ", text)

    # --- 4. Unicode normalisation ---
    text = unicodedata.normalize("NFKC", text)

    # --- 5. Replace URLs / phones / emails with sentinel tokens ---
    text = _RE_URL.sub(" <URL> ", text)
    text = _RE_PHONE.sub(" <PHONE> ", text)
    text = _RE_EMAIL.sub(" <EMAIL> ", text)

    # --- 6. Normalise bullet points ---
    text = _RE_BULLET.sub("\n", text)

    # --- 7. Collapse horizontal whitespace ---
    text = _RE_HORIZ_SPACE.sub(" ", text)

    # --- 8. Collapse excess newlines ---
    text = _RE_EXCESS_NEWLINES.sub("\n\n", text)

    return text.strip()


def clean_jobs_chunk(df: pd.DataFrame, desc_col: str = "description") -> pd.DataFrame:
    """Apply ``clean_description`` to every row in a DataFrame chunk.

    A new column ``description_clean`` is added; the original ``description``
    column is preserved for audit purposes.

    Parameters
    ----------
    df:
        A chunk of job postings (from ``ingestion.iter_jobs``).
    desc_col:
        Name of the column containing raw description text.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``description_clean`` column added.
    """
    if desc_col not in df.columns:
        logger.warning("Column '%s' not found in chunk; skipping cleaning.", desc_col)
        df["description_clean"] = ""
        return df

    df = df.copy()
    df["description_clean"] = df[desc_col].apply(clean_description)

    # Quick quality metrics for logging
    empty_count = (df["description_clean"] == "").sum()
    if empty_count:
        logger.debug("Chunk: %d / %d rows have empty descriptions after cleaning.", empty_count, len(df))

    return df


def get_description_stats(df: pd.DataFrame, col: str = "description_clean") -> dict:
    """Return a dict of descriptive statistics for the cleaned description column.

    Useful for the pipeline report.
    """
    lengths = df[col].str.len()
    return {
        "count": int(len(df)),
        "empty": int((lengths == 0).sum()),
        "mean_length": round(float(lengths.mean()), 1),
        "median_length": round(float(lengths.median()), 1),
        "max_length": int(lengths.max()),
        "min_nonzero_length": int(lengths[lengths > 0].min()) if (lengths > 0).any() else 0,
    }
