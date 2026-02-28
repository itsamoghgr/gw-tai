"""
skill_extractor.py
------------------
Two-stage skill extraction and taxonomy normalisation pipeline.

Stage 1 – Candidate Extraction
    Identify skill-like phrases in job description text using:
    (a) Vocabulary matching against the raw-skill dictionary built from
        ``colorado_processed.csv`` (exact and fuzzy match).
    (b) Optional spaCy NLP pass for noun-chunk extraction when the package
        is installed (yields higher recall but slower runtime).

Stage 2 – Taxonomy Mapping
    Map each candidate phrase to its best ESCO / O*NET taxonomy entry using
    a TF-IDF cosine similarity index built from the taxonomy vocabulary.
    The ``colorado_processed.csv`` correlation coefficients seed the initial
    index; at runtime we also attempt direct lookup before falling back to
    similarity search.

Architecture decisions
-----------------------
- The taxonomy index is built **once** per pipeline run (``build_taxonomy_index``)
  and reused across all chunks.  Building it takes ~1 s for 165 k rows.
- TF-IDF is chosen over sentence-transformers because it requires no GPU,
  no large model download, and still produces good results for short skill
  phrases (typically 2–5 tokens).  Accuracy degrades for very domain-specific
  jargon; sentence-transformers would help there.
- Match confidence is the product of the TF-IDF cosine score and the
  pre-existing correlation coefficient from the processed file, giving a
  calibrated, interpretable confidence value between 0 and 1.
- Skills with final confidence < ``min_confidence`` are excluded from output.

Bias / limitation notes
------------------------
- The raw-skill vocabulary reflects skills present in *existing* postings,
  creating a feedback loop that may under-represent emerging skills.
- Soft / interpersonal skills ("communication", "teamwork") are frequently
  mentioned and will dominate counts unless filtered; the ``skill_type``
  field distinguishes ESCO behavioural skills from O*NET technical tools.
- Skill co-occurrence is not analysed here; a future extension could build
  a skill graph from the output.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TaxonomyIndex = Dict[str, object]   # internal opaque type


# ---------------------------------------------------------------------------
# Optional spaCy support
# ---------------------------------------------------------------------------

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
    _NLP = None  # loaded lazily
except ImportError:
    _SPACY_AVAILABLE = False
    logger.debug("spaCy not installed; noun-chunk extraction disabled.")


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = _spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found.  "
                "Run: python -m spacy download en_core_web_sm"
            )
            _NLP = None
    return _NLP


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_taxonomy_index(taxonomy_df: pd.DataFrame) -> TaxonomyIndex:
    """Build an in-memory lookup + TF-IDF index from the taxonomy DataFrame.

    Parameters
    ----------
    taxonomy_df:
        Output of ``ingestion.load_taxonomy``.  Expected columns:
        ``raw_skill``, ``taxonomy_skill``, ``taxonomy_description``,
        ``taxonomy_source``, ``correlation``.

    Returns
    -------
    dict with keys:
        - ``lookup``: dict[raw_skill_str → best taxonomy row as dict]
        - ``vectorizer``: fitted TfidfVectorizer
        - ``tfidf_matrix``: sparse matrix (n_taxonomy_skills × vocab)
        - ``taxonomy_rows``: list[dict] aligned with tfidf_matrix rows
    """
    logger.info("Building taxonomy index from %d rows …", len(taxonomy_df))

    # --- Build exact-match lookup (raw skill → best taxonomy row) ---
    # Vectorized: sort descending by correlation, deduplicate on normalised key,
    # then build the dict in one pass — O(n log n) vs O(n) Python loops.
    normed = taxonomy_df["raw_skill"].str.lower().str.strip()
    best_per_raw = (
        taxonomy_df.assign(_norm_key=normed)
        .sort_values("correlation", ascending=False)
        .drop_duplicates(subset=["_norm_key"])
        .drop(columns=["_norm_key"])
        .reset_index(drop=True)
    )
    lookup: Dict[str, dict] = {
        row["raw_skill"].lower().strip(): row.to_dict()
        for _, row in best_per_raw.iterrows()
    }

    logger.info("Exact-match lookup: %d unique raw skills", len(lookup))

    # --- Build TF-IDF index over taxonomy_skill + taxonomy_description ---
    # Deduplicate taxonomy skills (keep highest-correlation row per skill)
    best_tax = (
        taxonomy_df
        .sort_values("correlation", ascending=False)
        .drop_duplicates(subset=["taxonomy_skill"])
        .reset_index(drop=True)
    )

    # Combine skill name + first 100 chars of description for richer signal
    corpus = (
        best_tax["taxonomy_skill"].fillna("") + " " +
        best_tax["taxonomy_description"].fillna("").str[:100]
    ).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    taxonomy_rows = best_tax.to_dict(orient="records")

    logger.info(
        "TF-IDF index built: %d taxonomy skills, vocab size %d",
        len(taxonomy_rows),
        len(vectorizer.vocabulary_),
    )

    return {
        "lookup": lookup,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "taxonomy_rows": taxonomy_rows,
    }


# ---------------------------------------------------------------------------
# Skill extraction from text
# ---------------------------------------------------------------------------

# Minimum phrase length (chars) and maximum token count for a skill candidate
_MIN_PHRASE_LEN = 3
_MAX_PHRASE_TOKENS = 6

# Single-token benefit terms that produce noise when matched as standalone skills.
# These are filtered *after* vocabulary scan so multi-word phrases like
# "dental assistant" are not affected.
_BENEFIT_NOISE_TERMS = {
    "dental", "vision", "medical", "health", "insurance",
    "401k", "retirement", "pension", "pto", "vacation",
    "tuition", "relocation", "bonus",
}

# Stopwords that should not form skill candidates on their own
_SKILL_STOPWORDS = {
    "experience", "years", "ability", "knowledge", "skills", "skill",
    "understanding", "work", "working", "strong", "excellent", "good",
    "required", "preferred", "plus", "must", "will", "our", "the", "and",
    "or", "to", "in", "of", "a", "an", "with", "for", "on", "at", "by",
    "as", "is", "are", "be", "been", "being",
}


def extract_skill_candidates(text: str, lookup: Dict[str, dict]) -> List[str]:
    """Return a deduplicated list of skill candidate phrases from ``text``.

    Two methods are combined:
    1. **Vocabulary scan**: slide a window over the text tokens and test each
       1–6 token phrase against the raw-skill lookup dictionary.
    2. **spaCy noun chunks** (if available): add root-normalised noun chunks
       as additional candidates.

    Parameters
    ----------
    text:
        Cleaned job description text.
    lookup:
        The ``lookup`` dict from ``build_taxonomy_index``.

    Returns
    -------
    List[str]
        Deduplicated candidate phrases (lowercased).
    """
    if not text:
        return []

    # Tokenise by whitespace / punctuation (simple but fast)
    raw_tokens = re.split(r"[\s,;:()\[\]{}'\"]+", text.lower())
    tokens = [t for t in raw_tokens if t and t not in _SKILL_STOPWORDS and len(t) >= 2]

    candidates = set()

    # Window scan against lookup vocabulary
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + _MAX_PHRASE_TOKENS + 1, len(tokens) + 1)):
            phrase = " ".join(tokens[i:j])
            if phrase in lookup and len(phrase) >= _MIN_PHRASE_LEN:
                candidates.add(phrase)

    # Optional spaCy noun-chunk extraction
    if _SPACY_AVAILABLE:
        nlp = _get_nlp()
        if nlp is not None:
            # Limit text length for spaCy (it's slow on very long strings)
            doc = nlp(text[:5_000])
            for chunk in doc.noun_chunks:
                phrase = chunk.root.lemma_.lower().strip()
                if phrase and len(phrase) >= _MIN_PHRASE_LEN and phrase not in _SKILL_STOPWORDS:
                    candidates.add(phrase)

    # Remove single-token benefit terms to suppress noise like "dental", "vision"
    # appearing as skills when a job mentions benefit packages.
    candidates = {c for c in candidates if c not in _BENEFIT_NOISE_TERMS}

    return list(candidates)


# ---------------------------------------------------------------------------
# Taxonomy mapping
# ---------------------------------------------------------------------------


def map_to_taxonomy(
    candidates: List[str],
    index: TaxonomyIndex,
    min_confidence: float = 0.40,
    top_k: int = 1,
) -> List[dict]:
    """Map a list of candidate skill phrases to taxonomy entries.

    For each candidate:
    1. Attempt an **exact lookup** in the raw-skill dictionary.
    2. If no exact match, perform **TF-IDF cosine similarity** search and
       take the top-k taxonomy skills above ``min_confidence``.

    The final confidence score is:
        ``tfidf_cosine * taxonomy_correlation``
    for similarity matches, or the raw ``taxonomy_correlation`` for exact
    matches (cosine = 1.0).

    Parameters
    ----------
    candidates:
        Output of ``extract_skill_candidates``.
    index:
        Output of ``build_taxonomy_index``.
    min_confidence:
        Minimum final confidence to include a mapping in the result.
    top_k:
        Maximum number of taxonomy matches to return per candidate.
        Set to 1 for a clean one-to-one mapping.

    Returns
    -------
    List[dict]
        Each dict has keys:
        ``raw_skill``, ``taxonomy_skill``, ``taxonomy_description``,
        ``taxonomy_source``, ``confidence``, ``match_method``
    """
    if not candidates:
        return []

    lookup = index["lookup"]
    vectorizer = index["vectorizer"]
    tfidf_matrix = index["tfidf_matrix"]
    taxonomy_rows = index["taxonomy_rows"]

    results = []
    seen_taxonomy_skills = set()  # dedup across candidates for one job

    for candidate in candidates:
        row = lookup.get(candidate)

        if row is not None:
            # --- Exact match ---
            confidence = float(row["correlation"])
            if confidence >= min_confidence:
                tax_skill = row["taxonomy_skill"]
                if tax_skill not in seen_taxonomy_skills:
                    seen_taxonomy_skills.add(tax_skill)
                    results.append({
                        "raw_skill": candidate,
                        "taxonomy_skill": tax_skill,
                        "taxonomy_description": row.get("taxonomy_description", ""),
                        "taxonomy_source": row.get("taxonomy_source", ""),
                        "confidence": round(confidence, 4),
                        "match_method": "exact",
                    })
            continue

        # --- TF-IDF similarity match ---
        try:
            candidate_vec = vectorizer.transform([candidate])
        except Exception:
            continue

        sims = cosine_similarity(candidate_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]

        for idx in top_indices:
            sim_score = float(sims[idx])
            if sim_score < 0.01:
                break
            tax_row = taxonomy_rows[idx]
            confidence = sim_score * float(tax_row.get("correlation", 0.60))
            if confidence < min_confidence:
                continue
            tax_skill = tax_row["taxonomy_skill"]
            if tax_skill not in seen_taxonomy_skills:
                seen_taxonomy_skills.add(tax_skill)
                results.append({
                    "raw_skill": candidate,
                    "taxonomy_skill": tax_skill,
                    "taxonomy_description": tax_row.get("taxonomy_description", ""),
                    "taxonomy_source": tax_row.get("taxonomy_source", ""),
                    "confidence": round(confidence, 4),
                    "match_method": "tfidf",
                })

    return results


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def extract_and_map_skills(
    df: pd.DataFrame,
    index: TaxonomyIndex,
    text_col: str = "description_clean",
    id_col: str = "system_job_id",
    min_confidence: float = 0.40,
    max_skills: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and taxonomy-map skills for every row in a DataFrame chunk.

    Parameters
    ----------
    df:
        A processed chunk containing ``text_col`` and ``id_col``.
    index:
        Pre-built taxonomy index.
    text_col:
        Column with cleaned description text.
    id_col:
        Job identifier column (used to link skills back to jobs).
    min_confidence:
        Minimum confidence for including a skill mapping.
    max_skills:
        Maximum skills per job (ranked by confidence, top-k kept).

    Returns
    -------
    (jobs_df, skills_df):
        - ``jobs_df``: input df with ``skill_count`` column added.
        - ``skills_df``: long-format table with one row per (job, skill) pair.
          Columns: ``system_job_id``, ``raw_skill``, ``taxonomy_skill``,
          ``taxonomy_description``, ``taxonomy_source``, ``confidence``,
          ``match_method``.
    """
    skills_rows = []

    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        job_id = str(row.get(id_col, ""))

        candidates = extract_skill_candidates(text, index["lookup"])
        mappings = map_to_taxonomy(candidates, index, min_confidence=min_confidence)

        # Keep top-k by confidence
        mappings = sorted(mappings, key=lambda m: m["confidence"], reverse=True)[:max_skills]

        for m in mappings:
            skills_rows.append({id_col: job_id, **m})

    skills_df = pd.DataFrame(skills_rows) if skills_rows else _empty_skills_df(id_col)

    # Add skill_count to the jobs dataframe
    if not skills_df.empty:
        skill_counts = skills_df.groupby(id_col).size().rename("skill_count")
        df = df.copy().merge(
            skill_counts.reset_index(), on=id_col, how="left"
        )
        df["skill_count"] = df["skill_count"].fillna(0).astype(int)
    else:
        df = df.copy()
        df["skill_count"] = 0

    return df, skills_df


def _empty_skills_df(id_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[id_col, "raw_skill", "taxonomy_skill", "taxonomy_description",
                 "taxonomy_source", "confidence", "match_method"]
    )
