"""
ingestion.py
------------
Load and validate the raw NLx job posting CSV (colorado.csv) and the
pre-built skill taxonomy mapping (colorado_processed.csv).

Key design decisions
---------------------
- We read colorado.csv in configurable chunks so the pipeline can handle
  arbitrarily large files without running out of RAM.
- Low-cardinality string columns are cast to ``pandas.Categorical`` to
  reduce memory consumption by ~60–70 % on a 994 k-row dataset.
- Strict dtype handling is avoided for the raw CSV because many fields are
  sparsely populated; instead, we let pandas infer and clean up afterwards.
- We surface a validation report so callers can make informed decisions
  about data quality before proceeding.

Bias / limitation notes
------------------------
- Source data is limited to Colorado postings; any derived skill rankings
  will reflect Colorado's labour market, not the national picture.
- Expired postings are included by default; callers can filter them out
  via the ``active_only`` flag.
"""

import logging
import os
from typing import Iterator, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column subsets
# ---------------------------------------------------------------------------

# Columns we actually use downstream – ignoring the rest saves memory.
JOB_COLS_KEEP = [
    "system_job_id",
    "job_id",
    "title",
    "description",
    "date_acquired",
    "created_date",
    "zipcode",
    "city",
    "state",
    "country",
    "parameters_salary_unit",
    "parameters_salary_min",
    "parameters_salary_max",
    "parameters_hours_per_week_max",
    "classifications_onet_code",
    "classifications_naics_code",
    "requirements_min_education",
    "requirements_experience",
    "requirements_license",
    "application_company",
    "expired",
    "fedcontractor",
    "ghostjob",
    "jobclass",
    "cip_codes",
    "moc_codes",
]

TAXONOMY_COLS = [
    "Research ID",
    "Raw Skill",
    "Taxonomy Skill",
    "Taxonomy Description",
    "Taxonomy Source",
    "Correlation Coefficient",
]

# Categorical columns (low cardinality) – converted to save memory
CATEGORICAL_JOB_COLS = [
    "state",
    "country",
    "parameters_salary_unit",
    "expired",
    "fedcontractor",
    "ghostjob",
    "jobclass",
]

CATEGORICAL_TAX_COLS = ["Taxonomy Source"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_taxonomy(path: str, threshold: float = 0.60) -> pd.DataFrame:
    """Load the pre-built skill taxonomy mapping file.

    Parameters
    ----------
    path:
        Absolute or relative path to ``colorado_processed.csv``.
    threshold:
        Drop rows whose ``Correlation Coefficient`` is below this value.
        Default 0.60 matches the minimum seen in the dataset and is a
        conservative choice; raise it to improve precision at the cost of
        recall.

    Returns
    -------
    pd.DataFrame
        Cleaned taxonomy DataFrame with columns renamed to snake_case.

    Notes
    -----
    - Duplicate (Research ID, Raw Skill) pairs are deduplicated by keeping
      the row with the highest correlation coefficient, so downstream
      skill matching uses the best available mapping.
    - Taxonomy Source is validated; any unexpected value is flagged.
    """
    logger.info("Loading taxonomy mapping from: %s", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    df = pd.read_csv(
        path,
        usecols=TAXONOMY_COLS,
        dtype={
            "Research ID": str,
            "Raw Skill": str,
            "Taxonomy Skill": str,
            "Taxonomy Description": str,
            "Taxonomy Source": str,
            "Correlation Coefficient": float,
        },
        low_memory=False,
    )

    # Rename to snake_case for consistent access
    df = df.rename(
        columns={
            "Research ID": "research_id",
            "Raw Skill": "raw_skill",
            "Taxonomy Skill": "taxonomy_skill",
            "Taxonomy Description": "taxonomy_description",
            "Taxonomy Source": "taxonomy_source",
            "Correlation Coefficient": "correlation",
        }
    )

    # Drop rows below the threshold
    before = len(df)
    df = df[df["correlation"] >= threshold].copy()
    logger.info(
        "Taxonomy rows after correlation filter (>= %.2f): %d / %d",
        threshold,
        len(df),
        before,
    )

    # Strip whitespace from text fields
    for col in ["raw_skill", "taxonomy_skill", "taxonomy_description"]:
        df[col] = df[col].str.strip().str.lower()

    # Deduplicate: keep highest-correlation match per (research_id, raw_skill)
    df = (
        df.sort_values("correlation", ascending=False)
        .drop_duplicates(subset=["research_id", "raw_skill"])
        .reset_index(drop=True)
    )

    # Validate taxonomy source values
    known_sources = {"esco", "onet_tech", "onet_skill"}
    unknown = set(df["taxonomy_source"].unique()) - known_sources
    if unknown:
        logger.warning("Unknown taxonomy sources encountered: %s", unknown)

    df["taxonomy_source"] = df["taxonomy_source"].astype("category")

    logger.info("Taxonomy loaded: %d mappings, %d unique jobs", len(df), df["research_id"].nunique())
    return df


def iter_jobs(
    path: str,
    chunk_size: int = 10_000,
    active_only: bool = False,
    sample_size: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the raw job posting CSV.

    Chunked reading prevents OOM errors on the 994 k-row dataset while
    keeping each chunk in memory long enough to process it.

    Parameters
    ----------
    path:
        Path to ``colorado.csv``.
    chunk_size:
        Rows per chunk.  A value of 10 000 uses ~150 MB RAM per chunk.
    active_only:
        If True, skip rows where ``expired == 'TRUE'``.
    sample_size:
        If set, stop after yielding this many total rows (useful for testing).

    Yields
    ------
    pd.DataFrame
        One chunk of job postings with a clean, typed schema.
    """
    logger.info("Streaming job postings from: %s (chunk_size=%d)", path, chunk_size)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Job posting file not found: {path}")

    # Identify which columns actually exist in the file
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in JOB_COLS_KEEP if c in header]
    missing = set(JOB_COLS_KEEP) - set(usecols)
    if missing:
        logger.warning("Expected columns not found in job file: %s", missing)

    rows_yielded = 0
    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,          # read everything as str first; cast below
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        chunk = _clean_chunk(chunk, active_only=active_only)

        if sample_size is not None:
            remaining = sample_size - rows_yielded
            if remaining <= 0:
                return
            chunk = chunk.iloc[:remaining]

        rows_yielded += len(chunk)
        logger.debug("Yielding chunk with %d rows (total so far: %d)", len(chunk), rows_yielded)
        yield chunk

        if sample_size is not None and rows_yielded >= sample_size:
            return


def load_jobs_full(
    path: str,
    active_only: bool = False,
    sample_size: Optional[int] = None,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """Load the entire job posting CSV into a single DataFrame.

    Only use this when you have sufficient RAM (≥ 8 GB for the full dataset).
    For production use, prefer ``iter_jobs`` and process chunk by chunk.
    """
    chunks = list(
        iter_jobs(path, chunk_size=chunk_size, active_only=active_only, sample_size=sample_size)
    )
    if not chunks:
        return pd.DataFrame(columns=JOB_COLS_KEEP)
    return pd.concat(chunks, ignore_index=True)


def validate_inputs(jobs_path: str, taxonomy_path: str) -> dict:
    """Run quick sanity checks and return a validation report dict."""
    report = {"jobs": {}, "taxonomy": {}}

    for label, path, rep_key in [
        ("Jobs CSV", jobs_path, "jobs"),
        ("Taxonomy CSV", taxonomy_path, "taxonomy"),
    ]:
        if not os.path.exists(path):
            report[rep_key]["status"] = "MISSING"
            report[rep_key]["error"] = f"{label} not found at {path}"
        else:
            size_mb = os.path.getsize(path) / (1024 ** 2)
            report[rep_key]["status"] = "OK"
            report[rep_key]["size_mb"] = round(size_mb, 2)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_chunk(df: pd.DataFrame, active_only: bool) -> pd.DataFrame:
    """Apply basic dtype casting and optional active-only filtering."""
    # Filter expired postings
    if active_only and "expired" in df.columns:
        df = df[df["expired"].str.upper() != "TRUE"].copy()

    # Boolean-ish columns
    for col in ["expired", "fedcontractor", "ghostjob"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper().map({"TRUE": True, "FALSE": False})

    # Numeric columns
    for col in ["parameters_salary_min", "parameters_salary_max", "parameters_hours_per_week_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date columns
    for col in ["date_acquired", "created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Categorical columns
    for col in CATEGORICAL_JOB_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Normalise system_job_id to string (used as join key)
    if "system_job_id" in df.columns:
        df["system_job_id"] = df["system_job_id"].str.strip()

    return df


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
