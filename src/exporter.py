"""
exporter.py
-----------
Write pipeline outputs to disk and generate a machine-readable pipeline
quality report.

Output artefacts
-----------------
1. ``structured_jobs.csv``
   One row per job posting with all structured fields:
   - Original metadata (id, title, location, dates, codes)
   - Structured salary (final merged values)
   - Extracted fields (education, experience, employment type, remote, benefits)
   - Skill summary (skill_count, top_3_skills)

2. ``structured_skills.csv``
   Long-format skill table – one row per (job, taxonomy skill) pair.
   Suitable for frequency analysis, skill co-occurrence graphs, and
   ESCO / O*NET alignment studies.

3. ``pipeline_report.json``
   JSON report covering:
   - Input file sizes and row counts
   - Cleaning statistics (empty descriptions, mean length)
   - Extraction coverage rates (% of jobs with salary, education, etc.)
   - Top 20 most frequent taxonomy skills
   - Methodology notes and known limitations

Design decisions
-----------------
- Outputs are written incrementally in append mode, so partial progress is
  preserved if the pipeline crashes mid-run on a large dataset.
- The jobs output omits the raw ``description`` column by default to keep
  file size manageable; it can be re-enabled via ``include_raw_description``.
- All floating-point columns are rounded to 4 decimal places.
"""

import json
import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns to include in the structured jobs output (ordered)
# ---------------------------------------------------------------------------

_JOB_OUTPUT_COLS = [
    # Identifiers
    "system_job_id",
    "job_id",
    "title",
    # Location
    "city",
    "state",
    "zipcode",
    "country",
    # Dates
    "date_acquired",
    "created_date",
    # Classification codes (from source)
    "classifications_onet_code",
    "classifications_naics_code",
    "cip_codes",
    "moc_codes",
    # Job metadata
    "fedcontractor",
    "ghostjob",
    "jobclass",
    "application_company",
    "parameters_hours_per_week_max",
    # Salary (merged: structured source + text fallback)
    "salary_final_min",
    "salary_final_max",
    "salary_final_unit",
    # Extracted fields
    "education_min",
    "experience_min_years",
    "experience_max_years",
    "is_fulltime",
    "is_parttime",
    "is_contract",
    "is_internship",
    "is_remote",
    # Benefits
    "has_health_insurance",
    "has_401k",
    "has_pto",
    "has_tuition",
    "has_relocation",
    "has_bonus",
    # Skill summary
    "skill_count",
    "top_3_skills",
    # Original requirements (from structured fields, if populated)
    "requirements_min_education",
    "requirements_experience",
    "requirements_license",
    # Raw description (optional)
    # "description_clean",
]

_SKILL_OUTPUT_COLS = [
    "system_job_id",
    "raw_skill",
    "taxonomy_skill",
    "taxonomy_description",
    "taxonomy_source",
    "confidence",
    "match_method",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_top_skills_column(
    jobs_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    id_col: str = "system_job_id",
    n: int = 3,
) -> pd.DataFrame:
    """Append a ``top_N_skills`` column to the jobs DataFrame.

    Parameters
    ----------
    jobs_df:
        Jobs chunk DataFrame.
    skills_df:
        Long-format skills DataFrame for the same chunk.
    id_col:
        Join key.
    n:
        Number of top skills to include (ranked by confidence).

    Returns
    -------
    pd.DataFrame
        jobs_df with ``top_3_skills`` column added (pipe-separated string).
    """
    if skills_df.empty or "taxonomy_skill" not in skills_df.columns:
        jobs_df = jobs_df.copy()
        jobs_df["top_3_skills"] = ""
        return jobs_df

    top_skills = (
        skills_df
        .sort_values("confidence", ascending=False)
        .groupby(id_col)["taxonomy_skill"]
        .apply(lambda x: " | ".join(x.drop_duplicates().head(n)))
        .rename("top_3_skills")
        .reset_index()
    )

    return jobs_df.merge(top_skills, on=id_col, how="left").fillna({"top_3_skills": ""})


def write_jobs_chunk(
    df: pd.DataFrame,
    output_path: str,
    include_raw_description: bool = False,
    is_first_chunk: bool = False,
) -> int:
    """Append a processed jobs chunk to the structured jobs CSV.

    Parameters
    ----------
    df:
        Fully processed jobs chunk.
    output_path:
        Full path to the output CSV file.
    include_raw_description:
        If True, append ``description_clean`` column to output.
    is_first_chunk:
        Write header only for the first chunk.

    Returns
    -------
    int
        Number of rows written.
    """
    cols = list(_JOB_OUTPUT_COLS)
    if include_raw_description and "description_clean" in df.columns:
        cols.append("description_clean")

    # Only keep columns that actually exist
    cols = [c for c in cols if c in df.columns]
    output_df = df[cols].copy()

    # Round floats
    for col in output_df.select_dtypes(include="float").columns:
        output_df[col] = output_df[col].round(4)

    # Convert bool columns to readable strings
    # Handles both numpy bool dtype and object-dtype columns holding Python bools
    # (e.g. fedcontractor/ghostjob which come from .map({...}) and stay object-typed)
    bool_cols = [
        c for c in output_df.columns
        if pd.api.types.is_bool_dtype(output_df[c])
        or (
            output_df[c].dtype == object
            and len(output_df[c].dropna()) > 0
            and set(output_df[c].dropna().unique()).issubset({True, False})
        )
    ]
    for col in bool_cols:
        output_df[col] = output_df[col].map({True: "Yes", False: "No"})

    mode = "w" if is_first_chunk else "a"
    output_df.to_csv(output_path, mode=mode, index=False, header=is_first_chunk)
    logger.debug("Wrote %d rows to %s", len(output_df), output_path)
    return len(output_df)


def write_skills_chunk(
    skills_df: pd.DataFrame,
    output_path: str,
    is_first_chunk: bool = False,
) -> int:
    """Append a skills chunk to the structured skills CSV."""
    if skills_df.empty:
        return 0

    cols = [c for c in _SKILL_OUTPUT_COLS if c in skills_df.columns]
    output_df = skills_df[cols].copy()

    if "confidence" in output_df.columns:
        output_df["confidence"] = output_df["confidence"].round(4)

    mode = "w" if is_first_chunk else "a"
    output_df.to_csv(output_path, mode=mode, index=False, header=is_first_chunk)
    logger.debug("Wrote %d skill rows to %s", len(output_df), output_path)
    return len(output_df)


def write_report(
    report: dict,
    output_path: str,
) -> None:
    """Write the pipeline quality report as pretty-printed JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Pipeline report written to %s", output_path)


# ---------------------------------------------------------------------------
# Report building helpers
# ---------------------------------------------------------------------------


def compute_coverage(df: pd.DataFrame) -> dict:
    """Compute extraction coverage rates for a completed jobs DataFrame.

    ``coverage`` = percentage of jobs where the field is non-null / non-empty.
    """
    total = len(df)
    if total == 0:
        return {}

    def _rate(col):
        if col not in df.columns:
            return None
        non_null = df[col].notna() & (df[col].astype(str) != "") & (df[col].astype(str) != "No Requirement Stated")
        return round(float(non_null.sum() / total * 100), 1)

    def _bool_rate(col):
        """Rate for boolean or Yes/No columns: count only True/'Yes'."""
        if col not in df.columns:
            return None
        series = df[col]
        if series.dtype == bool:
            yes_count = series.sum()
        else:
            yes_count = (series.astype(str).str.strip().isin({"Yes", "True", "true", "1"})).sum()
        return round(float(yes_count / total * 100), 1)

    return {
        "salary_coverage_pct": _rate("salary_final_min"),
        "education_coverage_pct": _rate("education_min"),
        "experience_coverage_pct": _rate("experience_min_years"),
        "remote_pct": _bool_rate("is_remote"),
        "fulltime_pct": _bool_rate("is_fulltime"),
        "parttime_pct": _bool_rate("is_parttime"),
        "contract_pct": _bool_rate("is_contract"),
        "skill_coverage_pct": _rate("skill_count"),
    }


def top_skills_summary(skills_df: pd.DataFrame, n: int = 20) -> List[dict]:
    """Return top-N most frequent taxonomy skills across the dataset."""
    if skills_df.empty or "taxonomy_skill" not in skills_df.columns:
        return []
    counts = skills_df["taxonomy_skill"].value_counts().head(n)
    return [{"taxonomy_skill": k, "job_count": int(v)} for k, v in counts.items()]


def build_final_report(
    input_validation: dict,
    cleaning_stats: dict,
    coverage: dict,
    top_skills: List[dict],
    total_jobs: int,
    total_skills: int,
    config: dict,
) -> dict:
    """Assemble the complete pipeline report dictionary."""
    return {
        "pipeline": "NLx Job Posting Structuring Pipeline",
        "version": "1.0.0",
        "methodology": {
            "cleaning": (
                "HTML tags stripped (BeautifulSoup / regex fallback); "
                "Unicode normalised (NFKC); URLs/phones replaced with tokens."
            ),
            "salary_extraction": (
                "Regex patterns matching $-prefixed amounts and K-suffix shorthand; "
                "structured CSV values preferred when available."
            ),
            "education_extraction": (
                "Pattern matching on 5 education tiers using US terminology; "
                "non-US qualifications may not be recognised."
            ),
            "experience_extraction": (
                "Regex for 'X years of experience' patterns; "
                "range expressions return the minimum value only."
            ),
            "skill_extraction": (
                "Two-stage: (1) vocabulary scan against colorado_processed.csv raw-skill dictionary; "
                "(2) TF-IDF cosine similarity fallback for unmatched candidates. "
                "spaCy noun-chunk extraction is added when en_core_web_sm is installed."
            ),
            "taxonomy_mapping": (
                "ESCO and O*NET (onet_tech, onet_skill) taxonomy labels from colorado_processed.csv. "
                "Final confidence = tfidf_cosine × taxonomy_correlation for similarity matches."
            ),
        },
        "limitations": [
            "Source data covers Colorado only; national generalisability is limited.",
            "Salary text patterns are US/USD-centric.",
            "Non-English descriptions produce empty extraction results.",
            "Emerging skills absent from the taxonomy vocabulary will not be extracted.",
            "Descriptions > 20,000 characters are truncated before extraction.",
            "Education recognition relies on US degree terminology.",
        ],
        "bias_notes": [
            "Raw-skill vocabulary is derived from existing postings, creating a feedback loop.",
            "Soft skills may be over-represented due to high mention frequency.",
            "Federal contractor postings may follow different language conventions.",
        ],
        "input_validation": input_validation,
        "cleaning_stats": cleaning_stats,
        "totals": {
            "total_jobs_processed": total_jobs,
            "total_skill_mappings": total_skills,
        },
        "extraction_coverage": coverage,
        "top_20_taxonomy_skills": top_skills,
        "config_used": {
            "taxonomy_match_threshold": config.get("processing", {}).get("taxonomy_match_threshold"),
            "sample_mode": config.get("processing", {}).get("sample_mode"),
            "sample_size": config.get("processing", {}).get("sample_size"),
        },
    }
