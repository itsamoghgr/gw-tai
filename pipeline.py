#!/usr/bin/env python3
"""
pipeline.py
-----------
NLx Job Posting Structuring Pipeline – main entry point.

Usage - by NLx
-----
# Run on full dataset (from the nlx_pipeline/ directory)
python pipeline.py \\
    --jobs    /path/to/colorado.csv \\
    --taxonomy /path/to/colorado_processed.csv \\
    --output  ./output

# Run in sample mode (first 1000 rows) for testing
python pipeline.py \\
    --jobs    /path/to/colorado.csv \\
    --taxonomy /path/to/colorado_processed.csv \\
    --output  ./output \\
    --sample  1000

# Active postings only (skip expired)
python pipeline.py ... --active-only

Pipeline stages
---------------
1. Validate & load taxonomy index (colorado_processed.csv)
2. Stream job postings in chunks (colorado.csv)
   For each chunk:
   a. Clean descriptions (HTML strip, normalise)
   b. Extract structured fields (salary, education, experience, etc.)
   c. Extract and taxonomy-map skills
   d. Merge salary from structured columns + text extraction
   e. Write to output CSVs
3. Generate pipeline quality report (JSON)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Add src/ to path so we can import without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import load_taxonomy, iter_jobs, validate_inputs, load_config
from src.cleaning import clean_jobs_chunk, get_description_stats
from src.extractors import extract_fields_batch, merge_salary
from src.skill_extractor import build_taxonomy_index, extract_and_map_skills
from src.exporter import (
    add_top_skills_column,
    write_jobs_chunk,
    write_skills_chunk,
    write_report,
    compute_coverage,
    top_skills_summary,
    build_final_report,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nlx_pipeline")


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_pipeline(
    jobs_path: str,
    taxonomy_path: str,
    output_dir: str,
    chunk_size: int = 10_000,
    sample_size: "Optional[int]" = None,
    active_only: bool = False,
    min_confidence: float = 0.40,
    max_skills: int = 20,
    include_raw_description: bool = False,
) -> dict:
    """Execute the full structuring pipeline.

    Parameters
    ----------
    jobs_path:
        Path to colorado.csv (raw job postings).
    taxonomy_path:
        Path to colorado_processed.csv (skill taxonomy mapping).
    output_dir:
        Directory to write output files.
    chunk_size:
        Rows per processing chunk.
    sample_size:
        If set, process only the first N rows (testing / demo mode).
    active_only:
        If True, skip expired postings.
    min_confidence:
        Minimum confidence for skill taxonomy matches.
    max_skills:
        Maximum number of skills to extract per job.
    include_raw_description:
        If True, include the cleaned description text in the jobs output.

    Returns
    -------
    dict
        The pipeline quality report (also written to disk as JSON).
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    jobs_output = os.path.join(output_dir, "structured_jobs.csv")
    skills_output = os.path.join(output_dir, "structured_skills.csv")
    report_output = os.path.join(output_dir, "pipeline_report.json")

    # -----------------------------------------------------------------------
    # Stage 0: Validate inputs
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("NLx Job Posting Structuring Pipeline")
    logger.info("=" * 60)

    validation = validate_inputs(jobs_path, taxonomy_path)
    for key, status in validation.items():
        if status.get("status") == "MISSING":
            logger.error("Input missing: %s", status.get("error"))
            sys.exit(1)
        logger.info("Input [%s]: %.1f MB", key, status.get("size_mb", 0))

    # -----------------------------------------------------------------------
    # Stage 1: Load taxonomy + build index
    # -----------------------------------------------------------------------
    logger.info("\n[Stage 1] Loading taxonomy and building skill index …")
    taxonomy_df = load_taxonomy(taxonomy_path, threshold=min_confidence)
    tax_index = build_taxonomy_index(taxonomy_df)

    # -----------------------------------------------------------------------
    # Stage 2: Stream and process job postings
    # -----------------------------------------------------------------------
    logger.info("\n[Stage 2] Processing job postings …")
    if sample_size:
        logger.info("SAMPLE MODE: processing first %d rows only.", sample_size)

    total_jobs = 0
    total_skills = 0
    all_cleaning_stats: list[dict] = []
    accumulated_jobs_df: list[pd.DataFrame] = []     # for final report stats
    accumulated_skills_df: list[pd.DataFrame] = []

    is_first_chunk = True

    for chunk_num, raw_chunk in enumerate(
        iter_jobs(
            jobs_path,
            chunk_size=chunk_size,
            active_only=active_only,
            sample_size=sample_size,
        ),
        start=1,
    ):
        logger.info("Chunk %d: %d rows ingested.", chunk_num, len(raw_chunk))

        # -- 2a. Clean descriptions --
        chunk = clean_jobs_chunk(raw_chunk)
        all_cleaning_stats.append(get_description_stats(chunk))

        # -- 2b. Extract structured fields --
        chunk = extract_fields_batch(chunk)

        # -- 2c. Extract skills and map to taxonomy --
        chunk, skills_chunk = extract_and_map_skills(
            chunk,
            tax_index,
            min_confidence=min_confidence,
            max_skills=max_skills,
        )

        # -- 2d. Merge salary columns --
        chunk = merge_salary(chunk)

        # -- 2e. Add top-3 skills summary column --
        chunk = add_top_skills_column(chunk, skills_chunk)

        # -- 2f. Write to disk --
        write_jobs_chunk(
            chunk,
            jobs_output,
            include_raw_description=include_raw_description,
            is_first_chunk=is_first_chunk,
        )
        write_skills_chunk(skills_chunk, skills_output, is_first_chunk=is_first_chunk)

        is_first_chunk = False
        total_jobs += len(chunk)
        total_skills += len(skills_chunk)

        # Accumulate small samples for the final report
        if total_jobs <= 50_000:
            accumulated_jobs_df.append(chunk)
        if total_skills <= 200_000:
            accumulated_skills_df.append(skills_chunk)

        elapsed = time.time() - start_time
        rate = total_jobs / elapsed if elapsed > 0 else 0
        logger.info(
            "  Processed: %d jobs total | %d skill mappings | %.0f jobs/s",
            total_jobs,
            total_skills,
            rate,
        )

    # -----------------------------------------------------------------------
    # Stage 3: Build and write quality report
    # -----------------------------------------------------------------------
    logger.info("\n[Stage 3] Generating pipeline report …")

    merged_jobs = pd.concat(accumulated_jobs_df, ignore_index=True) if accumulated_jobs_df else pd.DataFrame()
    merged_skills = pd.concat(accumulated_skills_df, ignore_index=True) if accumulated_skills_df else pd.DataFrame()

    # Aggregate cleaning stats
    agg_cleaning = {
        "total_rows": sum(s["count"] for s in all_cleaning_stats),
        "total_empty_descriptions": sum(s["empty"] for s in all_cleaning_stats),
        "avg_mean_length_chars": round(
            sum(s["mean_length"] for s in all_cleaning_stats) / max(len(all_cleaning_stats), 1), 1
        ),
    }

    config_used = {
        "processing": {
            "taxonomy_match_threshold": min_confidence,
            "sample_mode": sample_size is not None,
            "sample_size": sample_size,
        }
    }

    report = build_final_report(
        input_validation=validation,
        cleaning_stats=agg_cleaning,
        coverage=compute_coverage(merged_jobs) if not merged_jobs.empty else {},
        top_skills=top_skills_summary(merged_skills),
        total_jobs=total_jobs,
        total_skills=total_skills,
        config=config_used,
    )

    write_report(report, report_output)

    elapsed_total = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete in %.1f seconds.", elapsed_total)
    logger.info("  Jobs processed:       %d", total_jobs)
    logger.info("  Skill mappings:       %d", total_skills)
    logger.info("  Structured jobs CSV:  %s", jobs_output)
    logger.info("  Skills CSV:           %s", skills_output)
    logger.info("  Report:               %s", report_output)
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nlx_pipeline",
        description="Transform NLx unstructured job postings into structured, analysable data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jobs", required=True,
        help="Path to colorado.csv (raw job postings).",
    )
    p.add_argument(
        "--taxonomy", required=True,
        help="Path to colorado_processed.csv (skill taxonomy mapping).",
    )
    p.add_argument(
        "--output", default="./output",
        help="Directory for output files.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=10_000,
        help="Rows per processing chunk (trade RAM for speed).",
    )
    p.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Run in sample mode: process only the first N rows.",
    )
    p.add_argument(
        "--active-only", action="store_true",
        help="Skip expired job postings.",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.40,
        help="Minimum confidence threshold for skill taxonomy matches.",
    )
    p.add_argument(
        "--max-skills", type=int, default=20,
        help="Maximum number of skills to extract per job posting.",
    )
    p.add_argument(
        "--include-description", action="store_true",
        help="Include the cleaned description text in the jobs output CSV.",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to YAML config file (overrides CLI defaults if provided).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Optionally merge config file
    cfg: dict = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

    proc = cfg.get("processing", {})

    run_pipeline(
        jobs_path=args.jobs,
        taxonomy_path=args.taxonomy,
        output_dir=args.output,
        chunk_size=args.chunk_size or proc.get("chunk_size", 10_000),
        sample_size=args.sample or (proc.get("sample_size") if proc.get("sample_mode") else None),
        active_only=args.active_only,
        min_confidence=args.min_confidence or proc.get("taxonomy_match_threshold", 0.40),
        max_skills=args.max_skills or proc.get("max_skills_per_job", 20),
        include_raw_description=args.include_description,
    )


if __name__ == "__main__":
    main()
