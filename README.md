# NLx Job Posting Structuring Pipeline

> **Hackathon Submission – Problem Statement 4**
> National Labor Exchange (NLx) · Developer: Micah Sanders

---

## Overview

This pipeline transforms unstructured NLx job posting text into clean, analysable
labour-market data.  Starting from two raw inputs:

| Input | Description |
|---|---|
| `colorado.csv` | ~10.5k raw job postings (57 columns, sparse) |
| `colorado_processed.csv` | 165 k pre-mapped skill→taxonomy pairs |

It produces three structured outputs:

| Output | Description |
|---|---|
| `structured_jobs.csv` | One row per job; 30+ structured fields |
| `structured_skills.csv` | Long-format: one row per (job, taxonomy skill) pair |
| `pipeline_report.json` | Coverage rates, top skills, methodology notes |

---

## Architecture

```
colorado.csv  ──────────────────────────────────────────────────────┐
                                                                    ▼
                        ┌────────────────────────────────────────────┐
                        │              pipeline.py (CLI)             │
                        └────────────────────────────────────────────┘
                             │          │          │          │
                    ┌────────┘  ┌───────┘  ┌──────┘  ┌──────┘
                    ▼           ▼          ▼         ▼
              ingestion.py  cleaning.py  extractors.py  skill_extractor.py
              (chunked CSV  (HTML strip, (salary regex,  (vocabulary scan +
               streaming)   normalise)   edu/exp/type)   TF-IDF taxonomy map)
                    │           │          │         │
                    └───────────┴──────────┴─────────┘
                                    │
                              exporter.py
                    (structured_jobs.csv + structured_skills.csv +
                     pipeline_report.json)

colorado_processed.csv ──► build_taxonomy_index() ──► skill_extractor.py
```

### Pipeline Stages

1. **Ingestion** – Validates file existence; streams `colorado.csv` in configurable
   chunks (default 10 000 rows) to stay within RAM.  Low-cardinality columns are
   cast to `pandas.Categorical`.

2. **Cleaning** – Strips HTML tags (BeautifulSoup or regex fallback); normalises
   Unicode (NFKC); replaces URLs/phones with sentinel tokens; collapses whitespace.

3. **Field Extraction** (regex-based):
   - **Salary**: Detects `$XX,XXX/yr`, `$XX/hr`, `65K annually` patterns; merges
     with structured CSV columns, preferring the pre-structured value.
   - **Education**: Matches 5 tiers (High School → Doctoral) via keyword patterns.
   - **Experience**: Extracts minimum years from "X years of experience" patterns.
   - **Employment type**: Full-time, part-time, contract, internship.
   - **Remote work**: Boolean flag from "remote", "WFH", "hybrid", etc.
   - **Benefits**: Health insurance, 401k, PTO, tuition assistance, relocation,
     signing bonus.

4. **Skill Extraction + Taxonomy Mapping** (two-stage):
   - **Stage 1 – Candidate extraction**: Sliding window over text tokens matched
     against the raw-skill vocabulary from `colorado_processed.csv`.  Optional
     spaCy noun-chunk extraction adds higher-recall candidates.
   - **Stage 2 – Taxonomy mapping**: Exact lookup first; TF-IDF cosine similarity
     fallback.  Confidence = `tfidf_cosine × taxonomy_correlation`.
   - Supports ESCO, `onet_tech`, and `onet_skill` taxonomy sources.

5. **Export** – Writes output CSVs incrementally (append mode); generates a JSON
   report with coverage rates, top-20 taxonomy skills, and methodology notes.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: higher-recall skill extraction via noun chunks
python -m spacy download en_core_web_sm
```

### 2. Run in sample mode (1 000 rows, fast)

```bash
python pipeline.py \
  --jobs    /path/to/colorado.csv \
  --taxonomy /path/to/colorado_processed.csv \
  --output  ./output \
  --sample  1000
```

### 3. Run on full dataset

```bash
python pipeline.py \
  --jobs    /path/to/colorado.csv \
  --taxonomy /path/to/colorado_processed.csv \
  --output  ./output \
  --chunk-size 10000
```

### 4. Active postings only, with clean descriptions included

```bash
python pipeline.py \
  --jobs    /path/to/colorado.csv \
  --taxonomy /path/to/colorado_processed.csv \
  --output  ./output \
  --active-only \
  --include-description
```

### All CLI options

```
--jobs PATH            Path to colorado.csv (required)
--taxonomy PATH        Path to colorado_processed.csv (required)
--output DIR           Output directory (default: ./output)
--chunk-size N         Rows per chunk (default: 10000)
--sample N             Process only first N rows (testing)
--active-only          Skip expired postings
--min-confidence F     Min skill-match confidence (default: 0.40)
--max-skills N         Max skills per job (default: 20)
--include-description  Append cleaned description to jobs CSV
--config PATH          YAML config file path
--log-level LEVEL      DEBUG / INFO / WARNING / ERROR
```

---

## Output Schema

### `structured_jobs.csv`

| Column | Type | Source | Description |
|---|---|---|---|
| system_job_id | str | CSV | NLx system identifier |
| job_id | str | CSV | Source job ID |
| title | str | CSV | Job title |
| city / state / zipcode | str | CSV | Location |
| date_acquired / created_date | datetime | CSV | Posting dates |
| classifications_onet_code | str | CSV | O*NET SOC code |
| classifications_naics_code | str | CSV | NAICS industry code |
| fedcontractor | bool | CSV | Federal contractor flag |
| ghostjob | bool | CSV | Ghost-job flag |
| **salary_final_min** | float | CSV + text | Merged salary minimum |
| **salary_final_max** | float | CSV + text | Merged salary maximum |
| **salary_final_unit** | str | CSV + text | hourly / annual / monthly / weekly |
| **education_min** | str | text | Minimum education requirement |
| **experience_min_years** | float | text | Minimum years of experience |
| **is_fulltime / is_parttime** | bool | text | Employment type flags |
| **is_contract / is_internship** | bool | text | Employment type flags |
| **is_remote** | bool | text | Remote work indicator |
| **has_health_insurance** | bool | text | Benefits mentioned |
| **has_401k / has_pto / …** | bool | text | Benefits mentioned |
| **skill_count** | int | NLP | Number of taxonomy skills extracted |
| **top_3_skills** | str | NLP | Pipe-separated top-3 taxonomy skills |

*Bold* = fields extracted/enriched by this pipeline.

### `structured_skills.csv`

| Column | Description |
|---|---|
| system_job_id | Links back to `structured_jobs.csv` |
| raw_skill | Skill phrase as it appears in the posting |
| taxonomy_skill | Standardised ESCO / O*NET skill label |
| taxonomy_description | Full taxonomy description |
| taxonomy_source | `esco`, `onet_tech`, or `onet_skill` |
| confidence | Match confidence (0–1) |
| match_method | `exact` or `tfidf` |

---

## Methodology Notes & Known Limitations

### Transparency

- All extraction rules are **explicit regex patterns or TF-IDF similarity**—
  no black-box neural models are used for the core pipeline.  This makes every
  decision auditable.
- Confidence scores are interpretable: they are products of the TF-IDF cosine
  similarity and the pre-existing taxonomy correlation coefficient.
- The pipeline report (`pipeline_report.json`) documents coverage rates per
  field and lists all assumptions.

### Limitations

1. **Geographic scope**: Data covers Colorado only.  Skills rankings reflect the
   Colorado labour market; national extrapolation requires caution.
2. **Salary extraction**: Patterns are US-centric and assume USD.  International
   or non-standard formats will be missed.
3. **Non-English descriptions**: The pipeline does not detect language; non-English
   postings will silently produce empty extraction results.
4. **Emerging skills**: Skills absent from the `colorado_processed.csv` vocabulary
   will not be extracted at Stage 1; TF-IDF similarity provides partial coverage.
5. **Experience ranges**: "2–5 years of experience" extracts both the minimum
   (2 years) and maximum (5 years) into separate `experience_min_years` and
   `experience_max_years` columns.
6. **Description truncation**: Descriptions > 20 000 characters are truncated.

### Bias Notes

- **Vocabulary feedback loop**: The raw-skill dictionary reflects skills present
  in *existing* postings; if a skill never appeared in the training corpus it
  will never be extracted, creating a systematic blind spot for novel skills.
- **Soft-skill over-representation**: Terms like "communication" and "teamwork"
  appear frequently and will rank highly in counts, which may overstate their
  relative importance.
- **Federal contractor language**: Federal contractor postings often use different
  regulatory language that our patterns may not fully capture.

---

## Project Structure

```
nlx_pipeline/
├── pipeline.py              ← Main entry point / CLI
├── config.yaml              ← Default configuration
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── ingestion.py         ← Load & validate CSVs (chunked streaming)
│   ├── cleaning.py          ← HTML stripping, text normalisation
│   ├── extractors.py        ← Regex field extraction (salary, edu, exp)
│   ├── skill_extractor.py   ← NLP skill extraction + taxonomy mapping
│   └── exporter.py          ← Write output CSVs + quality report
└── output/                  ← Generated outputs (created at runtime)
    ├── structured_jobs.csv
    ├── structured_skills.csv
    └── pipeline_report.json
```

---

## Extending the Pipeline

| Extension | Where to add |
|---|---|
| Sentence-transformers skill matching | `skill_extractor.py` – add as alternative to TF-IDF |
| Named entity recognition (NER) | `skill_extractor.py` – replace/augment noun-chunk step |
| Multi-language support | `cleaning.py` – add `langdetect` before cleaning |
| Salary normalisation to annual | `extractors.py` – add conversion in `merge_salary` |
| Job de-duplication | `pipeline.py` – add dedup step after loading chunks |
| REST API wrapper | New `api.py` using FastAPI, calling `run_pipeline()` |
