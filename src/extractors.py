"""
extractors.py
-------------
Regex-based extractors that pull structured data from cleaned job description
text.  These operate on the plain-text output of ``cleaning.clean_description``.

Extracted fields
-----------------
- **Salary**: min / max / unit (hourly, annual, monthly, weekly)
- **Education**: minimum required level (mapped to a standard tier)
- **Experience**: min years required
- **Employment type**: full-time / part-time / contract / internship
- **Remote work**: boolean indicator
- **Benefits**: boolean flags for common benefits mentioned in descriptions

Design decisions
-----------------
- Pure-regex approach means zero additional ML dependencies and deterministic
  results.  It is fast enough to run on 994 k records in a few minutes.
- All patterns are applied to the *cleaned* description (HTML stripped) to
  avoid false positives from tag attributes like ``value="35"``.
- Where a structured column already contains a value (e.g. ``parameters_salary_min``),
  the text-extracted value is only used as a fallback.
- Salary values that look implausible (< $1/hr or > $500/hr; < $1k/yr or
  > $1M/yr) are discarded and logged as warnings.

Bias / limitation notes
------------------------
- Salary patterns are US-centric and assume USD.
- Education tier mapping is based on US terminology; international
  qualifications may not be recognised.
- "2 years" in a requirement sentence is reliably extracted, but ranges
  such as "2–5 years" return the *minimum* (2 years) only.
- Remote / hybrid language is highly idiomatic; our pattern covers common
  phrases but will miss unusual formulations.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Salary patterns
# ---------------------------------------------------------------------------

# Dollar amounts: $XX, $XX.XX, $XX,XXX, $XX,XXX.XX
_DOLLAR = r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)"

# Time-unit keywords
_UNIT_HOUR = r"(?:per\s+)?(?:hour|hr\.?|hourly)"
_UNIT_YEAR = r"(?:per\s+)?(?:year|yr\.?|annual(?:ly)?)"
_UNIT_MONTH = r"(?:per\s+)?(?:month|monthly|mo\.?)"
_UNIT_WEEK = r"(?:per\s+)?(?:week|weekly|wk\.?)"

# Combined: "$30 - $45 per hour" or "$30/hr" or "$60,000 annually"
_RE_SALARY_RANGE = re.compile(
    rf"{_DOLLAR}"                               # first amount
    r"(?:\s*[-–—to]+\s*"                       # optional separator
    rf"{_DOLLAR})?"                             # optional second amount
    r"\s*(?:/\s*)?"                             # optional slash
    rf"({_UNIT_HOUR}|{_UNIT_YEAR}|{_UNIT_MONTH}|{_UNIT_WEEK})",
    re.IGNORECASE,
)

# K-suffix: "65K/yr" or "65k a year"
_RE_SALARY_K = re.compile(
    r"(\d{2,3})[kK]\s*(?:[-–—to]+\s*(\d{2,3})[kK])?\s*"
    rf"(?:/\s*)?({_UNIT_HOUR}|{_UNIT_YEAR}|{_UNIT_MONTH}|{_UNIT_WEEK})",
    re.IGNORECASE,
)

_SALARY_BOUNDS = {
    "hourly": (1.0, 500.0),
    "annual": (1_000.0, 1_000_000.0),
    "monthly": (100.0, 100_000.0),
    "weekly": (50.0, 25_000.0),
}

# ---------------------------------------------------------------------------
# Education patterns
# ---------------------------------------------------------------------------

# Ordered tiers (highest wins when multiple are mentioned)
_EDUCATION_TIERS = [
    ("doctoral",   re.compile(r"\b(?:ph\.?d|doctorate|doctoral|d\.d\.s|m\.d\.?|j\.d\.?)\b", re.I)),
    ("masters",    re.compile(r"\b(?:master'?s?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|m\.?eng)\b", re.I)),
    ("bachelors",  re.compile(r"\b(?:bachelor'?s?|b\.?s\.?|b\.?a\.?|undergraduate|4-year college|four.year)\b", re.I)),
    ("associates", re.compile(r"\b(?:associate'?s?|2-year|two.year|a\.?a\.?s?\.?)\b", re.I)),
    ("highschool", re.compile(r"\b(?:high\s+school|ged|h\.?s\.?d\.?|secondary\s+school|hsed)\b", re.I)),
]

_EDUCATION_LABEL = {
    "doctoral":   "Doctoral Degree",
    "masters":    "Master's Degree",
    "bachelors":  "Bachelor's Degree",
    "associates": "Associate's Degree",
    "highschool": "High School Diploma / GED",
    "none":       "No Requirement Stated",
}

# ---------------------------------------------------------------------------
# Experience patterns
# ---------------------------------------------------------------------------

# "2 years", "2+ years", "two years", "minimum 3 years"
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_RE_EXPERIENCE = re.compile(
    r"(?:minimum\s+of\s+|at\s+least\s+|over\s+|more\s+than\s+)?"
    r"(\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)"
    r"\+?\s*"
    r"(?:to\s+(\d+(?:\.\d+)?)\+?\s*)?"   # optional capturing upper bound ("2 to 5 years")
    r"(?:years?|yrs?)(?:\s+of)?\s+(?:relevant\s+|related\s+|prior\s+|previous\s+)?(?:work\s+)?experience",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Employment type patterns
# ---------------------------------------------------------------------------

_RE_FULLTIME = re.compile(r"\b(?:full[- ]?time|full\s+time)\b", re.I)
_RE_PARTTIME = re.compile(r"\b(?:part[- ]?time|part\s+time)\b", re.I)
_RE_CONTRACT = re.compile(r"\b(?:contract|temp(?:orary)?|contingent|freelance|gig)\b", re.I)
_RE_INTERNSHIP = re.compile(r"\b(?:intern(?:ship)?|co-?op|trainee)\b", re.I)

# ---------------------------------------------------------------------------
# Remote work patterns
# ---------------------------------------------------------------------------

_RE_REMOTE = re.compile(
    r"\b(?:remote|work\s+from\s+home|wfh|telecommut|tele-?work|virtual\s+work|distributed\s+team|fully\s+remote|hybrid)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Benefits detection (boolean flags)
# ---------------------------------------------------------------------------

_BENEFIT_PATTERNS = {
    "has_health_insurance":  re.compile(r"\b(?:health\s+insurance|medical\s+(?:benefits?|coverage)|dental|vision)\b", re.I),
    "has_401k":              re.compile(r"\b(?:401[kK]|retirement|pension|403[bB])\b", re.I),
    "has_pto":               re.compile(r"\b(?:paid\s+(?:time\s+off|vacation|pto|leave)|pto|vacation\s+days?)\b", re.I),
    "has_tuition":           re.compile(r"\b(?:tuition\s+(?:reimburse|assist|benefit)|education\s+benefit)\b", re.I),
    "has_relocation":        re.compile(r"\b(?:relocation\s+(?:assist|allowance|package|benefit))\b", re.I),
    "has_bonus":             re.compile(r"\b(?:sign(?:ing)?[\s-]on\s+bonus|performance\s+bonus|annual\s+bonus)\b", re.I),
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFields:
    """Container for all structured fields extracted from one job description."""

    # Salary (from text; may be None if not mentioned)
    salary_text_min: Optional[float] = None
    salary_text_max: Optional[float] = None
    salary_text_unit: Optional[str] = None       # "hourly" | "annual" | "monthly" | "weekly"

    # Education
    education_min: str = "No Requirement Stated"

    # Experience
    experience_min_years: Optional[float] = None
    experience_max_years: Optional[float] = None

    # Employment type (non-exclusive: a posting can be both part-time and contract)
    is_fulltime: bool = False
    is_parttime: bool = False
    is_contract: bool = False
    is_internship: bool = False

    # Remote
    is_remote: bool = False

    # Benefits
    has_health_insurance: bool = False
    has_401k: bool = False
    has_pto: bool = False
    has_tuition: bool = False
    has_relocation: bool = False
    has_bonus: bool = False

    def to_dict(self) -> dict:
        return {
            "salary_text_min": self.salary_text_min,
            "salary_text_max": self.salary_text_max,
            "salary_text_unit": self.salary_text_unit,
            "education_min": self.education_min,
            "experience_min_years": self.experience_min_years,
            "experience_max_years": self.experience_max_years,
            "is_fulltime": self.is_fulltime,
            "is_parttime": self.is_parttime,
            "is_contract": self.is_contract,
            "is_internship": self.is_internship,
            "is_remote": self.is_remote,
            "has_health_insurance": self.has_health_insurance,
            "has_401k": self.has_401k,
            "has_pto": self.has_pto,
            "has_tuition": self.has_tuition,
            "has_relocation": self.has_relocation,
            "has_bonus": self.has_bonus,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> ExtractedFields:
    """Extract all structured fields from a single cleaned description.

    Parameters
    ----------
    text:
        Output of ``cleaning.clean_description`` for one job posting.

    Returns
    -------
    ExtractedFields
        Populated dataclass with all extracted values.
    """
    result = ExtractedFields()

    result.salary_text_min, result.salary_text_max, result.salary_text_unit = _extract_salary(text)
    result.education_min = _extract_education(text)
    result.experience_min_years, result.experience_max_years = _extract_experience(text)
    result.is_fulltime = bool(_RE_FULLTIME.search(text))
    result.is_parttime = bool(_RE_PARTTIME.search(text))
    result.is_contract = bool(_RE_CONTRACT.search(text))
    result.is_internship = bool(_RE_INTERNSHIP.search(text))
    result.is_remote = bool(_RE_REMOTE.search(text))

    for benefit_key, pattern in _BENEFIT_PATTERNS.items():
        setattr(result, benefit_key, bool(pattern.search(text)))

    return result


def extract_fields_batch(df: pd.DataFrame, text_col: str = "description_clean") -> pd.DataFrame:
    """Apply ``extract_fields`` to every row in a DataFrame chunk.

    New columns are appended in-place (non-destructively).

    Parameters
    ----------
    df:
        Chunk with a cleaned description column.
    text_col:
        Name of the clean-text column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with extracted field columns appended.
    """
    df = df.copy()

    if text_col not in df.columns:
        logger.warning("Column '%s' not found; skipping field extraction.", text_col)
        return df

    extracted = df[text_col].apply(lambda t: extract_fields(t).to_dict())
    extracted_df = pd.DataFrame(extracted.tolist(), index=df.index)

    return pd.concat([df, extracted_df], axis=1)


def merge_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile structured salary columns with text-extracted salary.

    Strategy:
    - If ``parameters_salary_min`` / ``parameters_salary_max`` are populated
      (from the structured CSV fields), use them as-is.
    - Otherwise, fall back to the text-extracted ``salary_text_min`` / max.
    - The final columns are ``salary_final_min``, ``salary_final_max``,
      ``salary_final_unit``.

    This is applied *after* ``extract_fields_batch``.
    """
    df = df.copy()

    def _pick(struct_col, text_col):
        if struct_col in df.columns:
            return df[struct_col].combine_first(df.get(text_col, pd.Series(dtype=float)))
        return df.get(text_col, pd.Series(dtype=float))

    df["salary_final_min"] = _pick("parameters_salary_min", "salary_text_min")
    df["salary_final_max"] = _pick("parameters_salary_max", "salary_text_max")

    # Unit: prefer structured field; fall back to text; then default to "annual"
    if "parameters_salary_unit" in df.columns:
        df["salary_final_unit"] = (
            df["parameters_salary_unit"]
            .astype(str)
            .replace({"nan": None, "": None})
            .combine_first(df.get("salary_text_unit"))
        )
    else:
        df["salary_final_unit"] = df.get("salary_text_unit")

    # Default unit to "annual" when a salary value exists but unit is still missing
    has_salary = df["salary_final_min"].notna()
    df.loc[has_salary & df["salary_final_unit"].isna(), "salary_final_unit"] = "annual"

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_dollar(s: str) -> float:
    """'$35,000.00' → 35000.0"""
    return float(s.replace(",", "").replace("$", "").strip())


def _normalise_unit(raw_unit: str) -> str:
    """Map matched unit string to canonical label."""
    low = raw_unit.lower()
    if re.search(r"hour|hr", low):
        return "hourly"
    if re.search(r"year|annual|yr", low):
        return "annual"
    if re.search(r"month|mo", low):
        return "monthly"
    if re.search(r"week|wk", low):
        return "weekly"
    return "annual"  # safe default


def _salary_plausible(value: float, unit: str) -> bool:
    lo, hi = _SALARY_BOUNDS.get(unit, (0, float("inf")))
    return lo <= value <= hi


def _extract_salary(text: str):
    """Return (min, max, unit) or (None, None, None)."""
    # Try dollar-sign pattern first
    match = _RE_SALARY_RANGE.search(text)
    if match:
        try:
            amt1 = _parse_dollar(match.group(1))
            amt2 = _parse_dollar(match.group(2)) if match.group(2) else None
            unit = _normalise_unit(match.group(3))
            sal_min = min(amt1, amt2) if amt2 else amt1
            sal_max = max(amt1, amt2) if amt2 else None
            if _salary_plausible(sal_min, unit):
                return sal_min, sal_max, unit
        except (ValueError, AttributeError):
            pass

    # Try K-suffix pattern
    match_k = _RE_SALARY_K.search(text)
    if match_k:
        try:
            amt1 = float(match_k.group(1)) * 1_000
            amt2 = float(match_k.group(2)) * 1_000 if match_k.group(2) else None
            unit = _normalise_unit(match_k.group(3))
            sal_min = min(amt1, amt2) if amt2 else amt1
            sal_max = max(amt1, amt2) if amt2 else None
            if _salary_plausible(sal_min, unit):
                return sal_min, sal_max, unit
        except (ValueError, AttributeError):
            pass

    return None, None, None


def _extract_education(text: str) -> str:
    """Return the highest education tier mentioned in text."""
    for tier_key, pattern in _EDUCATION_TIERS:
        if pattern.search(text):
            return _EDUCATION_LABEL[tier_key]
    return _EDUCATION_LABEL["none"]


def _extract_experience(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (min_years, max_years) of experience mentioned, or (None, None)."""
    match = _RE_EXPERIENCE.search(text)
    if not match:
        return None, None

    raw_min = match.group(1).lower()
    try:
        min_years = float(raw_min)
    except ValueError:
        min_years = float(_WORD_NUMS.get(raw_min, 0))

    if not (0 < min_years <= 50):
        return None, None

    max_years: Optional[float] = None
    raw_max = match.group(2)
    if raw_max is not None:
        try:
            candidate = float(raw_max)
            if 0 < candidate <= 50 and candidate >= min_years:
                max_years = candidate
        except ValueError:
            pass

    return min_years, max_years
