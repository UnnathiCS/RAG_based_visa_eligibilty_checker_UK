import csv
import os
from typing import List, Dict, Optional
import requests

# Cache for loaded sponsor names
_LICENSED_SPONSORS_CACHE: Optional[List[str]] = None

LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset_infosys', '2026-01-02_-_Worker_and_Temporary_Worker.csv')
STUDENT_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset_infosys', '2026-01-02_-_Student_and_Child_Student.csv')


def _normalize_name(n: str) -> str:
    if not n:
        return ''
    return ' '.join(n.replace('"', '').strip().split()).lower()


def load_local_sponsors(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load sponsor rows from a local CSV file and return list of dicts.

    The CSV used here (gov.uk exported) has headers including 'Organisation Name'.
    We return raw rows as dictionaries.
    """
    p = path or LOCAL_CSV_PATH
    rows: List[Dict[str, str]] = []
    if not os.path.exists(p):
        return rows

    with open(p, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v.strip() if v is not None else '') for k, v in r.items()})
    return rows


def fetch_and_parse_sponsors(url: str) -> List[Dict[str, str]]:
    """Fetch a CSV from a remote URL and parse into rows (dicts).

    This may raise requests exceptions to the caller.
    """
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    reader = csv.DictReader(lines)
    rows = [{k: (v.strip() if v is not None else '') for k, v in row.items()} for row in reader]
    return rows


def get_licensed_sponsor_names(limit: Optional[int] = None) -> List[str]:
    """Return a list of normalized sponsor organisation names (human-friendly)."""
    global _LICENSED_SPONSORS_CACHE
    if _LICENSED_SPONSORS_CACHE is not None:
        return _LICENSED_SPONSORS_CACHE[:limit] if limit else _LICENSED_SPONSORS_CACHE

    rows = load_local_sponsors()
    names = []
    # common fields that may contain organisation name
    candidate_columns = ['Organisation Name', 'OrganisationName', 'Organisation', 'Organisation name', 'Organisation Name ']
    for r in rows:
        name = None
        for col in candidate_columns:
            if col in r and r[col]:
                name = r[col]
                break
        if not name:
            # fallback to any first non-empty field
            for v in r.values():
                if v:
                    name = v
                    break
        if name:
            cleaned = ' '.join(name.replace('"', '').strip().split())
            names.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for n in names:
        key = _normalize_name(n)
        if key and key not in seen:
            seen.add(key)
            deduped.append(n)

    _LICENSED_SPONSORS_CACHE = deduped
    return deduped[:limit] if limit else deduped


def load_local_student_providers(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load student provider rows from a local CSV file and return list of dicts.

    Uses the student/provider CSV attached in dataset_infosys.
    """
    p = path or STUDENT_CSV_PATH
    rows: List[Dict[str, str]] = []
    if not os.path.exists(p):
        return rows

    with open(p, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v.strip() if v is not None else '') for k, v in r.items()})
    return rows


def get_licensed_student_provider_names(limit: Optional[int] = None) -> List[str]:
    """Return a deduplicated list of student provider names suitable for dropdowns."""
    rows = load_local_student_providers()
    names = []
    candidate_columns = ['Organisation Name', 'OrganisationName', 'Provider Name', 'Provider', 'Organisation']
    for r in rows:
        name = None
        for col in candidate_columns:
            if col in r and r[col]:
                name = r[col]
                break
        if not name:
            for v in r.values():
                if v:
                    name = v
                    break
        if name:
            cleaned = ' '.join(name.replace('"', '').strip().split())
            names.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for n in names:
        key = _normalize_name(n)
        if key and key not in seen:
            seen.add(key)
            deduped.append(n)

    return deduped[:limit] if limit else deduped


def is_licensed_student_provider(name: str) -> bool:
    if not name:
        return False
    target = _normalize_name(name)
    names = get_licensed_student_provider_names()
    normalized = [_normalize_name(n) for n in names]
    return target in normalized


def is_licensed_employer(employer_name: str) -> bool:
    """Return True if employer_name matches a licensed sponsor (case-insensitive, normalized).

    Matching is fuzzy/normalized: lowercased and whitespace-normalized equality.
    """
    if not employer_name:
        return False
    target = _normalize_name(employer_name)
    names = get_licensed_sponsor_names()
    normalized = [_normalize_name(n) for n in names]
    return target in normalized
