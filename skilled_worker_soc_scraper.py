"""
skilled_worker_soc_scraper.py

Fetches the GOV.UK Skilled Worker eligible occupations page, extracts the main SOC table,
cleans it, and writes `skilled_worker_soc_codes.csv` and `skilled_worker_soc_codes.json`.

Provides a helper that loads the CSV and returns dropdown labels in the format:
    "2134 — Programmers and software development professionals"

Includes a short Streamlit selectbox usage example.

Requirements:
    pip install requests pandas beautifulsoup4 lxml
"""

from typing import List, Optional, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://www.gov.uk/government/publications/skilled-worker-visa-eligible-occupations/skilled-worker-visa-eligible-occupations-and-codes"
CSV_FILENAME = "skilled_worker_soc_codes.csv"
JSON_FILENAME = "skilled_worker_soc_codes.json"


def _remove_bracketed_notes(text: Optional[str]) -> str:
    """Remove square/round bracketed footnotes and collapse whitespace."""
    if text is None:
        return ""
    # Remove bracketed content like [1], (a), (note), etc.
    cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", str(text))
    # Replace multiple spaces/newlines/tabs with single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_column_name(col: str) -> str:
    """
    Map a raw column header to one of the canonical names:
      - "SOC code"
      - "Job type"
      - "Related job titles" (optional)
    Uses simple heuristics to be robust to minor header changes.
    """
    if col is None:
        return col
    c = col.strip().lower()
    # Common patterns mapping to SOC code
    if "soc" in c or ("code" in c and ("soc" in c or "occupation" in c or len(c) <= 6)):
        return "SOC code"
    if "occupation code" in c or ("occupation" in c and "code" in c):
        return "SOC code"
    # Job type / Main occupation / Job family / Job type
    if "job type" in c or ("occupation" in c and "title" not in c) or "main job" in c or "job family" in c:
        return "Job type"
    if "job title" in c or "related" in c or "related job" in c or "related occupation" in c:
        return "Related job titles"
    # If 'title' appears, likely related job titles
    if "title" in c:
        return "Related job titles"
    # Fallback: if header contains 'job' treat as Job type
    if "job" in c:
        return "Job type"
    # otherwise return original (unknown)
    return col.strip()


def _select_table_from_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Given a list of DataFrames returned by pandas.read_html, pick the most likely
    SOC table by heuristics:
      - contains a column mapping to 'SOC code' and 'Job type'
    If none match, return the largest table (most rows).
    """
    best = None
    for df in dfs:
        # Normalize header candidates
        headers = [str(h) for h in df.columns]
        norm = set(_normalize_column_name(h) for h in headers)
        if "SOC code" in norm and "Job type" in norm:
            # Return with renamed columns applied
            rename_map = {h: _normalize_column_name(h) for h in headers}
            renamed = df.rename(columns=rename_map)
            return renamed
        # Keep candidate with code-like column too
        if best is None or len(df) > len(best):
            best = df
    # fallback: attempt to rename columns on the largest table
    if best is not None:
        headers = [str(h) for h in best.columns]
        rename_map = {h: _normalize_column_name(h) for h in headers}
        return best.rename(columns=rename_map)
    # If still nothing, return empty DataFrame
    return pd.DataFrame()


def _parse_tables_with_bs(html: str) -> pd.DataFrame:
    """
    Fallback parser using BeautifulSoup if pandas.read_html fails or returns no suitable table.
    Attempts to find the table element that contains 'SOC' or 'Job type' in the header cells.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    candidate_dfs = []
    for table in tables:
        # collect header names from thead or first row
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(separator=" ").strip() for th in thead.find_all("th")]
        else:
            # try first row
            first_row = table.find("tr")
            if first_row:
                headers = [th.get_text(separator=" ").strip() for th in first_row.find_all(["th", "td"]) ]
        if not headers:
            continue
        headers_norm = [h.lower() for h in headers]
        if any("soc" in h for h in headers_norm) or any("job" in h for h in headers_norm):
            # parse rows
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(separator=" ").strip() for td in tr.find_all(["td", "th"])]
                if cells and len(cells) == len(headers):
                    rows.append(cells)
                elif cells and len(cells) >= 1:
                    # if row has fewer/more cells, pad or trim to header length
                    # (best-effort)
                    row = cells[: len(headers)] + [""] * max(0, len(headers) - len(cells))
                    rows.append(row)
            try:
                df = pd.DataFrame(data=rows[1:], columns=headers) if len(rows) > 1 else pd.DataFrame(columns=headers)
                candidate_dfs.append(df)
            except Exception:
                continue
    if candidate_dfs:
        # pick the one with most rows
        df = max(candidate_dfs, key=lambda d: len(d))
        # Normalize column names
        rename_map = {h: _normalize_column_name(h) for h in df.columns}
        return df.rename(columns=rename_map)
    return pd.DataFrame()


def fetch_and_extract_soc(
    url: str = DEFAULT_URL,
    csv_path: str = CSV_FILENAME,
    json_path: str = JSON_FILENAME,
    requests_timeout: int = 15,
) -> Tuple[pd.DataFrame, int]:
    """
    Fetch the GOV.UK page and extract the SOC occupation table.

    Returns:
        (df, n_rows)
    Side effects:
        Writes csv_path and json_path to the current working directory.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; skilled_worker_soc_scraper/1.0; +https://example.com/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    logger.info("Requesting page: %s", url)
    try:
        resp = requests.get(url, headers=headers, timeout=requests_timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error("Failed to fetch URL %s: %s", url, e)
        raise

    # First attempt: pandas.read_html (fast and often works on gov.uk pages)
    df = pd.DataFrame()
    try:
        # Use io.StringIO to make sure pandas reads from the HTML text
        logger.info("Attempting to parse tables with pandas.read_html")
        dfs = pd.read_html(io.StringIO(html))
        logger.info("pandas.read_html found %d tables", len(dfs))
        if dfs:
            df = _select_table_from_dfs(dfs)
    except Exception as e:
        logger.warning("pandas.read_html failed or returned no suitable table: %s", e)
        df = pd.DataFrame()

    # Fallback: BeautifulSoup parsing
    if df.empty or df.shape[1] == 0:
        logger.info("Falling back to BeautifulSoup parsing")
        df = _parse_tables_with_bs(html)

    if df.empty:
        logger.error("Could not locate a suitable SOC table on the page.")
        raise RuntimeError("SOC table not found on the provided URL; page structure may have changed.")

    # Keep only relevant columns if present
    # Map existing df columns to canonical set
    col_map = {}
    for col in df.columns:
        norm = _normalize_column_name(str(col))
        col_map[col] = norm if norm in ("SOC code", "Job type", "Related job titles") else None

    # Build a reduced DataFrame with canonical column ordering
    selected_cols = []
    for canonical in ("SOC code", "Job type", "Related job titles"):
        # find original cols mapped to canonical
        originals = [orig for orig, mapped in col_map.items() if mapped == canonical]
        if originals:
            # prefer first mapping
            selected_cols.append(originals[0])

    # If heuristics missed exact headers, try fuzzy selection (headers containing 'soc' or 'job')
    if not selected_cols:
        for col in df.columns:
            c = str(col).lower()
            if "soc" in c or "occupation code" in c or (("code" in c) and len(str(col)) <= 8):
                selected_cols.append(col)
            elif "job" in c or "title" in c:
                selected_cols.append(col)
    # Deduplicate while preserving order
    seen = set()
    selected_cols = [c for c in selected_cols if not (c in seen or seen.add(c))]

    reduced = df[selected_cols].copy() if selected_cols else df.copy()

    # Rename reduced columns to canonical names where possible
    rename_dict = {}
    for col in reduced.columns:
        rename = _normalize_column_name(str(col))
        if rename in ("SOC code", "Job type", "Related job titles"):
            rename_dict[col] = rename
        else:
            # keep original name if unknown
            rename_dict[col] = str(col)
    reduced.rename(columns=rename_dict, inplace=True)

    # Clean data: strip, remove bracketed notes, drop empty rows
    for col in reduced.columns:
        # only operate on string-like fields; convert to str then clean
        reduced[col] = reduced[col].astype(str).apply(lambda s: _remove_bracketed_notes(s).strip())

    # Drop rows where SOC code is missing or empty (common primary key)
    if "SOC code" in reduced.columns:
        reduced = reduced[reduced["SOC code"].str.strip().astype(bool)]
    else:
        # Fallback: drop rows that are entirely empty
        reduced = reduced.dropna(how="all")
        reduced = reduced[~(reduced.apply(lambda row: all((not str(x).strip()) for x in row), axis=1))]

    # Drop duplicate rows entirely
    reduced = reduced.drop_duplicates().reset_index(drop=True)

    # Final column selection: ensure canonical column order
    final_cols = [c for c in ("SOC code", "Job type", "Related job titles") if c in reduced.columns]
    reduced = reduced[final_cols]

    # Save CSV and JSON
    reduced.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(reduced.to_dict(orient="records"), jf, ensure_ascii=False, indent=2)

    n_rows = len(reduced)
    logger.info("Extracted %d rows. Saved CSV to %s and JSON to %s", n_rows, csv_path, json_path)
    print(f"Extracted {n_rows} rows.")
    return reduced, n_rows


def get_dropdown_labels_from_csv(csv_path: str = CSV_FILENAME) -> List[str]:
    """
    Load the CSV produced by fetch_and_extract_soc and return a list of labels
    in the format "SOC_CODE — Job type".

    Example:
        "2134 — Programmers and software development professionals"
    """
    df = pd.read_csv(csv_path, dtype=str)
    # Ensure columns exist
    if "SOC code" not in df.columns or "Job type" not in df.columns:
        raise RuntimeError(f"CSV {csv_path} does not contain expected columns ('SOC code', 'Job type'). Found: {list(df.columns)}")

    labels = []
    for _, row in df.iterrows():
        soc = (row.get("SOC code") or "").strip()
        job = (row.get("Job type") or "").strip()
        if not soc:
            continue
        # Clean up any trailing punctuation or stray whitespace
        label = f"{soc} — {job}" if job else f"{soc}"
        labels.append(label)
    return labels


# Short Streamlit usage example (non-executing by default in this script).
STREAMLIT_SNIPPET = """
# In a Streamlit app file (e.g. streamlit_app.py):
import streamlit as st
from skilled_worker_soc_scraper import get_dropdown_labels_from_csv

options = get_dropdown_labels_from_csv("skilled_worker_soc_codes.csv")
choice = st.selectbox("Select your job role", options)
st.write("You selected:", choice)
"""

if __name__ == "__main__":
    # Simple CLI behavior: fetch page, extract table, save files
    try:
        df, count = fetch_and_extract_soc()
        # Print a tiny preview
        print("\nPreview (first 10 rows):")
        print(df.head(10).to_string(index=False))
        print("\nStreamlit usage example (copy into your streamlit app):\n")
        print(STREAMLIT_SNIPPET)
    except Exception as exc:
        logger.exception("Failed to fetch and extract SOC table: %s", exc)
        raise
