"""
services/tb.py

Functions to fetch TB required countries from GOV.UK and to validate passport and TB test dates.

Functions:
- fetch_tb_required_countries() -> set[str]
- check_passport(passport_issue, passport_expiry, application_date, passport_country, nationality) -> (failures, warnings)
- check_tb_test(nationality, tb_test_date, application_date, tb_required_countries) -> list[str]
- check_student_visa(data) -> dict with failures/warnings

Caching: the fetched country list is cached to `dataset_infosys/tb_countries.json` in the workspace.

Requirements: requests, beautifulsoup4
"""
from __future__ import annotations
from typing import Set, List, Tuple, Dict, Any
import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import date, timedelta

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset_infosys', 'tb_countries.json')
GOV_TB_URL = 'https://www.gov.uk/tb-test-visa/countries-where-you-need-a-tb-test-to-enter-the-uk'

# Authoritative fallback list (from user-provided copy of GOV.UK page)
DEFAULT_TB_COUNTRIES = {
    'Afghanistan','Algeria','Angola','Armenia','Azerbaijan','Bangladesh','Belarus','Benin','Bhutan',
    'Bolivia','Botswana','Brunei','Burkina Faso','Burundi','Cambodia','Cape Verde','Central African Republic',
    'Chad','Cameroon','China','Congo','Côte d’Ivoire','Democratic Republic of the Congo','Djibouti',
    'Dominican Republic','East Timor','Ecuador','Equatorial Guinea','Eritrea','Ethiopia','Gabon','Gambia',
    'Georgia','Ghana','Guatemala','Guinea','Guinea Bissau','Guyana','Haiti','Hong Kong','India','Indonesia',
    'Iraq','Kazakhstan','Kenya','Kiribati','Kyrgyzstan','Laos','Lesotho','Liberia','Macau','Madagascar',
    'Malawi','Malaysia','Mali','Marshall Islands','Mauritania','Micronesia, Federated States of','Moldova','Mongolia','Morocco',
    'Mozambique','Myanmar','Namibia','Nepal','Niger','Nigeria','North Korea','Pakistan','Palau','Papua New Guinea',
    'Panama','Paraguay','Peru','Philippines','Russia','Rwanda','São Tomé and Príncipe','Senegal','Sierra Leone',
    'Solomon Islands','Somalia','South Africa','South Korea','South Sudan','Sri Lanka','Sudan','Suriname','Eswatini',
    'Tajikistan','Tanzania','Togo','Thailand','Turkmenistan','Tuvalu','Uganda','Ukraine','Uzbekistan','Vanuatu',
    'Vietnam','Zambia','Zimbabwe'
}

# Conservative default list (kept internally) — includes the official countries list commonly published
DEFAULT_TB_COUNTRIES = {
    'Afghanistan','Algeria','Angola','Armenia','Azerbaijan','Bangladesh','Belarus','Benin','Bhutan',
    'Bolivia','Botswana','Brunei','Burkina Faso','Burundi','Cambodia','Cape Verde','Central African Republic',
    'Chad','Cameroon','China','Congo','Côte d’Ivoire','Democratic Republic of the Congo','Djibouti',
    'Dominican Republic','East Timor','Ecuador','Equatorial Guinea','Eritrea','Ethiopia','Gabon','Gambia',
    'Georgia','Ghana','Guatemala','Guinea','Guinea Bissau','Guyana','Haiti','Hong Kong','India','Indonesia',
    'Iraq','Kazakhstan','Kenya','Kiribati','Kyrgyzstan','Laos','Lesotho','Liberia','Macau','Madagascar',
    'Malawi','Malaysia','Mali','Marshall Islands','Mauritania','Micronesia','Moldova','Mongolia','Morocco',
    'Mozambique','Myanmar','Namibia','Nepal','Niger','Nigeria','North Korea','Pakistan','Palau','Papua New Guinea',
    'Panama','Paraguay','Peru','Philippines','Russia','Rwanda','Sao Tome and Principe','Senegal','Sierra Leone',
    'Solomon Islands','Somalia','South Africa','South Korea','South Sudan','Sri Lanka','Sudan','Suriname','Swaziland',
    'Tajikistan','Tanzania','Togo','Thailand','Turkmenistan','Tuvalu','Uganda','Ukraine','Uzbekistan','Vanuatu',
    'Vietnam','Zambia','Zimbabwe'
}


def fetch_tb_required_countries() -> Set[str]:
    """Fetch and parse GOV.UK TB required countries and cache the result locally.

    Returns a set of country names (strings).
    This function first tries to load a cached JSON file. If missing or invalid, it fetches
    the GOV.UK page and extracts country names from the main content list.
    """
    # Try load cache
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Validate cache looks like country names by checking overlap with defaults
                    lower_defaults = set(d.lower() for d in DEFAULT_TB_COUNTRIES)
                    overlap = sum(1 for item in data if isinstance(item, str) and item.strip().lower() in lower_defaults)
                    if overlap >= 1:
                        return set(data)
                    # otherwise ignore stale/incorrect cache and re-fetch
    except Exception:
        # ignore and re-fetch
        pass

    # Fetch page
    try:
        resp = requests.get(GOV_TB_URL, timeout=10)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        # If network fails and no cache present, return a conservative fallback set
        # so the UI can still prompt applicants correctly (includes India and other common entries).
        fallback = {
            'India', 'Pakistan', 'Bangladesh', 'Nigeria', 'Somalia', 'Ethiopia', 'Kenya', 'China',
            'Vietnam', 'Indonesia', 'Philippines'
        }
        return fallback

    soup = BeautifulSoup(html, 'lxml')

    # Heuristic: find the section that contains the heading mentioning "countries" and then collect list items
    country_names = []
    # Look for headings that indicate the list
    headings = soup.find_all(['h2', 'h3', 'h4'])
    target = None
    for h in headings:
        if 'countries' in h.get_text(separator=' ').lower() or 'where you need a tb test' in h.get_text(separator=' ').lower():
            target = h
            break

    if target:
        # try to find the following ul/ol
        sibling = target.find_next_sibling()
        # walk siblings until we find a list
        while sibling and sibling.name not in ('ul', 'ol'):
            sibling = sibling.find_next_sibling()
        if sibling and sibling.name in ('ul', 'ol'):
            for li in sibling.find_all('li'):
                txt = li.get_text(separator=' ').strip()
                if txt:
                    country_names.append(txt)

    # Fallback: search for lists on the page that contain many country-like items (commas, capitalized words)
    if not country_names:
        lists = soup.find_all(['ul', 'ol'])
        for l in lists:
            items = [li.get_text(separator=' ').strip() for li in l.find_all('li')]
            # crude heuristic: list with > 10 items and items contain words with initial capitals
            if len(items) >= 10:
                country_names.extend(items)
                break

    # Clean names: remove footnotes, parentheses content
    cleaned = set()
    import re
    for name in country_names:
        # strip bracketed/parenthesized content
        n = re.sub(r"\[.*?\]", "", name)
        n = re.sub(r"\(.*?\)", "", n)
        n = n.strip()
        if n:
            cleaned.add(n)

    # Normalize cleaned names (strip)
    normalized = set(n.strip() for n in cleaned if n and n.strip())

    # Save cache
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(normalized)), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Merge parsed results with DEFAULT_TB_COUNTRIES to ensure common entries like India are present
    result_set = set(normalized) | set(DEFAULT_TB_COUNTRIES)
    return result_set


def check_passport(passport_issue, passport_expiry, application_date, passport_country: str, nationality: str) -> Tuple[List[str], List[str]]:
    """Check passport dates and country mismatch.

    Returns (failures, warnings) where failures are blocking errors and warnings are non-blocking.

    Rules:
    - Issue date must be before expiry -> failure 'PASSPORT_ISSUE_ORDER'
    - Expiry date must be after application date -> failure 'PASSPORT_EXPIRES_BEFORE_APPLICATION'
    - If passport_country != nationality -> warning 'PASSPORT_COUNTRY_MISMATCH'

    Dates may be date objects or ISO strings. Invalid/missing dates produce failures.
    """
    failures: List[str] = []
    warnings: List[str] = []

    def to_date(d):
        if d is None:
            return None
        if isinstance(d, date):
            return d
        try:
            return date.fromisoformat(str(d))
        except Exception:
            return None

    issue = to_date(passport_issue)
    expiry = to_date(passport_expiry)
    app_date = to_date(application_date)

    if issue is None or expiry is None:
        failures.append('PASSPORT_DATES_MISSING_OR_INVALID')
        return failures, warnings

    if issue >= expiry:
        failures.append('PASSPORT_ISSUE_ORDER')

    if app_date is None:
        # if application date missing, we cannot validate expiry relative to it; treat as warning
        warnings.append('APPLICATION_DATE_MISSING')
    else:
        if expiry <= app_date:
            failures.append('PASSPORT_EXPIRES_BEFORE_APPLICATION')

    # Country mismatch warning
    if passport_country and nationality and passport_country.strip().lower() != nationality.strip().lower():
        warnings.append('PASSPORT_COUNTRY_MISMATCH')

    return failures, warnings


def check_tb_test(nationality: str, tb_test_date, application_date, tb_required_countries: Set[str]) -> List[str]:
    """Validate TB test requirement. Return list of failures (empty if pass).

    Rules:
    - If nationality not in tb_required_countries -> pass (no failures)
    - If nationality in list:
      - tb_test_date is required -> failure 'TB_TEST_MISSING'
      - tb_test_date must be within 6 months of application_date -> failure 'TB_TEST_TOO_OLD'
    """
    failures: List[str] = []

    if not nationality:
        return failures

    # Case-insensitive membership check
    tb_norm = set(c.strip().lower() for c in tb_required_countries if isinstance(c, str))
    if nationality.strip().lower() not in tb_norm:
        return failures

    # nationality requires TB test
    def to_date(d):
        if d is None:
            return None
        if isinstance(d, date):
            return d
        try:
            return date.fromisoformat(str(d))
        except Exception:
            return None

    test_date = to_date(tb_test_date)
    app_date = to_date(application_date) or date.today()

    if test_date is None:
        failures.append('TB_TEST_MISSING')
        return failures

    # within 6 months (approx 183 days)
    if (app_date - test_date).days > 183:
        failures.append('TB_TEST_TOO_OLD')

    return failures


def check_student_visa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run passport and TB checks and return dict with 'failures' and 'warnings'.

    Expects keys in data: 'passport_issue_date', 'passport_expiry_date', 'application_date',
    'passport_country', 'nationality', 'tb_test_date'.
    """
    failures: List[str] = []
    warnings: List[str] = []

    tb_countries = fetch_tb_required_countries()

    p_failures, p_warnings = check_passport(
        data.get('passport_issue_date'),
        data.get('passport_expiry_date'),
        data.get('application_date'),
        data.get('passport_country'),
        data.get('nationality')
    )
    failures.extend(p_failures)
    warnings.extend(p_warnings)

    tb_failures = check_tb_test(
        data.get('nationality'),
        data.get('tb_test_date'),
        data.get('application_date'),
        tb_countries
    )
    failures.extend(tb_failures)

    return {'failures': failures, 'warnings': warnings}
