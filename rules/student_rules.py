from datetime import date
from typing import Dict, Any
try:
    from services.tb import check_student_visa
except Exception:
    def check_student_visa(data: Dict[str, Any]):
        return {'failures': [], 'warnings': []}


def compute_age(dob: date) -> int:
    """Compute age in years from a date object. Returns 0 if dob is None."""
    if dob is None:
        return 0
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _to_date(d):
    """Convert a date or ISO-8601 string to a date object. Returns None if invalid."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        # assume ISO format YYYY-MM-DD
        return date.fromisoformat(str(d))
    except Exception:
        return None


def check_cas_within_6_months(cas_issue_date, application_date):
    """Return True if application_date is within 6 months (183 days) of cas_issue_date.

    Treats missing dates as False.
    """
    if cas_issue_date is None or application_date is None:
        return False
    try:
        return (application_date - cas_issue_date).days <= 183
    except Exception:
        return False


def _to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "1", "y")
    return bool(v)


def evaluate_student_eligibility(data: Dict[str, Any]) -> Dict[str, Any]:
    """A compact eligibility evaluator that matches the new UI contract.

    Expects a dict-like `data`. Returns {'eligible': bool, 'failures': [..]}.
    """
    failures = []

    # Helper to read fields with multiple possible keys
    def get(k, default=None):
        return data.get(k, default)

    has_cas = _to_bool(get('has_cas') or get('has_cas_bool') or get('has_cas_flag'))
    cas_reference_number = get('cas_reference_number') or get('cas_number') or get('cas')
    education_provider_is_licensed = _to_bool(get('education_provider_is_licensed') or get('provider_is_licensed') or get('provider_licensed'))
    course_level = get('course_level')
    course_full_time = _to_bool(get('course_full_time'))
    course_start_date = _to_date(get('course_start_date') or get('course_start'))
    course_end_date = _to_date(get('course_end_date') or get('course_end'))
    course_duration_months = get('course_duration_months') if get('course_duration_months') is not None else get('course_duration') or 0
    try:
        course_duration_months = int(course_duration_months)
    except Exception:
        try:
            course_duration_months = int(float(course_duration_months))
        except Exception:
            course_duration_months = 0

    meets_financial_requirement = _to_bool(get('meets_financial_requirement') or get('funds_ok') or get('meets_funds'))
    funds_held_for_28_days = _to_bool(get('funds_held_for_28_days') or get('funds_28') or get('funds_held_28_days'))
    english_requirement_met = _to_bool(get('english_requirement_met') or get('english_ok') or get('english_exempt'))

    # 1. CAS requirement
    if not has_cas:
        failures.append("NO_CAS")
    elif not cas_reference_number:
        failures.append("CAS_REFERENCE_MISSING")

    # 2. Provider must be licensed
    if not education_provider_is_licensed:
        failures.append("PROVIDER_NOT_LICENSED")

    # 3. Course level must be present
    if not course_level:
        failures.append("COURSE_LEVEL_MISSING")

    # 4. Course must be full-time
    if not course_full_time:
        failures.append("COURSE_NOT_FULL_TIME")

    # 5. Course dates must be valid
    if course_start_date and course_end_date:
        if course_start_date >= course_end_date:
            failures.append("INVALID_COURSE_DATES")
    else:
        # missing dates -> invalid
        failures.append("INVALID_COURSE_DATES")

    # 6. Course duration must be positive
    if not isinstance(course_duration_months, int) or course_duration_months <= 0:
        failures.append("INVALID_COURSE_DURATION")

    # 7. Financial requirement
    if not meets_financial_requirement:
        failures.append("FINANCIAL_REQUIREMENT_NOT_MET")
    elif not funds_held_for_28_days:
        failures.append("FUNDS_NOT_HELD_28_DAYS")

    # 8. English requirement (pass/fail only)
    if not english_requirement_met:
        failures.append("ENGLISH_REQUIREMENT_NOT_MET")

    return {
        "eligible": len(failures) == 0,
        "failures": failures
    }


def evaluate_student(step: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter that preserves the step-based API while delegating final
    eligibility checks to evaluate_student_eligibility for core/detailed steps.
    """
    # Merge any nested 'student' dict into a flat view so the evaluator works
    merged = dict(data or {})
    student_sub = {}
    try:
        student_sub = data.get('student') or {}
    except Exception:
        student_sub = {}
    if isinstance(student_sub, dict):
        # student-specific keys override top-level keys
        merged.update(student_sub)

    # For 'basic' keep a lightweight quick-check behavior
    if step == 'basic':
        passed = []
        failed = []
        dob = merged.get('date_of_birth') or merged.get('dob')
        age = compute_age(dob)
        if age >= 16:
            passed.append('AGE_OK')
        else:
            failed.append('AGE_OK')

        # minimal CAS presence check
        if merged.get('has_cas') or merged.get('has_cas_bool'):
            passed.append('CAS_PRESENT')
        else:
            failed.append('CAS_PRESENT')

        # Course level quick validation
        allowed_levels = {"RQF3", "RQF4", "RQF5", "RQF6", "RQF7", "RQF8"}
        course_level = merged.get('course_level')
        if course_level and str(course_level).upper() in allowed_levels:
            passed.append('COURSE_LEVEL_OK')
        else:
            failed.append('COURSE_LEVEL_OK')

        if merged.get('course_full_time'):
            passed.append('COURSE_FULL_TIME')
        else:
            failed.append('COURSE_FULL_TIME')

        eligible = len(failed) == 0
        return {'eligible': eligible, 'passed_rules': passed, 'failed_rules': failed}

    # For core/detailed, run the definitive eligibility function on merged data
    res = evaluate_student_eligibility(merged)
    return {'eligible': res.get('eligible', False), 'passed_rules': [], 'failed_rules': res.get('failures', [])}

