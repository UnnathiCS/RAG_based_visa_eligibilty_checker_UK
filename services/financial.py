from datetime import date
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Default minimal exempt nationalities (can be replaced by a GOV.UK-provided list file)
DEFAULT_EXEMPT_NATIONALITIES = {
    'united kingdom', 'ireland', 'united states', 'canada', 'australia', 'new zealand'
}


def load_exempt_nationalities(path: str = 'dataset_infosys/financial_exempt_nationalities.json') -> set:
    p = Path(path)
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set([x.strip().lower() for x in data if isinstance(x, str)])
        except Exception:
            pass
    return DEFAULT_EXEMPT_NATIONALITIES


EXEMPT_NATIONALITIES = load_exempt_nationalities()


def _to_date(d):
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return date.fromisoformat(str(d))
    except Exception:
        return None


def check_financial_requirement(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine whether financial evidence is required and why.

    Returns dict: {
        'required': bool,
        'reasons': list of reason codes,
        'exempt_nationality': bool
    }
    """
    reasons = []
    required = True

    # 1) Exempt nationalities
    nat = (data.get('nationality') or '').strip().lower()
    if nat and nat in EXEMPT_NATIONALITIES:
        required = False
        reasons.append('NATIONALITY_EXEMPT')

    # 2) Been in the UK for at least 12 months with valid visa
    months_in_uk = int(data.get('time_spent_in_uk_months') or 0)
    if months_in_uk >= 12:
        required = False
        reasons.append('12_MONTHS_IN_UK')

    # 3) Student Union Sabbatical Officer
    if data.get('is_sabbatical_officer'):
        required = False
        reasons.append('SABBATICAL_OFFICER')

    # 4) Doctor or dentist in training
    if data.get('is_doctor_or_dentist_in_training'):
        required = False
        reasons.append('DOCTOR_DENTIST_TRAINING')

    return {'required': required, 'reasons': reasons, 'exempt_nationality': (nat in EXEMPT_NATIONALITIES)}


def calculate_required_funds(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate required funds per GOV.UK rules.

    Expects:
      - course_months (int)
      - in_london (bool)
      - course_fee (int) optional
      - num_dependants (int) optional
    Returns breakdown dict with required totals.
    """
    try:
        course_months = int(data.get('course_months') or 0)
    except Exception:
        course_months = 0
    months = min(course_months, 9) if course_months > 0 else 9

    in_london = bool(data.get('in_london'))
    course_fee = float(data.get('course_fee') or 0.0)
    num_dep = int(data.get('num_dependants') or 0)

    if in_london:
        living_rate = 1529
        dep_rate = 845
    else:
        living_rate = 1171
        dep_rate = 680

    living_cost = living_rate * months
    dependant_cost = dep_rate * months * num_dep
    total_required = course_fee + living_cost + dependant_cost

    return {
        'course_months_used': months,
        'course_fee': course_fee,
        'living_cost': living_cost,
        'dependant_cost': dependant_cost,
        'total_required': total_required,
        'rates': {
            'living_rate': living_rate,
            'dependant_rate': dep_rate
        }
    }


def validate_financial_evidence(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate submitted financial evidence.

    Expects fields:
      - funds_amount
      - funds_held_since (date)
      - evidence_date (date) (date the statement/report was issued)
      - application_date (date)
      - funds_source (str)

    Returns dict with 'valid': bool and 'fail_reasons': list
    """
    fail_reasons = []
    valid = True

    funds = float(data.get('funds_amount') or 0.0)

    # Held for at least 28 days
    held_since = _to_date(data.get('funds_held_since'))
    app_date = _to_date(data.get('application_date')) or date.today()
    evidence_date = _to_date(data.get('evidence_date')) or held_since

    if held_since is None:
        valid = False
        fail_reasons.append('FUNDS_HELD_DATE_MISSING')
    else:
        try:
            days_held = (app_date - held_since).days
            if days_held < 28:
                valid = False
                fail_reasons.append('FUNDS_NOT_HELD_28_DAYS')
        except Exception:
            valid = False
            fail_reasons.append('FUNDS_HELD_DATE_INVALID')

    # Evidence date must be within 31 days of application
    if evidence_date is None:
        valid = False
        fail_reasons.append('EVIDENCE_DATE_MISSING')
    else:
        try:
            if abs((app_date - evidence_date).days) > 31:
                valid = False
                fail_reasons.append('EVIDENCE_TOO_OLD')
        except Exception:
            valid = False
            fail_reasons.append('EVIDENCE_DATE_INVALID')

    # Source type checks
    source = (data.get('funds_source') or '').strip().lower()
    banned_sources = ['crypto', 'stocks', 'pensions', 'overdraft']
    for b in banned_sources:
        if b in source:
            valid = False
            fail_reasons.append('FUNDS_FROM_DISALLOWED_SOURCE')
            break

    return {'valid': valid, 'fail_reasons': fail_reasons, 'funds_amount': funds}
