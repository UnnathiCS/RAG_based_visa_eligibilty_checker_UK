from typing import Dict, Any


def evaluate(step: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard Visitor visa evaluator aligned with final eligibility schema.

    Required fields:
      purpose_of_visit (str)
      purpose_is_permitted_under_visitor_rules (bool)
      intended_length_of_stay (int/float)
      stay_within_6_months_limit (bool)
      accommodation_arranged (bool)
      return_or_onward_travel_planned (bool)
      intends_to_leave_uk_after_visit (bool)
      sufficient_funds_for_stay (bool)
    """

    passed = []
    failed = []
    purpose_options = [
            'Tourism / holiday',
            'Visit family or friends',
            'Volunteer (up to 30 days with a registered charity)',
            'In transit (pass through to another country)',
            'Business (meetings, interviews)',
            'Permitted paid engagement / event',
            'School exchange programme',
            'Short recreational course (up to 30 days)',
            'Study / placement / exam',
            'Academic, senior doctor or dentist',
            'Medical treatment',
            'Other (specify)'
                        ]
    v = data.get("visitor", {})

    if v.get ("purpose_of_visit"):
        passed.append("PURPOSE_APPROVED")
    else:
        failed.append("PURPOSE_MISSING")

    if v.get("purpose_is_permitted_under_visitor_rules"):
        passed.append("PURPOSE_PERMITTED")
    else:
        failed.append("PURPOSE_NOT_PERMITTED")

    if v.get("stay_within_6_months_limit"):
        passed.append("STAY_WITHIN_LIMIT")
    else:
        failed.append("STAY_TOO_LONG")

    if v.get("accommodation_arranged"):
        passed.append("ACCOMMODATION_OK")
    else:
        failed.append("ACCOMMODATION_NOT_ARRANGED")

    if v.get("return_or_onward_travel_planned"):
        passed.append("RETURN_TRAVEL_OK")
    else:
        failed.append("RETURN_TRAVEL_MISSING")

    if v.get("intends_to_leave_uk_after_visit"):
        passed.append("INTENDS_TO_LEAVE")
    else:
        failed.append("NO_INTENTION_TO_LEAVE")

    if v.get("sufficient_funds_for_stay"):
        passed.append("FUNDS_SUFFICIENT")
    else:
        failed.append("FUNDS_INSUFFICIENT")

    eligible = len(failed) == 0

    return {
        "eligible": eligible,
        "passed_rules": passed,
        "failed_rules": failed,
    }
