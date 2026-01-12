from typing import Dict, Any


def evaluate(step: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Graduate visa evaluator aligned with final eligibility schema.

    Required fields:
      currently_in_uk (bool)
      current_uk_visa_type (Student / Tier 4)
      course_completed (bool)
      course_level_completed (RQF3–RQF8)
      education_provider_is_licensed (bool)
      provider_reported_completion_to_home_office (bool)
      original_cas_reference (str)
      student_visa_valid_on_application_date (bool)
    """

    passed = []
    failed = []

    g = data.get("graduate", {})

    if g.get("currently_in_uk"):
        passed.append("CURRENTLY_IN_UK")
    else:
        failed.append("CURRENTLY_IN_UK")

    if g.get("current_uk_visa_type") in ("Student", "Tier 4"):
        passed.append("VALID_CURRENT_VISA_TYPE")
    else:
        failed.append("INVALID_CURRENT_VISA_TYPE")

    if g.get("course_completed"):
        passed.append("COURSE_COMPLETED")
    else:
        failed.append("COURSE_NOT_COMPLETED")

    if g.get("course_level_completed"):
        passed.append("COURSE_LEVEL_OK")
    else:
        failed.append("COURSE_LEVEL_MISSING")

    if g.get("education_provider_is_licensed"):
        passed.append("PROVIDER_LICENSED")
    else:
        failed.append("PROVIDER_NOT_LICENSED")

    if g.get("provider_reported_completion_to_home_office"):
        passed.append("COMPLETION_REPORTED")
    else:
        failed.append("COMPLETION_NOT_REPORTED")

    if g.get("original_cas_reference"):
        passed.append("CAS_PRESENT")
    else:
        failed.append("CAS_MISSING")

    if g.get("student_visa_valid_on_application_date"):
        passed.append("VISA_VALID_ON_APPLICATION")
    else:
        failed.append("VISA_INVALID_ON_APPLICATION")

    eligible = len(failed) == 0

    return {
        "eligible": eligible,
        "passed_rules": passed,
        "failed_rules": failed,
    }
