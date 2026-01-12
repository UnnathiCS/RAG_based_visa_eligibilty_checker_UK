from typing import Dict, Any

# ---- Healthcare SOC allowlist for UI dropdowns ----
HEALTHCARE_SOC_CODES = {
    "1171", "1232", "2113", "2114",
    "2211", "2212",
    "2221", "2222", "2223", "2224", "2225", "2226", "2229",
    "2231", "2232", "2233", "2234", "2235", "2236", "2237",
    "2251", "2252", "2253", "2254", "2255", "2256",
    "2461", "3111", "3212",
    "6131"
}


def evaluate(step: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Health and Care Worker visa evaluator aligned with final eligibility schema.
    """

    passed = []
    failed = []

    h = data.get("health_care", {})

    if h.get("job_offer_confirmed"):
        passed.append("JOB_OFFER_CONFIRMED")
    else:
        failed.append("JOB_OFFER_MISSING")

    if h.get("employer_is_licensed_healthcare_sponsor"):
        passed.append("EMPLOYER_LICENSED")
    else:
        failed.append("EMPLOYER_NOT_LICENSED")

    if h.get("certificate_of_sponsorship_issued"):
        passed.append("COS_ISSUED")
    else:
        failed.append("COS_NOT_ISSUED")

    if h.get("cos_reference_number"):
        passed.append("COS_REFERENCE_PRESENT")
    else:
        failed.append("COS_REFERENCE_MISSING")

    if h.get("job_title"):
        passed.append("JOB_TITLE_PRESENT")
    else:
        failed.append("JOB_TITLE_MISSING")

    if h.get("soc_code"):
        passed.append("SOC_CODE_PRESENT")
    else:
        failed.append("SOC_CODE_MISSING")

    if h.get("job_is_eligible_healthcare_role"):
        passed.append("ROLE_ELIGIBLE")
    else:
        failed.append("ROLE_NOT_ELIGIBLE")

    if h.get("meets_healthcare_salary_rules"):
        passed.append("SALARY_OK")
    else:
        failed.append("SALARY_NOT_OK")

    reg_required = h.get("professional_registration_required")
    reg_provided = h.get("professional_registration_provided")

    if reg_required:
        if reg_provided:
            passed.append("REGISTRATION_PROVIDED")
        else:
            failed.append("REGISTRATION_REQUIRED_NOT_PROVIDED")
    else:
        passed.append("REGISTRATION_NOT_REQUIRED")

    if h.get("english_requirement_met"):
        passed.append("ENGLISH_MET")
    else:
        failed.append("ENGLISH_NOT_MET")

    eligible = len(failed) == 0

    return {
        "eligible": eligible,
        "passed_rules": passed,
        "failed_rules": failed,
    }
