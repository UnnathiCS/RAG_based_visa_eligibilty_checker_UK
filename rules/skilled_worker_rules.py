from typing import Dict, Any


def evaluate(step: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skilled Worker visa evaluator aligned with final eligibility schema.

    Required fields:
      job_offer_confirmed (bool)
      employer_is_licensed_sponsor (bool)
      certificate_of_sponsorship_issued (bool)
      cos_reference_number (str)
      job_title (str)
      soc_code (str)
      job_is_eligible_occupation (bool)
      salary_offered (float/int)
      meets_minimum_salary_threshold (bool)
      english_requirement_met (bool)
      criminal_record_certificate_required (bool)
      criminal_record_certificate_provided (bool)
    """

    passed = []
    failed = []

    s = data.get("skilled_worker", {})

    if s.get("job_offer_confirmed"):
        passed.append("JOB_OFFER_CONFIRMED")
    else:
        failed.append("JOB_OFFER_MISSING")

    if s.get("employer_is_licensed_sponsor"):
        passed.append("EMPLOYER_LICENSED")
    else:
        failed.append("EMPLOYER_NOT_LICENSED")

    if s.get("certificate_of_sponsorship_issued"):
        passed.append("COS_ISSUED")
    else:
        failed.append("COS_NOT_ISSUED")

    if s.get("cos_reference_number"):
        passed.append("COS_REFERENCE_PRESENT")
    else:
        failed.append("COS_REFERENCE_MISSING")

    if s.get("job_title"):
        passed.append("JOB_TITLE_PRESENT")
    else:
        failed.append("JOB_TITLE_MISSING")

    if s.get("soc_code"):
        passed.append("SOC_CODE_PRESENT")
    else:
        failed.append("SOC_CODE_MISSING")

    if s.get("job_is_eligible_occupation"):
        passed.append("JOB_ELIGIBLE")
    else:
        failed.append("JOB_NOT_ELIGIBLE")

    if s.get("meets_minimum_salary_threshold"):
        passed.append("SALARY_THRESHOLD_MET")
    else:
        failed.append("SALARY_TOO_LOW")

    if s.get("english_requirement_met"):
        passed.append("ENGLISH_MET")
    else:
        failed.append("ENGLISH_NOT_MET")

    crc_required = s.get("criminal_record_certificate_required")
    crc_provided = s.get("criminal_record_certificate_provided")

    if crc_required:
        if crc_provided:
            passed.append("CRC_PROVIDED")
        else:
            failed.append("CRC_REQUIRED_NOT_PROVIDED")
    else:
        passed.append("CRC_NOT_REQUIRED")

    eligible = len(failed) == 0

    return {
        "eligible": eligible,
        "passed_rules": passed,
        "failed_rules": failed,
    }
