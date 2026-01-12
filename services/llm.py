from typing import Dict, Any, List
import requests
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OLLAMA_URL = "http://localhost:11434/api/generate"


def _fallback_explanation(payload, retrieved_chunks, raw_text=None):
    per_rule = []
    failed_rules = []

    for c in retrieved_chunks:
        failed_rules.append(c.get("rule"))
        per_rule.append({
            "rule": c.get("rule"),
            "explanation": f"According to {c.get('doc')} (page {c.get('page')}, section {c.get('section')}): {c.get('text')}",
            "citation": {
                "doc": c.get("doc"),
                "page": c.get("page"),
                "section": c.get("section"),
                "quote": c.get("text")
            }
        })

    decision = "Eligible" if payload.get("rule_results", {}).get("eligible") else "Not eligible"
    summary = f"Deterministic fallback: {decision}. LLM explanation not available right now."

    if raw_text:
        snippet = raw_text[:500] + ("..." if len(raw_text) > 500 else "")
        summary += f" LLM raw response (truncated): {snippet}"

    recommendations = []
    for rule in sorted(set(failed_rules)):
        if rule == "CAS_PRESENT":
            recommendations.append("Obtain a valid CAS from a licensed sponsor before applying.")
        elif rule == "FUNDS_28":
            recommendations.append("Ensure you hold the required maintenance funds for at least 28 consecutive days.")
        elif rule == "ENGLISH_OK":
            recommendations.append("Provide an approved English test score (SELT) at CEFR B1 or higher, or evidence of exemption.")
        elif rule == "PROVIDER_LICENSED":
            recommendations.append("Use a course provider that holds a valid sponsor licence.")
        elif rule == "ATAS_OK":
            recommendations.append("If ATAS is required for your course, obtain the ATAS certificate before applying.")
        elif rule == "COURSE_FULL_TIME":
            recommendations.append("Confirm the course is full-time as defined by the sponsor and course level.")
        elif rule == "AGE_OK":
            recommendations.append("Applicant must be at least 16 years old.")
        else:
            recommendations.append(f"Address the requirement for {rule}.")

    return {
        "decision": decision,
        "summary": summary,
        "per_rule": per_rule,
        "recommendations": recommendations
    }


def llm_explain(payload: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_prompt = (
        "You are an eligibility explanation assistant. You do not decide eligibility. "
        "Using only the provided `rule_results` and `retrieved_chunks`, produce a concise, user-friendly explanation. "
        "For each failed rule produce an entry with keys: rule, explanation, citation (with doc, page, section, quote). "
        "Return ONLY valid JSON with top-level keys: decision, summary, per_rule (list), recommendations (list). "
        "Do NOT include any additional text, markdown, or commentary outside the JSON object."
    )

    if not retrieved_chunks:
        return {
            "decision": "Not eligible" if not payload.get("rule_results", {}).get("eligible") else "Eligible",
            "summary": "No policy text was retrieved to explain this decision.",
            "per_rule": [],
            "recommendations": []
        }

    user_payload = {
        "rule_results": payload.get("rule_results"),
        "retrieved_chunks": retrieved_chunks
    }

    prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{json.dumps(user_payload, indent=2)}"

    raw_text = None
    try:
        # Use a requests Session with retry/backoff to tolerate transient failures
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Ollama can take longer for large prompts; increase timeout and allow retries
        response = session.post(OLLAMA_URL, json={
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2000
            }
        }, timeout=60)

        raw_text = response.text if response is not None else None

        if response is None or response.status_code != 200:
            return _fallback_explanation(payload, retrieved_chunks, raw_text=raw_text)

        data = response.json()
        text = (data.get("response") or "").strip()

        # Try to parse returned text as JSON (we asked for JSON). If that fails,
        # fall back to a deterministic explanation built from retrieved chunks.
        try:
            parsed = json.loads(text)
            # Ensure recommendations key exists
            if "recommendations" not in parsed:
                parsed["recommendations"] = []
            # Attach raw text and prompt snippet for debugging
            parsed["llm_raw"] = (text[:1000] + "...") if text and len(text) > 1000 else (text or "")
            return parsed
        except Exception:
            fb = _fallback_explanation(payload, retrieved_chunks, raw_text=text)
            # include raw LLM response and prompt for debugging
            fb["llm_raw"] = (text[:1000] + "...") if text and len(text) > 1000 else (text or "")
            return fb

    except requests.exceptions.ReadTimeout as e:
        # Specific guidance for read timeouts — include exception text in fallback
        fb = _fallback_explanation(payload, retrieved_chunks, raw_text=raw_text)
        fb["llm_raw"] = f"ReadTimeout: {str(e)}"
        return fb
    except requests.exceptions.RequestException as e:
        fb = _fallback_explanation(payload, retrieved_chunks, raw_text=raw_text)
        fb["llm_raw"] = f"RequestException: {str(e)}"
        return fb
    except Exception as e:
        fb = _fallback_explanation(payload, retrieved_chunks, raw_text=raw_text)
        fb["llm_raw"] = (raw_text[:1000] + "...") if raw_text and len(raw_text) > 1000 else (raw_text or str(e))
        return fb
