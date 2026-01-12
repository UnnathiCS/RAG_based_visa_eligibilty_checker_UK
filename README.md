RAG System for UK Visa Policy

Overview

This repository contains a Retrieval-Augmented Generation (RAG) system focused on UK visa policy documents and a Streamlit-based eligibility assistant. The implementation combines a FAISS vector store, dense text embeddings provided by SentenceTransformers, and a local large language model served via Ollama (Mistral). The system is intended to provide auditable, deterministic eligibility decisions and to supply citation-backed, human-readable explanations when deterministic rules fail.

Design principles

- Deterministic first: eligibility decisions are produced by rule-based evaluators for each visa route so results are auditable and reproducible.
- Retrieve and cite: when rules fail, the system retrieves supporting policy passages and shows document, page and section metadata alongside extracts.
- LLM for explanation only: the language model is used only to synthesise plain-language explanations and recommendations for failed checks. The deterministic evaluator controls the final decision.
- Local-first: the pipeline can run entirely locally (embeddings, FAISS index and Ollama model) to avoid external data leakage.

Components

- `streamlit_app.py` — Main Streamlit application. Offers a query interface, analytics, deterministic eligibility flows (Basic → Core → Detailed) and a compact `eligibility-final` tab that collects common fields then shows only the selected visa's fields. LLM calls are on-demand and debug information is exposed when required.
- `rag_system_uk.py` — Ingestion helper and FAISS index builder for UK policy PDFs. Handles PDF parsing, chunking strategy, embedding, and index persistence.
- `services/llm.py` — Local LLM wrapper which calls the Ollama HTTP API. The wrapper expects the model id `mistral:latest`, enforces a JSON output contract from the model, and attaches raw model output to the explanation payload for debugging if parsing fails.
- `services/retrieval.py` — Deterministic fallback retrieval mapping rule identifiers to authoritative citations and summary text.
- `rules/` — Deterministic rule engines per visa type (student_rules.py, graduate_rules.py, skilled_worker_rules.py, health_care_rules.py, visitor_rules_clean.py). Each evaluator returns a dict with `eligible`, `passed_rules`, and `failed_rules`.
- `uk_visa_db/` (expected) — Directory containing `faiss.index`, `chunks.pkl` and `metadata.json` produced by the ingestion process.

Requirements

- Python 3.10 or newer
- Recommended packages (install with pip):
  - streamlit
  - sentence-transformers
  - faiss-cpu (or faiss-gpu where appropriate)
  - numpy
  - pandas
  - requests
  - PyPDF2
  - torch

Quick setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install Python dependencies

```bash
pip install streamlit sentence-transformers faiss-cpu numpy pandas requests PyPDF2 torch
```

3. Start Ollama and ensure the Mistral model is available

The application uses Ollama as the local LLM host. The expected HTTP endpoint is `http://localhost:11434/api/generate` and the model identifier must be `mistral:latest`.

```bash
# Start the Ollama server
ollama serve
# If the model is not present, pull it once
ollama pull mistral
```

4. Build or confirm the UK FAISS index exists

If `uk_visa_db/` is missing or incomplete, run the ingestion script to build `faiss.index`, `chunks.pkl` and `metadata.json`.

```bash
python rag_system_uk.py
```

5. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL printed by the command (usually http://localhost:8501).

How the eligibility flow operates

1. Deterministic evaluation

- For each visa route, rules in `rules/*.py` accept a structured `data` dict and return whether the applicant passes each check.
- The deterministic engine is authoritative; the system will not override the deterministic decision with model output.

2. Retrieval for failed checks

- When rules fail, `retrieve_with_rag` is used to fetch supporting policy passages. This helper prefers the in-memory FAISS RAG system if loaded (`st.session_state.rag_systems['uk']`) and falls back to `services.retrieval.retrieve_policy_chunks`.
- Retrieved objects include: `doc`, `page`, `section`, and a `text` excerpt suitable for displaying to end users and for inclusion in LLM prompts.

3. LLM explanation and recommendations

- The UI calls `services.llm.llm_explain` only for failed-rule cases or on-demand. The wrapper instructs the model to emit a strictly JSON response containing these fields: `decision`, `summary`, `per_rule` (list of {rule, explanation, citation}), and `recommendations` (practical next steps).
- The Streamlit UI displays the `recommendations` and a concise model excerpt (the wrapper attaches `llm_raw` when parsing issues occur). The deterministic rule engine remains the source of truth.

Debug and troubleshooting

- Ollama errors or non-JSON responses:
  - Confirm `ollama serve` is running and reachable at `http://localhost:11434`.
  - Confirm that `ollama list` shows `mistral:latest` installed. Using a different model identifier will cause the model server to return errors in plain text.
  - Check the Streamlit UI expander "Show raw LLM response (debug)" for raw model output.

- FAISS / index problems:
  - If FAISS import fails, install `faiss-cpu` or `faiss-gpu` according to your hardware.
  - If the app reports missing `faiss.index` or `chunks.pkl`, re-run `rag_system_uk.py` to rebuild the index.

- Timeouts and slow generation:
  - Model generation can take time depending on system resources. The LLM wrapper sets a longer timeout and retry policy; if you see repeated timeouts, increase system resources or run generation on a machine with more CPU/RAM.

Rebuilding the RAG database

1. Place source UK policy PDFs into a `pdfs/` or `data/` directory.
2. Run the ingestion script which will produce `uk_visa_db/`.

```bash
python rag_system_uk.py --pdf-dir ./pdfs --out-dir ./uk_visa_db
```

3. Confirm `uk_visa_db/` contains `faiss.index`, `chunks.pkl`, and `metadata.json` before running the Streamlit app.

Files of interest

- `streamlit_app.py`: primary UI and orchestration
- `rag_system_uk.py`: ingestion and FAISS builder
- `rules/`: deterministic rule engines
- `services/llm.py`: LLM wrapper (expects `mistral:latest`)
- `services/retrieval.py`: deterministic citation fallback
- `uk_visa_db/`: persisted index and chunk store

Forms and input fields (data shape)

The application collects a common set of fields and then shows visa-specific fields depending on the selected visa. Below are the canonical field names and expected shapes used throughout the codebase.

Common fields (available in `elig_final_common` and merged into `elig_final_form`):

- `full_name`: string
- `date_of_birth`: date
- `nationality`: string
- `passport_issuing_country`: string
- `passport_issue_date`: date
- `passport_expiry_date`: date
- `current_location`: string ("Inside the UK" or "Outside the UK")
- `purpose`: string (one of the `PURPOSE_OPTIONS`)
- `travel_start`: date
- `travel_end`: date
- `funds_available`: number (GBP)
- `english_proficiency`: string ("Yes" / "No" or similar selection)
- `criminal_convictions`: string ("Yes" / "No")
- `past_refusals`: string ("Yes" / "No")
- `contact`: object with `{ email, phone, address }`

Student-specific fields (under `elig_final_form['student']`):

- `has_cas`: boolean
- `cas_reference_number`: string or null
- `provider_is_licensed`: boolean
- `course_level`: string (RQF3..RQF8)
- `course_full_time`: boolean
- `course_start_date`: date
- `course_end_date`: date
- `course_duration_months`: integer
- `meets_financial_requirement`: boolean
- `funds_held_for_28_days`: boolean
- `english_requirement_met`: boolean

Graduate-specific fields (`elig_final_form['graduate']`):

- `completed_in_uk`: string ("Yes" / "No")
- `completion_date`: date
- `current_work`: string ("Yes" / "No")
- `job_title`: string

Skilled Worker fields (`elig_final_form['skilled_worker']`):

- `job_offer`: string ("Yes" / "No")
- `soc_code`: string
- `salary`: number
- `employer`: string
- `start_date`: date

Health and Care Worker fields (`elig_final_form['health_care']`):

- `job_offer`: string ("Yes" / "No")
- `registration`: string (professional registration number)
- `soc_code`: string
- `salary`: number
- `employer`: string

Visitor-specific fields (`elig_final_form['visitor']`):

- `purpose`: string
- `accommodation_booked`: string ("Yes" / "No")
- `return_ticket_booked`: string ("Yes" / "No")
- `length_of_stay`: integer (days)

Evaluation result shape

The deterministic evaluators in `rules/*.py` return a dictionary with at least the following keys:

- `eligible`: boolean — whether the applicant passes all configured deterministic checks
- `passed_rules`: list of rule identifiers that passed
- `failed_rules`: list of rule identifiers that failed

Optional attachments produced by evaluators or the UI include:

- `financial`: object containing `requirement`, `calculation`, and `validation` details for financial checks (see `services/financial.py` for exact keys)

Session state keys used by the `eligibility-final` flow

- `elig_final_common`: dict of the common form values after the first-step submit
- `elig_final_common_submitted`: boolean guard that toggles visa-specific form visibility
- `elig_final_form`: merged dict containing `full_name`, `visa_type`, `student` / `graduate` / `skilled_worker` / `health_care` / `visitor` nested objects
- `elig_final_result`: stored evaluator output (see Evaluation result shape)
- `elig_final_retrieved`: list of retrieved citation chunks used for explanation
- `elig_final_explanation`: parsed LLM explanation payload or None

LLM explanation payload shape

The `services.llm.llm_explain` wrapper expects (and instructs the model to produce) a JSON object with these fields:

- `decision`: short string (e.g., "Eligible" / "Not eligible")
- `summary`: short paragraph summarising why the decision was reached
- `per_rule`: list of objects, each with `{ "rule": rule_id, "explanation": text, "citation": {"doc", "page", "section"} }`
- `recommendations`: list of actionable next steps for the applicant

When parsing fails, the wrapper attaches raw model output under `llm_raw` (or `_llm_raw`) so the UI can show the model text for debugging. The UI prefers to render `recommendations` and a short `llm_raw` excerpt rather than trusting the model to change the deterministic decision.

Evaluation and explanation flow (what happens on submit)

1. User completes the common form and the visa-specific form and clicks "Run eligibility-final check".
2. The app builds `elig_final_form` and calls the corresponding evaluator (for example, `evaluate_student` for Student) with the merged `data` dict.
3. Evaluator returns `elig_final_result`. The deterministic `eligible` boolean is authoritative.
4. If `failed_rules` is non-empty, the UI calls `retrieve_with_rag(failed_rules, visa_type)` to obtain supporting policy chunks (prefer FAISS; fallback to deterministic mapping in `services/retrieval.py`). These are stored in `elig_final_retrieved`.
5. The app then calls `services.llm.llm_explain({'rule_results': result}, retrieved_chunks)` on-demand (or automatically in configured flows). The wrapper parses the model JSON and returns the explanation payload which is stored in `elig_final_explanation`.
6. The UI displays:
  - The deterministic `eligible` verdict and a quick short reason derived from `failed_rules` where possible
  - The LLM `recommendations` and a short `llm_raw` excerpt (if present) for richer guidance
  - Per-rule explanations and citations (from `per_rule`) when available
  - Supporting citations (document / page / section) and short excerpts from `elig_final_retrieved`

Notes and best practices for extending rules

- Keep rule implementations deterministic and side-effect free. They should accept a data dict and return the structured result.
- Use explicit rule identifiers (strings) so UI mappings and RAG retrieval can target specific rules.
- When adding a new rule that requires citation, add a mapping entry in `services/retrieval.py` so that the fallback retrieval will return an authoritative snippet even if the RAG index is not yet available.

Checks: Basic, Core, Detailed

The application organises eligibility evaluation into three progressive steps: Basic, Core and Detailed. These map to the UI flow and to the deterministic evaluators. The following summarises the intent of each step, the typical checks performed and the primary input fields used. Example rule identifiers are provided to help you map UI failures to policy citations and to the deterministic fallback.

1) Basic checks (purpose: routing and factual validation)

- Intent: fast routing and minimal factual validation to determine whether the applicant should continue to more complex checks.
- Typical inputs used:
  - `date_of_birth`, `nationality`, `passport_issue_date`, `passport_expiry_date`, `current_location`
  - For visitors: basic funds and intended travel dates
- Typical checks performed:
  - Passport validity and minimum remaining validity (rule example: `VALID_PASSPORT`)
  - Whether the applicant is applying from inside/outside the UK (used by Graduate flow) (`IN_UK` / `NOT_IN_UK`)
  - Quick funds heuristic for short visits (`FUNDS_MINIMUM_HEURISTIC`)
  - TB requirement advisory when nationality requires it (informational only)
- Outcome:
  - If Basic passes, the UI proceeds to Core; if Basic fails, the UI shows quick deterministic reasons and offers RAG + LLM explanations on demand.

2) Core checks (purpose: deterministically evaluate the main eligibility criteria)

- Intent: the core, deterministic eligibility logic runs here. Inputs are structured and objective (dates, numeric amounts, document references).
- Typical inputs used (by visa type):
  - Student: `provider_name`, `has_cas`, `cas_number`, `provider_is_licensed`, `course_level`, `course_full_time`, `course_start_date`, `course_end_date`, `funds_amount`, `funds_held_since`, `evidence_date`, `course_fee`, `in_london`, `num_dependants`, `english_exempt_or_test`
  - Graduate: `current_location`, `current_visa_type`, `student_visa_expiry_date`, `application_date`, `course_start_date`, `course_end_date`, `completion_confirmation`
  - Skilled Worker: `job_offer`, `cos_reference`/`soc_code`, `salary`, `employer_name`
  - Health and Care Worker: `job_offer`, `cos_reference`, `registration`, `employer_name`, `occupation_code`
  - Standard Visitor: `purpose_of_visit`, `intended_stay_days`, `funds_available` (factual amount), passport fields
- Typical checks performed:
  - Student: CAS presence and validity (`CAS_PRESENT`, `CAS_VALID_AGE`, `CAS_REFERENCE_MISSING`), provider sponsor licence (`PROVIDER_LICENSED`), course mode and level checks (`COURSE_FULL_TIME`, `COURSE_LEVEL_ELIGIBLE`), and maintenance funds (`FUNDS_28`, `FUNDS_INSUFFICIENT`, `UPLOAD_MISSING`).
  - Graduate: location, visa-type prerequisite, application date vs student expiry (`NOT_IN_UK`, `NOT_ON_STUDENT_VISA`, `APPLIED_AFTER_EXPIRY`), course completion checks (`COURSE_NOT_COMPLETED`).
  - Skilled Worker / Health & Care: Certificate of Sponsorship validity, SOC code match, minimum salary threshold and employer licence checks (`COS_VALID`, `SOC_MATCH`, `SALARY_THRESHOLD`, `EMPLOYER_LICENSED`).
  - Visitor: passport validity, evidence of funds, intent to leave, length of stay limits (`INTENDS_TO_LEAVE`, `RETURN_TRAVEL`, `LENGTH_EXCEEDS_MAX`).
- Outcome:
  - Deterministic evaluation returns `passed_rules` and `failed_rules`. If there are failures the UI retrieves relevant policy chunks and offers an LLM explanation (on-demand) built from the retrieved citations.

3) Detailed checks (purpose: finalisation and evidence collection)

- Intent: finalise the decision and surface any additional evidence requests (uploads, clarifications) required to meet the deterministic rules. For some visa types this step re-runs Core (e.g., Graduate finalisation) and for others it can be a place to request file uploads or regulator confirmations.
- Typical inputs and actions:
  - Re-run core rules with any newly uploaded evidence (bank statements, sponsor letters).
  - For Student: accept uploaded financial evidence; confirm CAS details; validate passport/BRP evidence.
  - For Skilled Worker: confirm CoS and employer validation, optionally upload contract or offer letter.
  - For Visitor: accept accommodation bookings, return ticket evidence, or extra proof of ties to home country.
- Outcome:
  - Final deterministic verdict. If still failed, supporting citations and LLM recommendations are shown to the user. The deterministic `eligible` boolean remains authoritative.

Session-state keys used across the multi-step flow

- `elig_step`: which step the user is currently on (`basic`, `core`, `detailed`)
- `elig_form`: stored inputs collected across steps (basic/core/detailed) for the legacy multi-step Eligibility tab
- `elig_result`: deterministic evaluator result for the current flow (legacy eligibility tab)
- `elig_retrieved`, `elig_explanation`: retrieved citations and parsed LLM explanation for the legacy tab
- `elig_final_common`, `elig_final_common_submitted`, `elig_final_form`, `elig_final_result`, `elig_final_retrieved`, `elig_final_explanation`: session keys used by the compact `eligibility-final` tab implemented as a two-step (common → visa-specific) flow

Example rule identifier mapping (useful when adding retrieval entries)

- `CAS_PRESENT`, `CAS_VALID_AGE`, `CAS_REFERENCE_MISSING`
- `PROVIDER_LICENSED`
- `COURSE_FULL_TIME`, `COURSE_LEVEL_ELIGIBLE`
- `FUNDS_28`, `FUNDS_INSUFFICIENT`, `FUNDS_NOT_HELD_28_DAYS`, `UPLOAD_MISSING`
- `NOT_IN_UK`, `NOT_ON_STUDENT_VISA`, `APPLIED_AFTER_EXPIRY` (Graduate)
- `COS_VALID`, `SOC_MATCH`, `SALARY_THRESHOLD`, `EMPLOYER_LICENSED` (Skilled Worker)

These identifiers are referenced by the deterministic evaluators and used to drive RAG retrieval and the deterministic fallback mapping in `services/retrieval.py`.

Security and privacy

Security and privacy

Keep the Ollama endpoint local and do not expose it without proper access controls. Avoid committing large model artifacts, the FAISS index, or user-uploaded documents to version control.

License

Follow the license terms of the included open-source components. This repository provides integration code and is not a replacement for legal advice.

Contact and contribution

If you want to extend the system (new visa routes, more rigorous rule coverage, or cloud vector DB support) open an issue or submit a pull request. The codebase is structured to make those extensions straightforward.
