"""Simple retrieval fallback for Eligibility tab.

This module provides a small, deterministic mapping of rule -> citation text so the
Eligibility flow can show supporting citations even when a FAISS index is not available.
"""
from typing import List, Dict, Any


_FALLBACK = {
	'CAS_PRESENT': {
		'doc': 'UKVI Policy: Confirmation of Acceptance for Studies',
		'page': 4,
		'section': 'CAS and sponsorship',
		'text': 'A valid Confirmation of Acceptance for Studies (CAS) must be issued by a licensed sponsor prior to application.'
	},
	'PROVIDER_LICENSED': {
		'doc': 'Sponsor guidance',
		'page': 2,
		'section': 'Sponsor licensing',
		'text': 'Institutions that sponsor international students must hold a valid sponsor licence.'
	},
	'COURSE_FULL_TIME': {
		'doc': 'Student route guidance',
		'page': 6,
		'section': 'Study requirements',
		'text': 'To qualify, the course must be full-time as defined by the sponsor and course level.'
	},
	'FUNDS_28': {
		'doc': 'Maintenance funds guidance',
		'page': 3,
		'section': 'Maintenance requirement',
		'text': 'Applicants must have held the required maintenance funds in their account for at least 28 consecutive days prior to application.'
	},
	'ENGLISH_OK': {
		'doc': 'English requirements',
		'page': 1,
		'section': 'English language',
		'text': 'Applicants must meet the minimum English language requirement through an approved test or exemption.'
	},
	'ATAS_OK': {
		'doc': 'ATAS guidance',
		'page': 1,
		'section': 'ATAS',
		'text': 'Some courses require an ATAS certificate; check the subject list and apply if required.'
	},
	'AGE_OK': {
		'doc': 'General eligibility',
		'page': 1,
		'section': 'Age requirements',
		'text': 'Applicants must be at least 16 years old to be eligible for the Student route.'
	}
}


def retrieve_policy_chunks(failed_rules: List[str], visa_type: str = 'Student', top_k: int = 2) -> List[Dict[str, Any]]:
	"""Return a list of citation dicts for the provided failed rules using fallback mapping.

	Each returned dict contains: rule, doc, page, section, text
	"""
	results = []
	for rule in failed_rules:
		if rule in _FALLBACK:
			item = _FALLBACK[rule].copy()
			item['rule'] = rule
			results.append(item)
		else:
			# Generic fallback
			results.append({
				'rule': rule,
				'doc': 'Policy document',
				'page': 'N/A',
				'section': '',
				'text': 'Relevant policy text not available in local fallback.'
			})

	return results[:top_k]

