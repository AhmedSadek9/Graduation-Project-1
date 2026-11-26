"""Step25: PROPOSED experimental extensions to the Step15 intent guard.
Not production rules. No model loading. Never uses labels, fact IDs, or split.
"""
from __future__ import annotations
import re
from fact_type_guard_step15 import allowed_fact_types as frozen_guard


def proposed_allowed_fact_types(question: str):
    allowed, reason = frozen_guard(question)
    # Preserve all existing explicit matches, especially contact-info refusal.
    if allowed is not None:
        return allowed, reason
    q = ' '.join(str(question).casefold().split())
    # A collection is a personal interest, not a shared event.
    if re.search(r'\b(what|which) collection (?:does|did)\b', q):
        return {'personal_interest'}, 'step25_collection_interest_PROPOSAL'
    # Specific episodic-memory cues absent from the frozen rules.
    if re.search(r'\b(?:at (?:a |the )?(?:family gathering|picnic)|on (?:a |the )?family trip)\b', q):
        if re.search(r'\b(what did|where did|what (?:happened|event))\b', q):
            return {'shared_memory'}, 'step25_shared_event_PROPOSAL'
    return allowed, reason
