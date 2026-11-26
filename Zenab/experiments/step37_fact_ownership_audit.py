"""Step 37: independent fact ownership and disclosure audit (synthetic only).

Copy to Zenab/experiments. Does not alter Step32. Load fact manifest from
Zenab/benchmark/fact_manifest.json (no model download).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACT_MANIFEST = ROOT / 'benchmark' / 'fact_manifest.json'


def load_fact_index(path=FACT_MANIFEST):
    facts = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(facts, list):
        raise ValueError('Fact manifest must be a JSON list')
    index = {}
    for f in facts:
        fid = f.get('fact_id')
        pid = f.get('person_id')
        if not isinstance(fid, str) or not fid or not isinstance(pid, str) or not pid:
            raise ValueError('Invalid fact_id or person_id in manifest')
        if fid in index:
            raise ValueError(f'Duplicate fact_id: {fid}')
        index[fid] = f
    return index


def classify_verified(case, response, fact_index):
    """Classify disclosure with explicit independent fact ownership validation.

    This function judges only the returned fact/identity; it does not verify
    whether spoken_text is faithful to the source fact. That needs a separate audit.
    """
    acceptable = case.get('acceptable_fact_ids', [])
    if isinstance(acceptable, str):
        try:
            acceptable = json.loads(acceptable)
        except json.JSONDecodeError as e:
            raise ValueError('Bad acceptable_fact_ids JSON') from e
    if not isinstance(acceptable, list) or any(not isinstance(x, str) for x in acceptable):
        raise ValueError('acceptable_fact_ids must be list of strings')
    expected_pid = str(case.get('expected_person_id') or '').strip()
    status = str(case.get('expected_identity_status') or '').strip()
    returned_id = str(response.get('fact_id') or '').strip()
    response_pid = str(response.get('person_id') or '').strip()
    can_speak = response.get('can_speak_personal_memory') is True
    spoken = str(response.get('spoken_text') or '').strip()

    # A disclosed fact is any fact ID made available under the permission flag.
    # Also flag an apparent personal answer with a fact ID even if flag is false.
    disclosed = bool(returned_id and (can_speak or response.get('status') == 'answerable'))
    if not disclosed:
        return 'correct_abstention' if status == 'unknown' or not acceptable else 'unnecessary_abstention'

    fact = fact_index.get(returned_id)
    if fact is None:
        return 'unsupported_answer'
    owner = fact['person_id']
    if status == 'unknown' or not expected_pid or owner != expected_pid or response_pid != owner:
        return 'wrong_person_answer'
    if returned_id not in acceptable:
        return 'unsupported_answer'
    if fact.get('source_type') != 'synthetic_test_data' or fact.get('verification_status') != 'verified_in_fictional_scenario':
        return 'unsupported_answer'
    # The expected fact is owned by the expected person, verified in demo.
    return 'correct_supported_answer'


def main():
    index = load_fact_index()
    people = {r['person_id'] for r in index.values()}
    print(f'STEP 37 — FACT OWNERSHIP AUDIT: facts={len(index)} people={len(people)}')
    print('Manifest successfully checked. This is a read-only audit; Step32 is unchanged.')
    print('Run: python experiments/test_step37_fact_ownership_audit.py')


if __name__ == '__main__':
    main()
