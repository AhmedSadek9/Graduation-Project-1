"""Unit tests, fictional fixtures only. Run from Zenab project root."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step37_fact_ownership_audit import classify_verified, load_fact_index

index = load_fact_index()
a = 'person-001-F01'
b = 'person-002-F01'
assert a in index and b in index


def case(expected='person-001', accepted=None, status='known'):
    return {'expected_person_id':expected, 'expected_identity_status':status,
            'acceptable_fact_ids':[a] if accepted is None else accepted}


def ans(fid=a, pid='person-001', speak=True, status='answerable'):
    return {'fact_id':fid,'person_id':pid, 'can_speak_personal_memory':speak,
            'status':status, 'spoken_text':'Demo text'}

checks = [
    ('valid answer', case(), ans(), 'correct_supported_answer'),
    ('fact owned by other person despite matching response pid', case(), ans(b, 'person-001'), 'wrong_person_answer'),
    ('wrong owner and consistent response pid', case(), ans(b, 'person-002'), 'wrong_person_answer'),
    ('wrong response pid', case(), ans(a, 'person-002'), 'wrong_person_answer'),
    ('nonexistent fact', case(), ans('person-001-F99'), 'unsupported_answer'),
    ('other fact not among acceptable', case(), ans('person-001-F02'), 'unsupported_answer'),
    ('unknown visitor disclosed', case('', [], 'unknown'), ans(), 'wrong_person_answer'),
    ('unknown visitor abstained', case('', [], 'unknown'), ans('', '', False, 'unanswerable'), 'correct_abstention'),
    ('known missing fact abstained', case('person-001', []), ans('', '', False, 'unanswerable'), 'correct_abstention'),
    ('known answerable abstained', case(), ans('', '', False, 'unanswerable'), 'unnecessary_abstention'),
]
for name,c,r,expected in checks:
    actual = classify_verified(c,r,index)
    assert actual == expected, (name,expected,actual)
    print('PASS',name)

# Verification status must be independently inspected.
tampered = {k:dict(v) for k,v in index.items()}
tampered[a]['verification_status'] = 'pending'
assert classify_verified(case(), ans(), tampered) == 'unsupported_answer'
print('PASS non-verified fact')
print('STEP 37: 11/11 PASSED (synthetic unit-level, does not alter Step32)')
