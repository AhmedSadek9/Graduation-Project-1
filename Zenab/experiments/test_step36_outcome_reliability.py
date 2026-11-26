"""WANAS Step 36: classification and paired reliability tests (synthetic only).

Run from project root: python experiments/test_step36_outcome_reliability.py
Uses Step 32's existing categorize()/summarize(). No model/data downloads.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step32_run_paired_evaluation import categorize, summarize


def case(cid='c1', person='person-001', ids='["person-001-F01"]', status='known'):
    return {
        'case_id': cid, 'expected_identity_status': status,
        'expected_person_id': person, 'acceptable_fact_ids': ids,
    }


def response(pid=None, fid=None, speak=False):
    return {'person_id': pid, 'fact_id': fid,
            'can_speak_personal_memory': speak}


def result(cid, run, category):
    return {'case_id': cid, 'run': run, 'outcome_category': category}


def check(name, actual, expected):
    assert actual == expected, f'{name}: expected {expected!r}, got {actual!r}'
    print('PASS', name)


checks = [
    ('correct supported fact', case(), response('person-001', 'person-001-F01', True), 'correct_supported_answer'),
    ('wrong-person fact', case(), response('person-002', 'person-002-F01', True), 'wrong_person_answer'),
    ('unsupported fact for correct person', case(), response('person-001', 'person-001-F99', True), 'unsupported_answer'),
    ('unknown visitor safely refused', case(status='unknown', person='', ids='[]'), response(), 'correct_abstention'),
    ('known person with absent fact refused', case(ids='[]'), response(), 'correct_abstention'),
    ('answerable question incorrectly refused', case(), response(), 'unnecessary_abstention'),
    ('unknown visitor improperly answered', case(status='unknown', person='', ids='[]'), response('person-001', 'person-001-F01', True), 'wrong_person_answer'),
]
for name, c, r, expected in checks:
    check(name, categorize(c, r), expected)

# A has an extra case absent from D: paired comparison MUST use intersection only.
rows = [
    result('case1', 'A', 'correct_supported_answer'),
    result('case2', 'A', 'correct_abstention'),
    result('extra_A_only', 'A', 'correct_supported_answer'),
    result('case1', 'B', 'unnecessary_abstention'),
    result('case1', 'C', 'correct_supported_answer'),
    result('case1', 'D', 'unnecessary_abstention'),
    result('case2', 'D', 'correct_abstention'),
]
report = summarize(rows)
paired = report['paired_A_D']
check('paired A-D denominator', paired['paired_cases'], 2)
check('paired A correct count', paired['run_A_correct'], 2)
check('paired D correct count', paired['run_D_correct'], 1)
check('reliability loss percentage points', paired['reliability_loss_percentage_points'], 50.0)
check('B counted separately', report['run_summaries']['B']['cases'], 1)
check('C counted separately', report['run_summaries']['C']['cases'], 1)
check('empty A-D has no invented score', summarize([])['paired_A_D']['reliability_loss_percentage_points'], None)

print('STEP 36 OUTCOME / PAIRED RELIABILITY: 14/14 PASSED (synthetic logic-only)')
print('IMPORTANT: the current Step32 classifier does not independently verify fact ownership from the fact manifest.')
print('NO FACE OR STT ACTUAL OUTPUTS USED; NO E2E RELIABILITY CLAIM.')
