"""Step 3 additional regression tests. Run from the Wanas31 root:
python Task4/test_identity_mismatch_step3.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Task2'))
from memory_retrieval import MemoryRetriever

r = MemoryRetriever(str(ROOT / 'Task1' / 'memory_database.json'))
cases = [
    ('person-001', 'Who is Salma Nabil?', 'person_mismatch_abstain'),
    ('person-001', 'What can I talk to Salma Nabil about?', 'person_mismatch_abstain'),
    ('person-002', 'Who is Omar Hassan?', 'person_mismatch_abstain'),
    ('person-001', 'Who is person-002?', 'person_mismatch_abstain'),
    ('person-001', 'Who is Omar Hassan and Salma Nabil?', 'person_mismatch_abstain'),
    ('person-001', 'Who is Omar Hassan?', 'answerable'),
    ('person-002', 'Who is Salma Nabil?', 'answerable'),
    ('person-001', 'What is Omar Hassan favorite color?', 'unanswerable'),
    ('unknown-001', 'Who is Salma Nabil?', 'unknown_identity_abstain'),
    ('person-001', 'Who is Salma Nabil?', 'low_confidence_abstain', 0.3),
]
for i, case in enumerate(cases, 1):
    pid, q, expected, *score = case
    result = r.query_fact(pid, q, recognition_confidence=score[0] if score else .95)
    actual = result['status']
    assert actual == expected, f'Case {i} expected {expected}, got {actual}: {q}'
    if expected == 'person_mismatch_abstain':
        assert result['matched_fact'] is None, f'Case {i}: leaked fact'
    print(f'{i:02d}. PASS  [{actual}] {q}')
print(f'\nSTEP 3: {len(cases)}/{len(cases)} PASSED')
