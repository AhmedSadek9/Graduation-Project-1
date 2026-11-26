"""WANAS Step 5: voice outputs must respect identity/verification gates."""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Task2'))
sys.path.insert(0, str(ROOT / 'Task3'))
from spoken_memory_cues import SpokenMemoryFormatter

DB = ROOT / 'Task1' / 'memory_database.json'
checks = []
def check(name, condition):
    checks.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}: {name}")

demo = SpokenMemoryFormatter(str(DB), data_mode='demo')
production = SpokenMemoryFormatter(str(DB), data_mode='production')
check('Demo greeting works with explicit confidence', demo.format_greeting_cue('person-001', .95)['status'] == 'success')
check('Production synthetic greeting is blocked', production.format_greeting_cue('person-001', .95)['status'] == 'no_verified_facts')
check('Production synthetic question is blocked', production.format_question_cue('person-001', 'Who is Omar Hassan?', .95)['status'] == 'no_verified_facts')
check('Production synthetic conversation starter is blocked', production.format_conversation_starter('person-001', .95)['status'] == 'no_verified_facts')
check('Missing confidence cannot produce spoken memory', demo.format_greeting_cue('person-001')['status'] == 'invalid_confidence_abstain')
check('Invalid confidence cannot produce spoken question', demo.format_question_cue('person-001', 'Who is Omar Hassan?', float('nan'))['status'] == 'invalid_confidence_abstain')
check('Mismatched name cannot produce spoken fact', demo.format_question_cue('person-001', 'Who is Salma Nabil?', .95)['status'] == 'person_mismatch_abstain')
check('Unknown person cannot produce spoken fact', demo.format_greeting_cue('unknown-001', .95)['status'] == 'unknown_identity_abstain')

with DB.open(encoding='utf-8') as fh:
    data = json.load(fh)
modified = {'schema_version': data['schema_version'], 'persons': [copy.deepcopy(data['persons'][0])]}
p = modified['persons'][0]
p['source_type'] = 'caregiver_provided'
p['status'] = 'active'
for f in p['facts']:
    f['source_type'] = 'caregiver_provided'
    f['verification_status'] = 'pending'
    f['review_status'] = 'draft_for_team_review'
# Allow a non-identity fact only. No inferred identity should be spoken.
nonidentity = next(f for f in p['facts'] if f['fact_type'] != 'identity_relationship' and f.get('display_tier') == 1)
nonidentity['verification_status'] = 'caregiver_verified'
nonidentity['review_status'] = 'approved'
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'db.json'
    path.write_text(json.dumps(modified), encoding='utf-8')
    prod = SpokenMemoryFormatter(str(path), data_mode='production')
    result = prod.format_greeting_cue('person-001', .95)
    check('No fabricated greeting when identity fact unapproved', result['status'] == 'no_verified_identity_fact' and 'Omar' not in result['spoken_text'])
    identity = next(f for f in p['facts'] if f['fact_type'] == 'identity_relationship')
    identity['verification_status'] = 'caregiver_verified'
    identity['review_status'] = 'approved'
    path.write_text(json.dumps(modified), encoding='utf-8')
    prod = SpokenMemoryFormatter(str(path), data_mode='production')
    check('Approved identity fact can be spoken', prod.format_greeting_cue('person-001', .95)['status'] == 'success')

try:
    SpokenMemoryFormatter(str(DB), data_mode='unsafe')
except ValueError:
    check('Invalid mode is rejected', True)
else:
    check('Invalid mode is rejected', False)
print(f'\nSTEP 5: {sum(checks)}/{len(checks)} PASSED')
if not all(checks):
    sys.exit(1)
