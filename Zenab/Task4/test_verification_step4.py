"""WANAS Step 4: synthetic demo versus caregiver-approved production gating."""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Task2'))
from memory_retrieval import MemoryRetriever

DB = ROOT / 'Task1' / 'memory_database.json'
checks = []

def check(label, condition):
    checks.append((label, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}: {label}")

# Original dataset is entirely synthetic, so it should only return facts in demo.
demo = MemoryRetriever(str(DB), data_mode='demo')
production = MemoryRetriever(str(DB), data_mode='production')
check('Demo can retrieve fictional test memories', demo.query_fact('person-001', 'Who is Omar Hassan?', 0.95)['status'] == 'answerable')
prod_result = production.query_fact('person-001', 'Who is Omar Hassan?', 0.95)
check('Production rejects synthetic person', prod_result['status'] == 'no_verified_facts' and prod_result['matched_fact'] is None)
check('Production does not disclose fictional identity fields', production.get_person_profile('person-001', 0.95)['name'] is None)

# Synthetic person converted to real, with only one fact approved.
with DB.open(encoding='utf-8') as fh:
    data = json.load(fh)
modified = {'schema_version': data['schema_version'], 'dataset_name': 'approved-test', 'persons': [copy.deepcopy(data['persons'][0])]}
p = modified['persons'][0]
p['source_type'] = 'caregiver_provided'
p['status'] = 'active'
for fact in p['facts']:
    fact['source_type'] = 'caregiver_provided'
    fact['verification_status'] = 'pending'
    fact['review_status'] = 'draft_for_team_review'
f = p['facts'][0]
f['verification_status'] = 'caregiver_verified'
f['review_status'] = 'approved'
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'db.json'
    path.write_text(json.dumps(modified), encoding='utf-8')
    r = MemoryRetriever(str(path), data_mode='production')
    result = r.get_person_profile('person-001', 0.95)
    check('Production returns only approved facts', result['status'] == 'success' and len(result['facts']) == 1 and result['facts'][0]['fact_id'] == f['fact_id'])
    check('Approved fact can answer', r.query_fact('person-001', 'Who is Omar Hassan?', 0.95)['status'] == 'answerable')
    check('Unapproved interest fact cannot answer', r.query_fact('person-001', 'What does Omar Hassan enjoy?', 0.95)['status'] == 'unanswerable')
    f['review_status'] = 'draft_for_team_review'
    path.write_text(json.dumps(modified), encoding='utf-8')
    r = MemoryRetriever(str(path), data_mode='production')
    check('Pending review rejects all facts', r.get_person_profile('person-001', 0.95)['status'] == 'no_verified_facts')

try:
    MemoryRetriever(str(DB), data_mode='anything')
except ValueError:
    check('Invalid mode rejected', True)
else:
    check('Invalid mode rejected', False)

print(f'\nSTEP 4: {sum(ok for _,ok in checks)}/{len(checks)} PASSED')
if not all(ok for _,ok in checks):
    sys.exit(1)
