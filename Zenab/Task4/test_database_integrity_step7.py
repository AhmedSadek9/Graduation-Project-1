"""WANAS Step 7: read-only integrity and handover checks.

Place this file in Wanas31/Task4. Run from anywhere with:
    python Task4/test_database_integrity_step7.py
No third-party dependencies; does NOT modify the database.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Task1'
checks = []

def record(label, ok, detail=''):
    checks.append((label, bool(ok), detail))

def load(name):
    with (DATA / name).open(encoding='utf-8-sig') as f:
        return json.load(f)

db = load('memory_database.json')
schema = load('memory_schema.json')
queries = load('retrieval_test_queries.json')
unknown = load('unknown_test_cases.json')
people = db.get('persons', [])
ids = [p.get('person_id') for p in people]
all_facts = [(p.get('person_id'), f) for p in people for f in p.get('facts', [])]
fact_ids = [f.get('fact_id') for _, f in all_facts]

record('Database has required top-level fields', all(k in db for k in ('schema_version','dataset_name','persons')))
record('20 unique known person IDs', len(people) == len(set(ids)) == 20 and all(re.fullmatch(r'person-\d{3}', x or '') for x in ids))
record('Every profile has six facts', all(len(p.get('facts', [])) == 6 for p in people))
record('120 unique fact IDs', len(fact_ids) == len(set(fact_ids)) == 120)
record('Every fact ID belongs to its person', all(isinstance(f.get('fact_id'),str) and f['fact_id'].startswith(pid + '-F') for pid,f in all_facts))
record('Person identity fields are nonempty', all(isinstance(p.get(k),str) and p[k].strip() for p in people for k in ('name','relationship_to_wearer')))
record('Each fact has required provenance and review metadata', all(all(isinstance(f.get(k),str) and f[k].strip() for k in ('source_type','verification_status','review_status','last_updated','memory_fact')) for _,f in all_facts))
# Avoid letting one invalid date stop reporting other tests.
def valid_date(value):
    try:
        return isinstance(value,str) and date.fromisoformat(value).isoformat() == value
    except (TypeError,ValueError):
        return False
record('Last-updated dates are valid ISO dates', all(valid_date(f.get('last_updated')) for _,f in all_facts))
record('Demo records marked as synthetic, not real patient data', all(p.get('source_type') == 'synthetic_test_data' and p.get('status') == 'fictional_for_testing' for p in people) and all(f.get('source_type') == 'synthetic_test_data' and f.get('review_status') == 'draft_for_team_review' for _,f in all_facts))
record('Schema defines mandatory person and fact fields', schema.get('properties',{}).get('persons',{}).get('items',{}).get('type') == 'object' and set(('person_id','name','relationship_to_wearer','facts')).issubset(schema['properties']['persons']['items'].get('required',[])))
record('Schema restricts fact types and display tiers', set(schema['properties']['persons']['items']['properties']['facts']['items']['properties']['display_tier'].get('enum',[])) == {1,2} and 'identity_relationship' in schema['properties']['persons']['items']['properties']['facts']['items']['properties']['fact_type'].get('enum',[]))
record('160 queries with unique query IDs', len(queries) == 160 and len({q.get('query_id') for q in queries}) == 160)
valid_outcomes = {'answerable','unanswerable','unknown_identity_abstain'}
record('Every evaluation query has a valid expected outcome', all(q.get('expected_outcome') in valid_outcomes and isinstance(q.get('question'),str) for q in queries))
index = {p['person_id']: {f['fact_id'] for f in p['facts']} for p in people}
record('Answerable queries reference facts owned by queried person', all(q.get('person_id') in index and q.get('expected_fact_id') in index[q['person_id']] for q in queries if q.get('expected_outcome') == 'answerable'))
record('Unanswerable/unknown queries do not require any fact', all(q.get('expected_fact_id') is None for q in queries if q.get('expected_outcome') != 'answerable'))
record('30 unknown cases have distinct IDs outside known profiles', len(unknown) == 30 and len({u.get('image_person_id') for u in unknown}) == 30 and all(isinstance(u.get('image_person_id'),str) and u['image_person_id'].startswith('unknown-') and u['image_person_id'] not in index for u in unknown))
expected = Counter({'answerable':100,'unanswerable':20,'unknown_identity_abstain':40})
record('Evaluation labels match published 100/20/40 split', Counter(q.get('expected_outcome') for q in queries) == expected)

print('='*67)
print('WANAS + ARC — STEP 7 DATA INTEGRITY TEST')
print('='*67)
for i,(label,ok,detail) in enumerate(checks,1):
    print(f'{i:02d}. {"PASS" if ok else "FAIL"}  {label}' + (f': {detail}' if detail else ''))
passed=sum(ok for _,ok,_ in checks)
print(f'\nSTEP 7: {passed}/{len(checks)} PASSED')
if passed != len(checks):
    raise SystemExit(1)
