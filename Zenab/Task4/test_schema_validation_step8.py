"""STEP 8: actual JSON Schema validation and negative regression cases.
Run: python Task4/test_schema_validation_step8.py
Install: python -m pip install 'jsonschema>=4.18,<5'
"""
import copy
import json
from pathlib import Path

try:
    from jsonschema import Draft7Validator, FormatChecker
except ImportError as exc:
    raise SystemExit('Install first: python -m pip install "jsonschema>=4.18,<5"') from exc

root = Path(__file__).resolve().parent.parent
schema = json.loads((root/'Task1'/'memory_schema.json').read_text(encoding='utf8'))
db = json.loads((root/'Task1'/'memory_database.json').read_text(encoding='utf8'))
Draft7Validator.check_schema(schema)
validator=Draft7Validator(schema,format_checker=FormatChecker())
cases=[]
def check(label,document,should_pass):
    valid=validator.is_valid(document)
    cases.append((label,valid==should_pass))

check('Original synthetic dataset conforms to schema',db,True)
for label,modify in [
    ('Missing person source rejected', lambda d:d['persons'][0].pop('source_type')),
    ('Missing person status rejected', lambda d:d['persons'][0].pop('status')),
    ('Blank person name rejected', lambda d:d['persons'][0].__setitem__('name','')),
    ('Missing fact verification rejected', lambda d:d['persons'][0]['facts'][0].pop('verification_status')),
    ('Missing fact review rejected', lambda d:d['persons'][0]['facts'][0].pop('review_status')),
    ('Missing fact source rejected', lambda d:d['persons'][0]['facts'][0].pop('source_type')),
    ('Blank memory fact rejected', lambda d:d['persons'][0]['facts'][0].__setitem__('memory_fact','')),
    ('Invalid fact ID rejected', lambda d:d['persons'][0]['facts'][0].__setitem__('fact_id','BAD-ID')),
    ('Invalid date rejected', lambda d:d['persons'][0]['facts'][0].__setitem__('last_updated','2026-02-30')),
    ('Invalid tier rejected', lambda d:d['persons'][0]['facts'][0].__setitem__('display_tier',3)),
    ('Missing fact date rejected', lambda d:d['persons'][0]['facts'][0].pop('last_updated')),
]:
    modified=copy.deepcopy(db)
    modify(modified)
    check(label,modified,False)
print('='*67)
print('WANAS + ARC — STEP 8 JSON SCHEMA VALIDATION')
print('='*67)
for i,(label,ok) in enumerate(cases,1):
    print(f'{i:02d}. {"PASS" if ok else "FAIL"}  {label}')
n=sum(ok for _,ok in cases)
print(f'\nSTEP 8: {n}/{len(cases)} PASSED')
if n!=len(cases):raise SystemExit(1)
