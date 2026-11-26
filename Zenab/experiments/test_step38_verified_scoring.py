"""Tests Step38 manifest-backed re-scoring on fictional fixtures. No model load."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step37_fact_ownership_audit import load_fact_index
from step38_verified_scoring import score_pair, verify_prediction_rows
idx=load_fact_index()
case={'case_id':'c1','split':'validation','image_id':'img_fixture','question_id':'q1',
      'expected_person_id':'person-001','expected_identity_status':'known',
      'acceptable_fact_ids':'["person-001-F01"]'}
def response(fid='person-001-F01',pid='person-001',label='correct_supported_answer',status='answerable',speak=True):
    return {'case_id':'c1','run':'A','split':'validation','image_id':'img_fixture','question_id':'q1',
      'expected_person_id':'person-001','retrieved_fact_id':fid,'retrieved_person_id':pid,
      'can_speak_personal_memory':speak,'response_status':status,'spoken_text':'fixture text',
      'outcome_category':label}
checks=[
 ('correct owner',response(),'correct_supported_answer',False),
 ('wrong owner with spoofed response person',response('person-002-F01'),'wrong_person_answer',True),
 ('fact not in acceptable set',response('person-001-F02'),'unsupported_answer',True),
 ('nonexistent fact',response('person-001-F99'),'unsupported_answer',True),
 ('false-flag answer with status answerable',response('person-002-F01',speak=False),'wrong_person_answer',True),
 ('answerable abstention',response(None,None,'unnecessary_abstention','unanswerable',False),'unnecessary_abstention',False),
]
for name,p,expect,changed in checks:
    r=score_pair(case,p,idx)
    assert (r['step38_manifest_verified_outcome'],r['step38_outcome_changed'])==(expect,changed),(name,r)
    print('PASS',name)
unknown=dict(case,expected_person_id='',expected_identity_status='unknown',acceptable_fact_ids='[]')
r=score_pair(unknown,response(),idx)
assert r['step38_manifest_verified_outcome']=='wrong_person_answer'
print('PASS unknown must not disclose')
try:
    verify_prediction_rows([case],[dict(response(),question_id='bad')],idx)
except ValueError:print('PASS reject mismatched case')
else:raise AssertionError('Mismatched case was accepted')
try:
    verify_prediction_rows([case],[response(),response()],idx)
except ValueError:print('PASS reject duplicate prediction')
else:raise AssertionError('Duplicate accepted')
print('STEP 38: 9/9 PASSED (synthetic logic-only; no actual predictions)')
