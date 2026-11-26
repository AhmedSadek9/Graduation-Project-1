"""Synthetic contract tests for Step40. No real Face/STT outputs, no E5."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step37_fact_ownership_audit import load_fact_index
from step40_context_verified_audit import audit_rows

idx=load_fact_index()
case={'case_id':'fixture-1','split':'validation','image_id':'synthetic-image',
      'question_id':'q-fixture','expected_person_id':'person-001',
      'expected_identity_status':'known','acceptable_fact_ids':'["person-001-F01"]'}
def p(fid='person-001-F01',pid='person-001',allow=True,status='answerable',text='sample answer'):
    return {'case_id':'fixture-1','run':'A','split':'validation','image_id':'synthetic-image',
      'question_id':'q-fixture','expected_person_id':'person-001',
      'retrieved_fact_id':fid,'retrieved_person_id':pid,'can_speak_personal_memory':allow,
      'response_status':status,'spoken_text':text,'outcome_category':'correct_supported_answer'}
count=0
def check(name, predicate):
    global count
    assert predicate,name
    count+=1
    print('PASS',name)
def single(row, c=case):return audit_rows([c],[row],idx)[0]
r=single(p())
check('correct fact ID has provisional verified classification',r['step40_verified_outcome']=='correct_supported_answer')
check('spoken response stays pending human review',r['step40_spoken_text_human_review_required'] and r['step40_spoken_text_review_status']=='pending')
check('not eligible for final outcome claim',r['step40_eligible_for_final_correct_outcome_claim'] is False)
r=single(p(fid='person-002-F01',pid='person-001'))
check('wrong owner caught independently',r['step40_verified_outcome']=='wrong_person_answer')
check('wrong owner plus speaking permission flagged',bool(r['step40_safety_flags']))
r=single(p(fid='person-001-F99'))
check('missing fact marked unsupported',r['step40_verified_outcome']=='unsupported_answer')
r=single(p(fid=None,pid=None,allow=False,status='unanswerable',text='No verified information'))
check('answerable abstention is unnecessary',r['step40_verified_outcome']=='unnecessary_abstention')
check('generic refusal not marked personal text',r['step40_spoken_text_human_review_required'] is False)
unknown={**case,'expected_person_id':'','expected_identity_status':'unknown','acceptable_fact_ids':'[]'}
r=single({**p(),'expected_person_id':''},unknown)
check('unknown with fact is wrong person',r['step40_verified_outcome']=='wrong_person_answer')
r=single({**p(fid=None,pid=None,allow=False,status='unanswerable',text='No verified information'),'expected_person_id':''},unknown)
check('unknown correctly abstains',r['step40_verified_outcome']=='correct_abstention')
try:
    audit_rows([case],[p(),p()],idx)
except ValueError:
    check('duplicate case-run rejected',True)
else:raise AssertionError('Duplicate row accepted')
try:
    audit_rows([case],[{**p(),'question_id':'other'}],idx)
except ValueError:
    check('mismatched question rejected',True)
else:raise AssertionError('Mismatch accepted')
print(f'STEP 40: {count}/{count} PASSED (fictional contract tests only)')
