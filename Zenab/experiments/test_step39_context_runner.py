"""Synthetic unit tests. Do not assert actual Face/STT or held-out performance."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step39_context_aware_runner import parse_context, evaluate_context
from step32_run_paired_evaluation import prepare,summarize

PERSON='person-001'
NAME='Omar Hassan'
S='session-01'
QUESTION='Tell me about him'
CONTEXT={'referent_person_id':PERSON,'session_id':S,'fresh':True}

class FakeBridge:
    def __init__(self):self.calls=[]
    def process(self,face,question):
        self.calls.append((face,question))
        if face.get('status')!='recognized':
            return {'status':'face_not_recognized_abstain','person_id':None,'fact_id':None,
                    'spoken_text':'Please confirm identity','can_speak_personal_memory':False,
                    'abstention_reason':'face_not_recognized'}
        return {'status':'answerable','person_id':face['person_id'],'fact_id':'person-001-F01',
                'spoken_text':'Demo fact','can_speak_personal_memory':True}

def data(question=QUESTION, ctx=CONTEXT, face='recognized', pid=PERSON, session=S, stt=None):
    c=dict(case_id='C1',split='validation',image_id='image-existing',question_id='Q1',
           utterance_id='utterance-existing',session_id=session,expected_person_id=PERSON,
           expected_identity_status='known',reference_transcript=question,
           conversation_context=json.dumps(ctx) if not isinstance(ctx,str) else ctx,
           acceptable_fact_ids='["person-001-F01"]',pairing_rationale='synthetic test',
           face_decision=face,face_predicted_person_id=pid if face=='recognized' else '',
           face_score='0.95' if face=='recognized' else '',face_model_version='fake-face',
           actual_stt_transcript=stt or question,stt_model_version='fake-stt',
           upstream_data_origin='actual_face_and_stt',case_review_status='reviewed')
    q={'question_id':'Q1','source_split':'validation','expected_outcome':'answerable',
       'expected_person_id':PERSON,'reference_transcript':question,
       'acceptable_fact_ids':'["person-001-F01"]'}
    valid,issues,_=prepare([c],{'Q1':q})
    assert not issues
    return valid

def main():
    passed=0
    def check(label,fn):
        nonlocal passed
        fn();passed+=1;print('PASS',label)
    check('valid JSON context',lambda: assert_true(parse_context(json.dumps(CONTEXT))==CONTEXT))
    check('reject non-JSON text',lambda: assert_true(parse_context('previous Omar') is None))
    check('reject fresh string',lambda: assert_true(parse_context(json.dumps({**CONTEXT,'fresh':'true'})) is None))
    check('reject missing referent',lambda: assert_true(parse_context(json.dumps({'session_id':S,'fresh':True})) is None))
    bridge=FakeBridge()
    rows=evaluate_context(data(),bridge,lambda pid:NAME)
    check('four runs produced in synthetic paired fixture',lambda: assert_true([r['run'] for r in rows]==list('ABCD')))
    check('pronoun expanded for correct current face',lambda: assert_true(all(r['resolved_question']==f'Who is {NAME}?' for r in rows)))
    check('correct current person returned',lambda: assert_true(all(r['outcome_category']=='correct_supported_answer' for r in rows)))
    bridge=FakeBridge();rows=evaluate_context(data(face='unknown'),bridge,lambda pid:NAME)
    check('oracle A and C valid',lambda: assert_true(rows[0]['outcome_category']=='correct_supported_answer' and rows[2]['outcome_category']=='correct_supported_answer'))
    check('unknown face blocks B and D',lambda: assert_true(all(rows[i]['can_speak_personal_memory'] is False for i in (1,3))))
    check('unknown face never invokes retrieval for B D',lambda: assert_true(len(bridge.calls)==2))
    bridge=FakeBridge();rows=evaluate_context(data(ctx={**CONTEXT,'session_id':'old'}),bridge,lambda pid:NAME)
    check('mismatched context session blocks all four',lambda: assert_true(all(r['abstention_reason']=='context_session_mismatch' for r in rows)))
    check('no retrieval on invalid session',lambda: assert_true(not bridge.calls))
    bridge=FakeBridge();rows=evaluate_context(data(ctx={**CONTEXT,'fresh':False}),bridge,lambda pid:NAME)
    check('stale context blocks',lambda: assert_true(all(not r['can_speak_personal_memory'] for r in rows)))
    bridge=FakeBridge();rows=evaluate_context(data(ctx={**CONTEXT,'referent_person_id':'person-002'}),bridge,lambda pid:NAME)
    check('wrong referent blocks',lambda: assert_true(all(not r['can_speak_personal_memory'] for r in rows)))
    bridge=FakeBridge();rows=evaluate_context(data(ctx='not JSON'),bridge,lambda pid:NAME)
    check('malformed context blocks pronouns',lambda: assert_true(all(r['context_parsed'] is False and r['can_speak_personal_memory'] is False for r in rows)))
    bridge=FakeBridge();rows=evaluate_context(data(question='Who is Omar Hassan?',ctx='not JSON'),bridge,lambda pid:NAME)
    check('explicit question forwarded unchanged',lambda: assert_true(len(bridge.calls)==4 and all(q=='Who is Omar Hassan?' for _,q in bridge.calls)))
    check('paired summary still works',lambda: assert_true(summarize(rows)['paired_A_D']['paired_cases']==1))
    print(f'STEP 39: {passed}/{passed} PASSED (SYNTHETIC STUB ONLY, NO REAL UPSTREAM)')

def assert_true(condition):
    if not condition:raise AssertionError('Condition failed')
if __name__=='__main__':main()
