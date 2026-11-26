"""Logic-only checks; no E5 download or actual upstream outputs needed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step32_run_paired_evaluation import prepare,evaluate,summarize

q={'question_id':'Q1','source_split':'validation','expected_outcome':'answerable','expected_person_id':'person-001','reference_transcript':'Who is Omar?','acceptable_fact_ids':'["person-001-F01"]'}
def sample():
    return dict(case_id='C1',split='validation',image_id='real_image_1',question_id='Q1',utterance_id='utt1',session_id='S1',expected_person_id='person-001',expected_identity_status='known',reference_transcript='Who is Omar?',conversation_context='prior_turn_1',acceptable_fact_ids='["person-001-F01"]',pairing_rationale='same person',face_decision='unknown',face_predicted_person_id='',face_score='',face_model_version='face-v1',actual_stt_transcript='Who is Omar?',stt_model_version='stt-v1',upstream_data_origin='actual_face_and_stt',case_review_status='reviewed')
class FakeBridge:
    def process(self,face,question):
        if face['status']!='recognized':return dict(status='face_not_recognized_abstain',person_id=None,fact_id=None,can_speak_personal_memory=False,abstention_reason='missing_or_unrecognized_face')
        return dict(status='answerable',person_id=face['person_id'],fact_id='person-001-F01',can_speak_personal_memory=True,abstention_reason=None)

valid,issues,counts=prepare([sample()],{'Q1':q})
assert not issues and dict(counts)==dict(A=1,B=1,C=1,D=1)
rows=evaluate(valid,FakeBridge())
assert [x['outcome_category'] for x in rows]==['correct_supported_answer','unnecessary_abstention','correct_supported_answer','unnecessary_abstention']
summary=summarize(rows)
assert summary['paired_A_D']=={'paired_cases':1,'run_A_correct':1,'run_D_correct':0,'reliability_loss_percentage_points':100.0,'note':'No reliability loss reported if paired A/D count is zero.'}
valid2,issues2,_=prepare([dict(sample(),expected_person_id='person-002')],{'Q1':q})
assert issues2 and not valid2
valid3,issues3,counts3=prepare([],{'Q1':q})
assert not issues3 and not valid3 and not counts3
print('STEP 32 LOGIC TESTS: 3/3 PASSED (fictional fixture / fake bridge only)')
