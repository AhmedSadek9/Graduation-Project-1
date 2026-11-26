"""WANAS Step 9 bridge contract regression tests; all data are synthetic."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BRIDGE_FILE = ROOT / 'memory_integration.py'
if not BRIDGE_FILE.is_file():
    raise SystemExit(
        f'Missing integration module: {BRIDGE_FILE}\n'
        'Put memory_integration.py directly inside Wanas31 (not inside Task4).\n'
        'If you downloaded memory_integration_step9.py, rename it to memory_integration.py.'
    )
from memory_integration import FaceMemoryBridge

checks=[]
def check(label, condition):
    checks.append(bool(condition))
    print(f'{len(checks):02}. {"PASS" if condition else "FAIL"}  {label}')

demo=FaceMemoryBridge(data_mode='demo')
prod=FaceMemoryBridge(data_mode='production')
face={'status':'recognized','person_id':'person-001','recognition_confidence':.95}
check('Demo recognition greeting', demo.process(face)['status']=='success')
check('Demo matching question', demo.process(face,'retrieve_memory','Who is Omar Hassan?')['status']=='answerable')
check('Demo conversation starter', demo.process(face,'conversation_starter')['status']=='success')
check('Production refuses synthetic facts', prod.process(face)['status']=='no_verified_facts')
check('Production cannot speak synthetic facts', not prod.process(face)['can_speak_personal_memory'])
check('Face unknown status rejected', demo.process({**face,'status':'unknown'})['status']=='face_not_recognized_abstain')
check('Face low-confidence status rejected', demo.process({**face,'status':'low_confidence'})['status']=='face_not_recognized_abstain')
check('Missing status rejected', demo.process({'person_id':'person-001','recognition_confidence':.95})['status']=='face_not_recognized_abstain')
check('Missing score rejected', demo.process({'status':'recognized','person_id':'person-001'})['status']=='invalid_score_abstain')
check('NaN score rejected', demo.process({**face,'recognition_confidence':float('nan')})['status']=='invalid_score_abstain')
check('Boolean score rejected', demo.process({**face,'recognition_confidence':True})['status']=='invalid_score_abstain')
check('Below threshold cannot speak', not demo.process({**face,'recognition_confidence':.5})['can_speak_personal_memory'])
check('Unknown person ID rejected', demo.process({**face,'person_id':'unknown-001'})['status']=='invalid_person_id_abstain')
check('Unlisted person cannot speak', not demo.process({**face,'person_id':'person-999'})['can_speak_personal_memory'])
check('Mismatched name cannot speak', not demo.process(face,'retrieve_memory','Who is Salma Nabil?')['can_speak_personal_memory'])
check('Unanswerable question cannot speak facts', not demo.process(face,'retrieve_memory','What is Omar Hassan favorite color?')['can_speak_personal_memory'])
check('Blank question rejected', demo.process(face,'retrieve_memory',' ')['status']=='invalid_question_abstain')
check('Invalid intent rejected', demo.process(face,'delete_memory')['status']=='invalid_intent_abstain')
check('Malformed face payload rejected', demo.process(None)['status']=='invalid_face_result_abstain')
check('Default is production', FaceMemoryBridge().process(face)['status']=='no_verified_facts')
print(f'\nSTEP 9: {sum(checks)}/{len(checks)} PASSED')
if not all(checks): sys.exit(1)
