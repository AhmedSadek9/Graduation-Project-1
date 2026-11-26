"""STEP 33: Build a NON-EXECUTABLE validation pairing plan from the actual question pool.

Does not pair images, mark cases approved, populate CASE_MANIFEST_WORKING.csv,
run models, or reinterpret old test questions as independent evaluation.
Run from project root: python experiments/step33_prepare_validation_pairing.py
"""
import csv, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'handoff'/'step29'
SRC=BASE/'STEP29_QUESTION_POOL.csv'
DEST=BASE/'STEP33_VALIDATION_PAIRING_PLAN.csv'
REPORT=BASE/'STEP33_VALIDATION_PAIRING_SUMMARY.json'
if not SRC.exists():
    raise SystemExit('ERROR: Missing Step29 question pool. Run step29_prepare_case_pool.py first.')
with SRC.open('r',encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    questions=list(reader)
    required={'question_id','source_split','expected_outcome','expected_person_id','reference_transcript','acceptable_fact_ids'}
    if not required.issubset(set(reader.fieldnames or [])):
        raise SystemExit('ERROR: Step29 question pool schema mismatch.')
validation=[q for q in questions if q['source_split'].strip()=='validation']
if len(set(q['question_id'] for q in validation))!=len(validation):
    raise SystemExit('ERROR: duplicate validation question IDs')

fields=['question_id','source_split','expected_outcome','expected_person_id','expected_identity_status',
        'reference_transcript','acceptable_fact_ids','suggested_face_group','pairing_instruction',
        'image_id_TO_BE_FILLED_FROM_ACTUAL_CAPTURE','utterance_id_TO_BE_FILLED_FROM_ACTUAL_STT',
        'conversation_context_TO_BE_AUTHORED_AND_REVIEWED','pairing_rationale_TO_BE_REVIEWED',
        'pairing_status']
with DEST.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    for q in validation:
        outcome=q['expected_outcome'].strip()
        unknown=outcome=='unknown_identity_abstain'
        person=q['expected_person_id'].strip()
        if unknown and person: raise SystemExit(f'ERROR unknown question has expected person: {q["question_id"]}')
        if not unknown and not person: raise SystemExit(f'ERROR known question missing person: {q["question_id"]}')
        writer.writerow({
            'question_id':q['question_id'], 'source_split':'validation','expected_outcome':outcome,
            'expected_person_id':person,'expected_identity_status':'unknown' if unknown else 'known',
            'reference_transcript':q['reference_transcript'],
            'acceptable_fact_ids':q['acceptable_fact_ids'],
            'suggested_face_group':'UNKNOWN_REAL_FACE' if unknown else 'KNOWN_REAL_FACE_SAME_MAPPED_PERSON',
            'pairing_instruction':('Choose an actual unknown visitor image with documented ground truth; never substitute old identity' if unknown
                else 'Choose a real image whose independently verified identity maps to this expected_person_id; no arbitrary person substitution'),
            'image_id_TO_BE_FILLED_FROM_ACTUAL_CAPTURE':'',
            'utterance_id_TO_BE_FILLED_FROM_ACTUAL_STT':'',
            'conversation_context_TO_BE_AUTHORED_AND_REVIEWED':'',
            'pairing_rationale_TO_BE_REVIEWED':'',
            'pairing_status':'PROPOSAL_ONLY_NO_ACTUAL_PAIRING'})
summary={'step':33,'status':'PAIRING_PLAN_ONLY_NO_REAL_PAIRINGS','validation_questions':len(validation),
         'outcomes':dict(Counter(x['expected_outcome'] for x in validation)),
         'known_person_ids':sorted({x['expected_person_id'] for x in validation if x['expected_person_id']}),
         'unknown_questions':sum(x['expected_outcome']=='unknown_identity_abstain' for x in validation),
         'actual_images_paired':0,'actual_stt_pairs':0,'evaluation_runs_executed':0,
         'warnings':['Source 26-question validation covers only listed known person IDs; do not force unrelated 50-image cohort into these cases.',
                     'Use actual image IDs, documented independently verified identity, stable person mapping, and case review before running Step32.',
                     'Do not treat prior inspected test questions as new independent held-out evaluation.',
                     'The manifest must include reviewed nonempty context for Step32; Step32 currently records but does not consume conversation context.',
                     'Do not fabricate unknown/no_face/no_valid_face outputs; preserve Ahmed actual decision.']}
REPORT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('STEP 33 — VALIDATION PAIRING PLAN ONLY')
print('Validation questions:',len(validation),'| outcomes:',summary['outcomes'])
print('Known person IDs:',', '.join(summary['known_person_ids']))
print('Actual image IDs: 0 | Actual STT pairs: 0 | A/B/C/D runs: 0')
print('Saved:',DEST.relative_to(ROOT))
print('Saved:',REPORT.relative_to(ROOT))
print('Next: coordinate actual image-person mapping with Ahmed + Hanaa; leave existing CASE_MANIFEST_WORKING.csv untouched.')
