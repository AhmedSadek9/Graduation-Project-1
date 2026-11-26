"""Step 29: export the existing question inventory and an EMPTY case-manifest template.

Run from WANAS project root: python experiments/step29_prepare_case_pool.py
No model loading, no simulated face/STT outputs, no held-out scoring.
"""
from pathlib import Path
from collections import Counter
import json, csv, sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'benchmark' / 'question_manifest.json'
OUT = ROOT / 'handoff' / 'step29'
OUT.mkdir(parents=True, exist_ok=True)
if not SRC.is_file():
    raise SystemExit(f'Missing benchmark file: {SRC}')
questions = json.loads(SRC.read_text(encoding='utf-8-sig'))
if not isinstance(questions, list) or not questions:
    raise SystemExit('Question manifest must contain a nonempty list.')
ids = [q.get('query_id') for q in questions]
if len(ids) != len(set(ids)):
    raise SystemExit('Duplicate query_id in source benchmark; inspect before pairing.')

pool_fields = ['question_id','original_question_id','source_split','expected_outcome',
               'expected_person_id','reference_transcript','acceptable_fact_ids',
               'last_request','context_scenario','origin_id','pairing_status']
pool_path = OUT / 'STEP29_QUESTION_POOL.csv'
with pool_path.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=pool_fields)
    w.writeheader()
    for q in questions:
        acceptable = q.get('acceptable_fact_ids', '[]')
        if isinstance(acceptable, list): acceptable = json.dumps(acceptable, ensure_ascii=False)
        try: ids_here = json.loads(acceptable)
        except (ValueError, TypeError): raise SystemExit(f"Invalid acceptable_fact_ids in {q.get('query_id')}")
        if not isinstance(ids_here,list): raise SystemExit(f"Invalid fact list in {q.get('query_id')}")
        w.writerow({
            'question_id':q['query_id'], 'original_question_id':q.get('original_question_id',''),
            'source_split':q.get('split',''),'expected_outcome':q.get('expected_outcome',''),
            'expected_person_id':q.get('expected_person_id',''),
            'reference_transcript':q.get('question',''), 'acceptable_fact_ids':acceptable,
            'last_request':q.get('last_request',''), 'context_scenario':q.get('context_scenario',''),
            'origin_id':q.get('origin_id',''), 'pairing_status':'NOT_PAIRED_WITH_REAL_IMAGE_OR_UTTERANCE'
        })

fields = ['case_id','split','image_id','question_id','utterance_id','session_id',
          'expected_person_id','expected_identity_status','reference_transcript',
          'conversation_context','acceptable_fact_ids','pairing_rationale',
          'face_decision','face_predicted_person_id','face_score','face_model_version',
          'actual_stt_transcript','stt_model_version','upstream_data_origin','case_review_status']
case_path = OUT / 'CASE_MANIFEST_WORKING.csv'
if not case_path.exists():
    with case_path.open('w',encoding='utf-8-sig',newline='') as f:
        csv.writer(f).writerow(fields)
    case_info='CREATED EMPTY — no actual image/utterance IDs available'
else:
    case_info='PRESERVED EXISTING — not overwritten'
summary = {'source_questions':len(questions),'source_splits':dict(Counter(q.get('split','') for q in questions)),
           'source_outcomes':dict(Counter(q.get('expected_outcome','') for q in questions)),
           'question_pool_file':str(pool_path.relative_to(ROOT)),
           'case_manifest_file':str(case_path.relative_to(ROOT)),
           'case_manifest_action':case_info,
           'evaluation_status':'NOT RUN; NO ACTUAL FACE/STT OUTPUTS',
           'note':'Source test questions are previously inspected, NOT a fresh independent held-out set.'}
(OUT/'STEP29_POOL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('STEP 29 — QUESTION POOL PREPARATION')
print('Questions:',len(questions),'| Splits:',summary['source_splits'])
print('Outcomes:',summary['source_outcomes'])
print('Saved:',pool_path.relative_to(ROOT))
print('Case manifest:',case_info)
print('NO images, face decisions or STT outputs invented. NO models loaded.')
