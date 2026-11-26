"""WANAS Step 30: read-only preflight for manually paired A/B/C/D evaluation cases.

Run from project root: python experiments/step30_validate_case_manifest.py
Uses existing Step29 question pool; does NOT invent face/STT data, load models,
or execute the evaluation. Accepts validation pairings only at this stage.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'handoff' / 'step29'
POOL = BASE / 'STEP29_QUESTION_POOL.csv'
CASES = BASE / 'CASE_MANIFEST_WORKING.csv'
OUTPUT = BASE / 'STEP30_CASE_MANIFEST_CHECK.json'
REQ = ['case_id','split','image_id','question_id','utterance_id','session_id',
       'expected_person_id','expected_identity_status','reference_transcript',
       'conversation_context','acceptable_fact_ids','pairing_rationale',
       'face_decision','face_predicted_person_id','face_score','face_model_version',
       'actual_stt_transcript','stt_model_version','upstream_data_origin','case_review_status']


def read_csv(path):
    if not path.is_file():
        raise ValueError(f'Missing {path.relative_to(ROOT)}. Run Step 29 first.')
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        headers=reader.fieldnames or []
        rows=list(reader)
    return headers, rows


def parse_ids(val):
    ids=json.loads(val)
    if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids):
        raise ValueError('acceptable_fact_ids must be a JSON list of strings')
    return ids


def main():
    errors=[]; warnings=[]
    try:
        pool_cols,pool_rows=read_csv(POOL)
        cols,rows=read_csv(CASES)
    except (ValueError,OSError) as exc:
        print('ERROR:',exc); return 2
    for key in REQ:
        if key not in cols: errors.append(f'Missing column {key}')
    questions={}
    for q in pool_rows:
        qid=q.get('question_id','').strip()
        if not qid or qid in questions: errors.append(f'Missing/duplicate question_id in pool: {qid}')
        questions[qid]=q
    seen=set(); face_ready=0; stt_ready=0; linked=0
    for i,row in enumerate(rows,2):
        tag=f'CSV row {i}'
        cid=row.get('case_id','').strip(); qid=row.get('question_id','').strip()
        if not cid: errors.append(f'{tag}: missing case_id')
        elif cid in seen: errors.append(f'{tag}: duplicate case_id {cid}')
        seen.add(cid)
        q=questions.get(qid)
        if q is None:
            errors.append(f'{tag}: question_id {qid!r} not present in Step29 pool')
            continue
        # A/B: source transcript and reference identity must agree with trusted question pool.
        for field,expected in [('expected_person_id',q['expected_person_id']),
                               ('reference_transcript',q['reference_transcript'])]:
            if (row.get(field) or '').strip()!=expected.strip():
                errors.append(f'{tag}: {field} does not match question pool ({qid})')
        try:
            ids=parse_ids((row.get('acceptable_fact_ids') or '').strip())
            base_ids=parse_ids(q['acceptable_fact_ids'])
            if ids!=base_ids: errors.append(f'{tag}: acceptable_fact_ids mismatch with pool')
        except (ValueError,json.JSONDecodeError) as exc:
            errors.append(f'{tag}: invalid acceptable_fact_ids: {exc}')
        if (row.get('split') or '').strip()!='validation':
            errors.append(f'{tag}: step 30 accepts validation cases only, got {row.get("split")!r}')
        if q['source_split']!='validation':
            warnings.append(f'{tag}: linked question comes from {q["source_split"]}, not validation; document why')
        if not row.get('image_id','').strip(): errors.append(f'{tag}: missing real image_id')
        if not row.get('conversation_context','').strip(): errors.append(f'{tag}: missing documented context')
        if not row.get('pairing_rationale','').strip(): errors.append(f'{tag}: missing pairing rationale')
        outcome=q.get('expected_outcome','')
        status=(row.get('expected_identity_status') or '').strip().lower()
        if outcome=='unknown_identity_abstain':
            if status!='unknown': errors.append(f'{tag}: unknown visitor must have expected_identity_status=unknown')
        elif status!='known':
            errors.append(f'{tag}: known-person question requires expected_identity_status=known')
        decision=(row.get('face_decision') or '').strip().lower()
        face_pid=(row.get('face_predicted_person_id') or '').strip()
        if decision:
            if decision not in {'recognized','unknown','no_face','no_valid_face','low_confidence'}:
                warnings.append(f'{tag}: verify face_decision={decision!r} against Ahmed actual contract')
            if decision=='recognized' and not face_pid:
                errors.append(f'{tag}: recognized decision without predicted person ID')
            if decision in {'unknown','no_face','no_valid_face'} and face_pid:
                warnings.append(f'{tag}: non-recognized decision has predicted ID, ensure downstream ignores it')
            if not row.get('face_model_version','').strip(): warnings.append(f'{tag}: face model version missing')
            face_ready+=1
        if (row.get('actual_stt_transcript') or '').strip():
            if not row.get('utterance_id','').strip(): errors.append(f'{tag}: actual STT text without utterance_id')
            if not row.get('stt_model_version','').strip(): warnings.append(f'{tag}: STT model version missing')
            stt_ready+=1
        origin=(row.get('upstream_data_origin') or '').strip()
        if decision or (row.get('actual_stt_transcript') or '').strip():
            if not origin: errors.append(f'{tag}: upstream_data_origin required for actual module outputs')
            elif 'actual' not in origin.lower(): warnings.append(f'{tag}: verify source is actual output, not mock/simulated')
        linked+=1
    if not rows: warnings.append('No real image/question case pairings yet; evaluation remains pending.')
    verdict='INVALID' if errors else ('READY_FOR_VALIDATION_PREP' if rows else 'WAITING_FOR_REAL_PAIRINGS')
    report={'step':30,'status':verdict,'question_pool_n':len(pool_rows),'case_rows':len(rows),
            'linked_rows':linked,'face_decision_rows':face_ready,'actual_stt_rows':stt_ready,
            'errors':errors,'warnings':warnings,
            'note':'Preflight only. Not a face/STT integration run, and not independent held-out evaluation.'}
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP 30 — CASE MANIFEST PREFLIGHT (READ ONLY)')
    print(f'Question pool: {len(pool_rows)} | Linked cases: {len(rows)} | Actual face decisions: {face_ready} | Actual STT rows: {stt_ready}')
    print(f'Errors: {len(errors)} | Warnings: {len(warnings)} | Status: {verdict}')
    for item in errors[:12]:print('ERROR:',item)
    for item in warnings[:8]:print('NOTE:',item)
    print('Saved:',OUTPUT.relative_to(ROOT))
    return 1 if errors else 0

if __name__=='__main__': sys.exit(main())
