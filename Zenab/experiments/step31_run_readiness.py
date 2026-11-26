"""Step 31: READINESS ONLY for paired Runs A/B/C/D. No models or scoring.

From project root: python experiments/step31_run_readiness.py
Inputs: handoff/step29/CASE_MANIFEST_WORKING.csv and STEP29_QUESTION_POOL.csv.
Do not fill absent upstream fields with mock output. Only validation at this stage.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'handoff' / 'step29'
CASES = BASE / 'CASE_MANIFEST_WORKING.csv'
POOL = BASE / 'STEP29_QUESTION_POOL.csv'
OUT = BASE / 'STEP31_RUN_READINESS.json'
FACE_STATUS = {'recognized','unknown','no_face','no_valid_face','low_confidence'}

def load(path):
    if not path.is_file():
        raise ValueError(f'File not found: {path}')
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        return set(r.fieldnames or []), list(r)

def s(row,key): return (row.get(key) or '').strip()

def check(rows,questions):
    diagnostics=[]; run_cases={k:[] for k in 'ABCD'}; seen=set()
    for n,row in enumerate(rows,2):
        cid=s(row,'case_id') or f'ROW_{n}'
        reasons=[]
        if cid in seen or cid.startswith('ROW_'): reasons.append('case_id_missing_or_duplicate')
        seen.add(cid)
        q=questions.get(s(row,'question_id'))
        if not q: reasons.append('question_not_in_pool')
        if s(row,'split')!='validation' or (q and s(q,'source_split')!='validation'):
            reasons.append('not_validation_pair')
        if not s(row,'image_id'): reasons.append('missing_real_image_id')
        if not s(row,'conversation_context'): reasons.append('missing_conversation_context')
        if not s(row,'pairing_rationale'): reasons.append('missing_pairing_rationale')
        if q:
            for key,pool_key in [('expected_person_id','expected_person_id'),('reference_transcript','reference_transcript'),('acceptable_fact_ids','acceptable_fact_ids')]:
                if s(row,key)!=s(q,pool_key): reasons.append(key+'_mismatch_pool')
        status=s(row,'expected_identity_status').lower()
        if status not in ('known','unknown'): reasons.append('invalid_or_missing_ground_truth_status')
        if q:
            expected_status='unknown' if s(q,'expected_outcome')=='unknown_identity_abstain' else 'known'
            if status!=expected_status: reasons.append('expected_identity_status_mismatch')
        # A requires ground truth and reference text. B/C/D additionally require real upstream outputs.
        base_ok=not reasons
        face=s(row,'face_decision').lower()
        face_origin=s(row,'upstream_data_origin').lower()
        face_ok=bool(face in FACE_STATUS and 'actual' in face_origin and s(row,'face_model_version'))
        if face=='recognized':
            face_ok=face_ok and bool(s(row,'face_predicted_person_id'))
            try:
                x=float(s(row,'face_score'))
                face_ok=face_ok and (0<=x<=1) and x==x
            except ValueError: face_ok=False
        # A real output MUST have a unique utterance ID, a model version and actual provenance.
        stt_ok=bool(s(row,'actual_stt_transcript') and s(row,'utterance_id') and s(row,'stt_model_version') and 'actual' in face_origin)
        possible={'A':base_ok, 'B':base_ok and face_ok, 'C':base_ok and stt_ok, 'D':base_ok and face_ok and stt_ok}
        for k,v in possible.items():
            if v: run_cases[k].append(cid)
        diagnostics.append({'case_id':cid,'question_id':s(row,'question_id'), 'ready':possible,
                            'blocking_base_reasons':reasons, 'face_actual_ready':face_ok, 'stt_actual_ready':stt_ok})
    return run_cases,diagnostics

def main():
    try:
        cols,rows=load(CASES); poolcols,poolrows=load(POOL)
    except (ValueError,OSError) as e:
        print('ERROR:',e); return 2
    required={'case_id','split','image_id','question_id','utterance_id','session_id',
              'expected_person_id','expected_identity_status','reference_transcript',
              'conversation_context','acceptable_fact_ids','pairing_rationale',
              'face_decision','face_predicted_person_id','face_score','face_model_version',
              'actual_stt_transcript','stt_model_version','upstream_data_origin'}
    missing=sorted(required-cols)
    if missing:
        print('ERROR: Case manifest missing columns:', ', '.join(missing)); return 2
    questions={s(x,'question_id'):x for x in poolrows}
    runs,details=check(rows,questions)
    counts={k:len(v) for k,v in runs.items()}
    # Same-case comparisons are critical; D cases are always A/B/C eligible.
    report={'step':31,'assessment':'PREFLIGHT_ONLY_NO_RETRIEVAL_OR_MODEL_EXECUTION',
            'source':'existing previously inspected synthetic benchmark',
            'case_rows':len(rows),'ready_by_run':counts,
            'paired_A_D_count':len(set(runs['A']) & set(runs['D'])),
            'ready_case_ids_by_run':runs,'details':details,
            'requirements':['real image/question pairing reviewed','Hanaa/Ahmed identity mapping',
                            'face decision from Ahmed actual pipeline','paired Toqa actual STT output',
                            'frozen retrieval configuration','same cases and context across A/B/C/D'],
            'heldout_status':'NOT_EVALUATED',
            'note':'A/B/C/D are NOT executed by this script. Do not claim reliability results from this file.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP 31 — A/B/C/D READINESS (NO MODEL RUNS / NO SCORES)')
    print('Linked validation cases:',len(rows))
    for run in 'ABCD': print(f'Run {run} ready cases: {counts[run]}')
    print('Comparable paired A vs D:',report['paired_A_D_count'])
    print('Report:',OUT.relative_to(ROOT))
    if not rows:
        print('STATUS: WAITING_FOR_REAL_CASE_PAIRINGS — correct for an empty manifest')
    else:
        bad=[d for d in details if d['blocking_base_reasons']]
        print(f'Cases with invalid/missing baseline fields: {len(bad)}')
        print('WARNING: Readiness is not evaluation; run Step 30 to inspect manifest errors.')
    return 0
if __name__=='__main__': sys.exit(main())
