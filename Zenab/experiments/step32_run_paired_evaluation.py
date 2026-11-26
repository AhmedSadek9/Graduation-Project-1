"""Step 32: experimental paired A/B/C/D E5 memory evaluation (synthetic data only).

Place inside Zenab/experiments. Execute from Zenab project root:
    python experiments/step32_run_paired_evaluation.py
    python experiments/step32_run_paired_evaluation.py --execute --face-score-contract-confirmed

No results produced for missing/incomplete real case pairings. The second command
must be used ONLY after Ahmed confirms face_score semantics and ID mapping,
and Hanaa verifies case pairings. STT and face inputs must be actual module outputs.
This is NOT a production integration and is NOT a fresh independent held-out test.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'handoff'/'step29'
REQUIRED=('case_id','split','image_id','question_id','expected_person_id','expected_identity_status','reference_transcript','conversation_context','acceptable_fact_ids','pairing_rationale','face_decision','face_predicted_person_id','face_score','face_model_version','actual_stt_transcript','stt_model_version','utterance_id','upstream_data_origin','case_review_status')
ALLOWED_FACE={'recognized','unknown','no_face','no_valid_face','low_confidence'}

def rows_from(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f); return set(r.fieldnames or []),list(r)

def s(row,key):return (row.get(key) or '').strip()

def fact_ids(raw):
    try:
        value=json.loads(raw)
        if not isinstance(value,list) or any(not isinstance(v,str) for v in value):raise ValueError()
        return value
    except (ValueError,TypeError):raise ValueError('acceptable_fact_ids must be a JSON array of strings')

def validate_case(c,q):
    errors=[]
    for key in ('case_id','image_id','question_id','reference_transcript','conversation_context','pairing_rationale'):
        if not s(c,key):errors.append('missing_'+key)
    if s(c,'split')!='validation' or s(q,'source_split')!='validation':errors.append('not_validation')
    for key in ('expected_person_id','reference_transcript','acceptable_fact_ids'):
        if s(c,key)!=s(q,key):errors.append('mismatch_'+key)
    outcome=s(q,'expected_outcome')
    expected_status='unknown' if outcome=='unknown_identity_abstain' else 'known'
    if s(c,'expected_identity_status')!=expected_status:errors.append('ground_truth_status_mismatch')
    if expected_status=='known' and not s(c,'expected_person_id'):errors.append('missing_expected_person')
    try: fact_ids(s(c,'acceptable_fact_ids'))
    except ValueError:errors.append('invalid_acceptable_ids')
    return errors

def actual_face_ready(c):
    origin=s(c,'upstream_data_origin').lower()
    if 'actual' not in origin:return False
    face=s(c,'face_decision').lower()
    if face not in ALLOWED_FACE or not s(c,'face_model_version'):return False
    if face=='recognized':
        if not s(c,'face_predicted_person_id'):return False
        try:
            score=float(s(c,'face_score'))
            if not math.isfinite(score) or not 0<=score<=1:return False
        except ValueError:return False
    return True

def actual_stt_ready(c):
    return bool(s(c,'utterance_id') and s(c,'stt_model_version') and s(c,'actual_stt_transcript') and 'actual' in s(c,'upstream_data_origin').lower())

def oracle_face(c):
    if s(c,'expected_identity_status')=='unknown':return {'status':'unknown','person_id':None}
    # An oracle gate, NOT Ahmed's model and NOT a real calibrated probability.
    return {'status':'recognized','person_id':s(c,'expected_person_id'),'recognition_confidence':1.0}

def actual_face(c):
    face=s(c,'face_decision').lower()
    result={'status':face,'person_id':s(c,'face_predicted_person_id') or None}
    if face=='recognized':result['recognition_confidence']=float(s(c,'face_score'))
    return result

def categorize(c,response):
    expected=s(c,'expected_identity_status'); valid=set(fact_ids(s(c,'acceptable_fact_ids')))
    returned=s(response,'fact_id'); pid=s(response,'person_id')
    disclosed=bool(response.get('can_speak_personal_memory') and returned)
    if disclosed:
        if expected=='known' and pid!=s(c,'expected_person_id'):return 'wrong_person_answer'
        if expected=='unknown':return 'wrong_person_answer'
        return 'correct_supported_answer' if returned in valid else 'unsupported_answer'
    if expected=='unknown' or not valid:return 'correct_abstention'
    return 'unnecessary_abstention'

def prepare(cases,pool):
    seen=set(); valid=[]; issues=[]; counts=Counter()
    for c in cases:
        cid=s(c,'case_id');q=pool.get(s(c,'question_id'))
        errors=validate_case(c,q) if q else ['question_not_found']
        if cid in seen:errors.append('duplicate_case_id')
        seen.add(cid)
        if errors:
            issues.append({'case_id':cid,'errors':errors});continue
        ready={'A':True,'B':actual_face_ready(c),'C':actual_stt_ready(c)}
        ready['D']=ready['B'] and ready['C']
        for run,ok in ready.items():
            if ok:counts[run]+=1
        valid.append((c,ready))
    return valid,issues,counts

def evaluate(valid,bridge):
    out=[]
    for c,ready in valid:
        for run in 'ABCD':
            if not ready[run]:continue
            face=oracle_face(c) if run in 'AC' else actual_face(c)
            text=s(c,'reference_transcript') if run in 'AB' else s(c,'actual_stt_transcript')
            response=bridge.process(face,text)
            category=categorize(c,response)
            out.append({'case_id':s(c,'case_id'),'run':run,'split':s(c,'split'),
                'image_id':s(c,'image_id'),'question_id':s(c,'question_id'),
                'utterance_id':s(c,'utterance_id'),'session_id':s(c,'session_id'),
                'expected_person_id':s(c,'expected_person_id'),
                'face_decision_used':face['status'],'face_person_id_used':face.get('person_id'),
                'face_score_used':face.get('recognition_confidence'),
                'reference_transcript':s(c,'reference_transcript'),
                'actual_stt_transcript':s(c,'actual_stt_transcript'),
                'question_used':text,'conversation_context_recorded_not_consumed':s(c,'conversation_context'),
                'acceptable_fact_ids':fact_ids(s(c,'acceptable_fact_ids')),
                'retrieved_fact_id':response.get('fact_id'),
                'retrieved_person_id':response.get('person_id'),
                'response_status':response.get('status'),'spoken_text':response.get('spoken_text'),
                'abstention_reason':response.get('abstention_reason'),
                'can_speak_personal_memory':response.get('can_speak_personal_memory'),
                'outcome_category':category})
    return out

def summarize(rows):
    summaries={}
    for run in 'ABCD':
        part=[x for x in rows if x['run']==run]
        summary=dict(Counter(x['outcome_category'] for x in part))
        correct=summary.get('correct_supported_answer',0)+summary.get('correct_abstention',0)
        summaries[run]={'cases':len(part),'correct_outcomes':correct,'correct_outcome_rate':round(correct/len(part),6) if part else None,'categories':summary}
    a={x['case_id']:x for x in rows if x['run']=='A'}
    d={x['case_id']:x for x in rows if x['run']=='D'}
    common=sorted(set(a)&set(d)); n=len(common)
    ca=sum(a[k]['outcome_category'] in ('correct_supported_answer','correct_abstention') for k in common)
    cd=sum(d[k]['outcome_category'] in ('correct_supported_answer','correct_abstention') for k in common)
    paired={'paired_cases':n,'run_A_correct':ca,'run_D_correct':cd,'reliability_loss_percentage_points':round(100*(ca-cd)/n,4) if n else None,
        'note':'No reliability loss reported if paired A/D count is zero.'}
    return {'run_summaries':summaries,'paired_A_D':paired}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Run E5 on eligible validation cases only')
    parser.add_argument('--face-score-contract-confirmed',action='store_true',help='Only after Ahmed confirms score is suitable for 0..1 gate and Hanaa confirms mapping')
    args=parser.parse_args()
    path=BASE/'CASE_MANIFEST_WORKING.csv';qpath=BASE/'STEP29_QUESTION_POOL.csv'
    if not path.is_file() or not qpath.is_file():
        print('ERROR: First run Steps 29/30/31 to create manifest and question pool.');return 2
    cols,cases=rows_from(path);_,poolrows=rows_from(qpath)
    missing=set(REQUIRED)-cols
    if missing:
        print('ERROR missing columns:',', '.join(sorted(missing)));return 2
    pool={s(q,'question_id'):q for q in poolrows}
    valid,issues,counts=prepare(cases,pool)
    print('STEP 32 — PAIRED EVALUATION PREFLIGHT')
    print('Cases:',len(cases),'valid:',len(valid),'issues:',len(issues),'ready:',dict(counts))
    if issues:
        print('ERROR: Correct invalid case rows before execution; example:',issues[0]);return 2
    if not args.execute:
        print('PREFLIGHT ONLY. For actual evaluation, review contracts then pass --execute --face-score-contract-confirmed.')
        return 0
    if not args.face_score_contract_confirmed:
        print('BLOCKED: Score semantics / person ID mapping must be explicitly confirmed before upstream testing.');return 2
    if not valid:
        print('NO CASES: No real image/question pairings; no model loaded, no scores invented.');return 0
    if not any(counts.values()):
        print('NO READY CASES: Nothing executed.');return 0
    # Demo only: no patient data, real upstream outputs may be used as *inputs*.
    from e5_demo_bridge_step21 import E5DemoBridge, THRESHOLD
    bridge=E5DemoBridge(data_mode='demo',threshold=THRESHOLD)
    predictions=evaluate(valid,bridge)
    output=ROOT/'experiments'/'results';output.mkdir(parents=True,exist_ok=True)
    predpath=output/'step32_paired_validation_predictions.jsonl'
    with predpath.open('w',encoding='utf-8') as f:
        for r in predictions:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=summarize(predictions)
    report.update({'step':32,'status':'EXPLORATORY_VALIDATION_NOT_HELDOUT',
      'model':'intfloat/multilingual-e5-small','threshold':THRESHOLD,
      'data':'fictional_demo_facts','actual_face_inputs_available':counts['B'],
      'actual_stt_inputs_available':counts['C'],
      'configuration_note':'Frozen Step16 E5 question-only adapter; last-request context is recorded, not yet consumed by model.',
      'critical_limitation':'Face score convention must be signed off; oracle score 1.0 is an explicit ground-truth gate, not real face confidence; do not report patient performance.'})
    report_path=output/'step32_paired_validation_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Executed prediction rows:',len(predictions),'| Saved:',predpath.relative_to(ROOT))
    print('Summary:',report_path.relative_to(ROOT),'| Paired:',report['paired_A_D'])
    return 0
if __name__=='__main__':sys.exit(main())
