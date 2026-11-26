"""Step 38: independent fact-manifest audit of Step32 prediction files.

No model execution, no edits to Step32, no actual-data claims.
Use from Zenab root: python experiments/step38_verified_scoring.py
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from step32_run_paired_evaluation import fact_ids, summarize
from step37_fact_ownership_audit import load_fact_index, classify_verified

MANIFEST=ROOT/'handoff'/'step29'/'CASE_MANIFEST_WORKING.csv'
PREDICTIONS=ROOT/'experiments'/'results'/'step32_paired_validation_predictions.jsonl'
OUTPUT=ROOT/'experiments'/'results'


def score_pair(case, prediction, index):
    """Re-score one recorded Step32 output, without trusting response person/fact IDs."""
    expected_case={
        'expected_person_id':case.get('expected_person_id',''),
        'expected_identity_status':case.get('expected_identity_status',''),
        'acceptable_fact_ids':fact_ids(case.get('acceptable_fact_ids','[]')),
    }
    response={
        'fact_id':prediction.get('retrieved_fact_id'),
        'person_id':prediction.get('retrieved_person_id'),
        'can_speak_personal_memory':prediction.get('can_speak_personal_memory'),
        'status':prediction.get('response_status'),
        'spoken_text':prediction.get('spoken_text'),
    }
    category=classify_verified(expected_case,response,index)
    old=prediction.get('outcome_category')
    # Spoken text faithfulness cannot be proven by checking the ID alone.
    spoken_review_required=(category=='correct_supported_answer' and bool(str(response.get('spoken_text') or '').strip()))
    return {**prediction, 'step38_manifest_verified_outcome':category,
            'step38_outcome_changed':old!=category,
            'step38_spoken_faithfulness_review_required':spoken_review_required,
            'step38_fact_source':'benchmark/fact_manifest.json'}


def verify_prediction_rows(cases, predictions, index):
    by_id={}
    for c in cases:
        cid=(c.get('case_id') or '').strip()
        if not cid or cid in by_id:raise ValueError('Missing or duplicate case_id in manifest')
        by_id[cid]=c
    processed=[];seen=set()
    for p in predictions:
        key=(p.get('case_id'),p.get('run'))
        if key in seen:raise ValueError('Duplicate case/run prediction')
        seen.add(key)
        c=by_id.get(p.get('case_id'))
        if c is None:raise ValueError('Prediction not linked to case manifest')
        if p.get('run') not in ('A','B','C','D'):raise ValueError('Invalid run')
        if p.get('split')!='validation' or c.get('split')!='validation':raise ValueError('Only validation supported')
        if p.get('question_id')!=c.get('question_id') or p.get('image_id')!=c.get('image_id'):raise ValueError('Case identity mismatch')
        if p.get('expected_person_id')!=c.get('expected_person_id'):raise ValueError('Ground truth mismatch')
        processed.append(score_pair(c,p,index))
    return processed


def main():
    index=load_fact_index()
    if not MANIFEST.exists():
        print('MISSING case manifest; run Step29 first.');return 2
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f:cases=list(csv.DictReader(f))
    if not PREDICTIONS.exists():
        print('STEP 38 — PREPARATION ONLY')
        print('Fact index:',len(index),'| Linked cases:',len(cases),'| Step32 prediction file: NOT PRESENT')
        print('No scores computed. Step32 original unchanged.');return 0
    with PREDICTIONS.open(encoding='utf-8-sig') as f:preds=[json.loads(line) for line in f if line.strip()]
    if not preds:
        print('Prediction file empty: no metrics computed.');return 0
    verified=verify_prediction_rows(cases,preds,index)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    out=OUTPUT/'step38_verified_validation_predictions.jsonl'
    with out.open('w',encoding='utf-8') as f:
        for row in verified:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for row in verified:row['outcome_category']=row['step38_manifest_verified_outcome']
    metrics=summarize(verified)
    metrics.update({'status':'EXPLORATORY_VALIDATION_ONLY','step':38,'prediction_rows':len(verified),
      'changed_outcomes':sum(r['step38_outcome_changed'] for r in verified),
      'spoken_faithfulness_review_required_n':sum(r['step38_spoken_faithfulness_review_required'] for r in verified),
      'limitations':['Not independent held-out','Spoken text is not verified against factual content','No invented upstream results']})
    report=OUTPUT/'step38_verified_validation_report.json'
    report.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP 38 — verified rows:',len(verified),'| changed:',metrics['changed_outcomes'])
    print('Report:',report.relative_to(ROOT))
    print('Spoken-text review remains mandatory; no real integration claimed.')
    return 0

if __name__=='__main__':raise SystemExit(main())
