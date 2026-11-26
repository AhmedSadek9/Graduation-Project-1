"""WANAS Step 11: offline threshold diagnostics from saved semantic predictions.
No model inference or downloads. Tune on validation only; test is diagnostic only.
Run from repository root: python experiments/threshold_diagnostic_step11.py
"""
from pathlib import Path
import json, collections

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments'/'results'
MODELS=('all_MiniLM_L6_v2','multilingual_e5_small')

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def evaluate(records, threshold):
    counts=collections.Counter()
    examples=[]
    for p in records:
        is_answerable=(p['expected_outcome']=='answerable')
        top=(p.get('ranked_fact_ids') or [None])[0]
        scores=p.get('scores') or []
        score=scores[0] if scores else float('-inf')
        accepted=(top is not None and score>=threshold)
        correct=accepted and is_answerable and top in p['acceptable_fact_ids']
        wrong=accepted and is_answerable and not correct
        unsupported=accepted and not is_answerable
        counts['total']+=1
        counts['answerable' if is_answerable else 'unanswerable']+=1
        counts['answered']+=int(accepted)
        counts['correct']+=int(correct)
        counts['wrong_fact']+=int(wrong)
        counts['unsupported']+=int(unsupported)
        counts['correct_abstention']+=int(not accepted and not is_answerable)
        counts['missed_answerable']+=int(not accepted and is_answerable)
        if (wrong or unsupported or (not accepted and is_answerable)) and len(examples)<5:
            examples.append({'query_id':p['query_id'],'case':'wrong_fact' if wrong else 'unsupported' if unsupported else 'missed_answerable','score':score,'top_fact_id':top,'acceptable_fact_ids':p['acceptable_fact_ids']})
    n=counts['total']; a=counts['answered']
    return dict(counts)|{'coverage':round(a/n,4) if n else None,'answer_precision':round(counts['correct']/a,4) if a else None,'correct_answer_rate_all':round(counts['correct']/n,4) if n else None,'correct_answer_rate_answerable':round(counts['correct']/counts['answerable'],4) if counts['answerable'] else None,'examples':examples}

for model in MODELS:
    report_path=RESULTS/f'{model}_report.json'
    pred_path=RESULTS/f'{model}_predictions.jsonl'
    if not report_path.exists() or not pred_path.exists():
        print('MISSING:',report_path.name if not report_path.exists() else pred_path.name)
        continue
    rep=json.loads(report_path.read_text(encoding='utf-8'))
    rows=load_jsonl(pred_path)
    validation=[p for p in rows if p['split']=='validation']
    test=[p for p in rows if p['split']=='test']
    chosen=rep['validation_tuned_threshold']
    print('\n'+'='*65+'\n'+model+'\n'+'='*65)
    print('validation_n:',len(validation),'test_n:',len(test),'saved_threshold:',chosen)
    print('Validation at frozen threshold:',json.dumps(evaluate(validation,chosen),ensure_ascii=False,indent=2))
    print('Test at frozen threshold:',json.dumps(evaluate(test,chosen),ensure_ascii=False,indent=2))
    candidates=sorted({float(p['scores'][0]) for p in validation if p.get('scores')})
    if not candidates:
        print('No scores available'); continue
    cuts=[candidates[0]-.000001]+[(x+y)/2 for x,y in zip(candidates,candidates[1:])]+[candidates[-1]+.000001]
    trial=[]
    for t in cuts:
        m=evaluate(validation,t)
        trial.append({'threshold':round(t,6),'correct':m['correct'],'wrong_fact':m['wrong_fact'],'unsupported':m['unsupported'],'precision':m['answer_precision'],'coverage':m['coverage']})
    safe=[x for x in trial if x['wrong_fact']==0 and x['unsupported']==0]
    safe.sort(key=lambda x:(-x['correct'],-x['coverage'],x['threshold']))
    print('Best zero-error-on-validation candidates (NOT guaranteed safe on unseen data):')
    print(json.dumps(safe[:5],indent=2))
    dest=RESULTS/f'{model}_step11_threshold_diagnostic.json'
    dest.write_text(json.dumps({'model':model,'frozen_threshold':chosen,'validation':evaluate(validation,chosen),'test_at_frozen_threshold':evaluate(test,chosen),'validation_zero_error_candidates':safe[:5],'note':'Never select a threshold using held-out test results. These are synthetic-fixture metrics, not clinical reliability.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',dest.relative_to(ROOT))
