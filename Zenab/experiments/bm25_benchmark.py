"""Synthetic, person-filtered BM25 evaluation. Standard library only.
Run from repository root: python experiments/bm25_benchmark.py
All synthetic dataset records must NEVER be marked caregiver-verified.
"""
import csv, json, math, re, time, collections, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FACTS=list(csv.DictReader(open(ROOT/'benchmark/fact_manifest.csv',encoding='utf-8-sig')))
QUESTIONS=list(csv.DictReader(open(ROOT/'benchmark/question_manifest.csv',encoding='utf-8-sig')))
GROUP_BY_QUERY={q['query_id']:q['original_question_id'] for q in QUESTIONS}
OUT=ROOT/'experiments/results';OUT.mkdir(exist_ok=True)
TOK=lambda s:re.findall(r"[a-z0-9]+",str(s).lower())

def eligible(row,mode='demo'):
    if mode=='demo':return row['source_type']=='synthetic_test_data' and row['verification_status']=='verified_in_fictional_scenario'
    return row['source_type']!='synthetic_test_data' and row['verification_status']=='caregiver_verified' and row['review_status']=='approved'

def candidates(q,mode='demo'):
    return [f for f in FACTS if f['person_id']==q['person_id'] and eligible(f,mode)]

def bm25_rank(query, docs, k1=1.5,b=0.75):
    if not docs:return []
    # Passage includes the actual fact, its semantic type and an explicit person name.
    tokens=[TOK(f['person_name']+' '+f['fact_type'].replace('_',' ')+' '+f['memory_fact']) for f in docs]
    df=collections.Counter(w for t in tokens for w in set(t))
    avdl=sum(map(len,tokens))/len(tokens); qs=TOK(query); scores=[]
    for f,t in zip(docs,tokens):
        tf=collections.Counter(t); score=0.0
        for term in qs:
            n=df[term];idf=math.log(1+(len(docs)-n+.5)/(n+.5))
            score+=idf*(tf[term]*(k1+1))/(tf[term]+k1*(1-b+b*len(t)/avdl)) if tf[term] else 0
        scores.append((f,score))
    return sorted(scores,key=lambda pair:(-pair[1],pair[0]['fact_id']))

def predict(q, threshold=0.0):
    start=time.perf_counter(); pool=candidates(q)
    ranks=bm25_rank(q['question'],pool)
    rankids=[f['fact_id'] for f,s in ranks]
    best=ranks[0] if ranks else None
    accepted=bool(best and best[1]>threshold)
    # No identity fallback or cross-person search.
    return {'query_id':q['query_id'],'person_id':q['person_id'],'split':q['split'],'expected_outcome':q['expected_outcome'],'acceptable_fact_ids':json.loads(q['acceptable_fact_ids']),'candidate_count':len(pool),'candidate_ids':[f['fact_id'] for f in pool],'ranked_fact_ids':rankids,'scores':[round(s,6) for f,s in ranks], 'selected_fact_id':best[0]['fact_id'] if accepted else None,'status':'answerable' if accepted else ('unknown_identity_abstain' if not pool else 'unanswerable_abstain'),'latency_ms':(time.perf_counter()-start)*1000,'threshold':threshold,'upstream_origin':'oracle_synthetic'}

def metrics(preds):
    a=[p for p in preds if p['expected_outcome']=='answerable'];u=[p for p in preds if p['expected_outcome']!='answerable']
    rank=lambda p:next((i for i,fid in enumerate(p['ranked_fact_ids'],1) if fid in p['acceptable_fact_ids']),None)
    top1=sum(p['selected_fact_id'] in p['acceptable_fact_ids'] for p in a)/len(a) if a else None
    rec3=sum(rank(p) is not None and rank(p)<=3 for p in a)/len(a) if a else None
    mrr5=sum(1/rank(p) for p in a if rank(p) is not None and rank(p)<=5)/len(a) if a else None
    correct_abstain=sum(p['selected_fact_id'] is None for p in u)
    unsupported=sum(bool(p['selected_fact_id']) for p in u)
    wrongly_answered=sum(bool(p['selected_fact_id']) and p['selected_fact_id'] not in p['acceptable_fact_ids'] for p in a)
    answered=sum(bool(p['selected_fact_id']) for p in preds)
    supported=sum(bool(p['selected_fact_id']) and p['selected_fact_id'] in p['acceptable_fact_ids'] for p in a)
    return {'question_count':len(preds),'independent_original_question_groups':len({GROUP_BY_QUERY[p['query_id']] for p in preds}), 'answerable_n':len(a),'unanswerable_or_unknown_n':len(u),'top1_correctness_answerable':top1,'recall_at_3_answerable':rec3,'mrr_at_5_answerable':mrr5,'correct_abstention_n':correct_abstain,'correct_abstention_rate':correct_abstain/len(u) if u else None,'unsupported_answer_n':unsupported,'wrong_answer_on_answerable_n':wrongly_answered,'answer_coverage':answered/len(preds) if preds else None,'answer_precision':supported/answered if answered else None,'median_latency_ms':statistics.median(p['latency_ms'] for p in preds) if preds else None, 'p95_latency_ms': sorted(p['latency_ms'] for p in preds)[min(len(preds)-1,math.ceil(.95*len(preds))-1)] if preds else None}

# Tune threshold on validation only with strong penalty for unsafe answers.
validation=[q for q in QUESTIONS if q['split']=='validation']
raw=[predict(q,-1) for q in validation]
vals=sorted({0.0,*[max(p['scores']) for p in raw if p['scores']]})
thresholds=[-0.00001]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[max(vals)+.00001]
def objective(thresh):
    p=[predict(q,thresh) for q in validation];m=metrics(p)
    supported=sum(x['selected_fact_id'] in x['acceptable_fact_ids'] for x in p if x['expected_outcome']=='answerable')
    wrong=sum(x['selected_fact_id'] is not None and x['selected_fact_id'] not in x['acceptable_fact_ids'] for x in p)
    return (supported-3*wrong, supported,-wrong,thresh)
choice=max(thresholds,key=objective)
(OUT/'bm25_frozen_configuration.json').write_text(json.dumps({'method':'BM25','k1':1.5,'b':0.75,'acceptance_threshold':choice,'tuned_on':'validation_only','candidate_filter':'exact_person_id_and_demo_verification_before_ranking','mode':'synthetic_demo','source':'WANAS_Fictional_Relationships_V2','status':'BM25 baseline, not a selected winner against semantic models'},indent=2))
all_preds=[predict(q,choice) for q in QUESTIONS]
with (OUT/'bm25_predictions.jsonl').open('w',encoding='utf-8') as f:
    for p in all_preds:f.write(json.dumps(p,ensure_ascii=False)+'\n')
report={'method':'BM25','data_origin':'synthetic_demo','threshold':choice,'counts_by_split':{s:collections.Counter(q['expected_outcome'] for q in QUESTIONS if q['split']==s) for s in ['train','validation','test']},'results':{s:metrics([p for p in all_preds if p['split']==s]) for s in ['train','validation','test']},'limitations':['Only six candidate facts/person; lexical matching only','Current source contains 120 synthetic facts; no caregiver approval','Question prompts are template-style; metrics not real-world reliability','No actual face or STT predictions']}
(OUT/'bm25_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2));print('Wrote:',OUT)
