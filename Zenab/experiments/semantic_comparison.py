"""Optional offline/online pretrained model comparison; requires sentence-transformers and cached/downloadable checkpoints.
Command: python experiments/semantic_comparison.py
Uses identical question IDs, person/verification filters and split IDs as BM25.
No result file is claimed until models actually run.
"""
from pathlib import Path
import json, csv, math, time, statistics, collections
ROOT=Path(__file__).resolve().parents[1]
try:
 from sentence_transformers import SentenceTransformer
except ImportError as e:
 raise SystemExit('Install dependencies first: python -m pip install sentence-transformers . No semantic results generated.') from e
F=list(csv.DictReader(open(ROOT/'benchmark/fact_manifest.csv',encoding='utf-8-sig')))
Q=list(csv.DictReader(open(ROOT/'benchmark/question_manifest.csv',encoding='utf-8-sig')))
OUT=ROOT/'experiments/results';OUT.mkdir(exist_ok=True)
MODELS={'all-MiniLM-L6-v2':'sentence-transformers/all-MiniLM-L6-v2','multilingual-e5-small':'intfloat/multilingual-e5-small'}

def cosine(a,b):
 return sum(float(x)*float(y) for x,y in zip(a,b)) / (math.sqrt(sum(float(x)**2 for x in a))*math.sqrt(sum(float(x)**2 for x in b)) +1e-12)

def rank(model, q, pool, e5):
 # Official E5 prefix conventions (query: / passage:) and same prefixes at train/validation/test.
 qt=('query: ' if e5 else '')+q['question']
 ps=[('passage: ' if e5 else '')+f['person_name']+'. '+f['memory_fact'] for f in pool]
 enc=model.encode([qt]+ps,normalize_embeddings=True,show_progress_bar=False)
 return sorted([(f,cosine(enc[0],x)) for f,x in zip(pool,enc[1:])],key=lambda z:(-z[1],z[0]['fact_id']))

for label,checkpoint in MODELS.items():
 model=SentenceTransformer(checkpoint)
 e5='e5' in label
 pred=[]
 for q in Q:
  start=time.perf_counter()
  pool=[f for f in F if f['person_id']==q['person_id'] and f['source_type']=='synthetic_test_data' and f['verification_status']=='verified_in_fictional_scenario']
  ranked=rank(model,q,pool,e5) if pool else []
  pred.append({'query_id':q['query_id'],'split':q['split'],'person_id':q['person_id'],'expected_outcome':q['expected_outcome'],'acceptable_fact_ids':json.loads(q['acceptable_fact_ids']),'candidate_ids':[f['fact_id'] for f in pool],'ranked_fact_ids':[f['fact_id'] for f,v in ranked],'scores':[round(float(v),6) for f,v in ranked],'latency_ms':(time.perf_counter()-start)*1000})
 # On validation, choose threshold maximizing supported answer - 3*unsupported/wrong;
 # then evaluate untouched test using that frozen score threshold.
 val=[p for p in pred if p['split']=='validation']
 scores=sorted({0,*[p['scores'][0] for p in val if p['scores']]});candidates=[min(scores)-1]+[(a+b)/2 for a,b in zip(scores,scores[1:])]+[max(scores)+1]
 def utility(t):
  return sum((1 if p['ranked_fact_ids'][0] in p['acceptable_fact_ids'] else -3) for p in val if p['scores'] and p['scores'][0]>=t)
 t=max(candidates,key=utility)
 for p in pred:
  p['selected_fact_id']=p['ranked_fact_ids'][0] if p['scores'] and p['scores'][0]>=t else None
  p['threshold']=t;p['model']=label;p['upstream_origin']='oracle_synthetic'
 def report(split):
  arr=[p for p in pred if p['split']==split];ans=[p for p in arr if p['expected_outcome']=='answerable'];un=[p for p in arr if p['expected_outcome']!='answerable']
  ranks=[next((i for i,f in enumerate(p['ranked_fact_ids'],1) if f in p['acceptable_fact_ids']),None) for p in ans]
  return {'n':len(arr),'answerable_n':len(ans),'unanswerable_n':len(un),'top1':sum(p['selected_fact_id'] in p['acceptable_fact_ids'] for p in ans)/len(ans) if ans else None,'recall_at_3':sum(r is not None and r<=3 for r in ranks)/len(ans) if ans else None,'mrr_at_5':sum(1/r for r in ranks if r is not None and r<=5)/len(ans) if ans else None,'correct_abstentions':sum(p['selected_fact_id'] is None for p in un),'unsupported_answers':sum(p['selected_fact_id'] is not None for p in un),'latency_ms_median':statistics.median(p['latency_ms'] for p in arr)}
 file=label.replace('/','_').replace('-','_')
 (OUT/f'{file}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pred))
 (OUT/f'{file}_report.json').write_text(json.dumps({'checkpoint':checkpoint,'query_prefix':'query: ' if e5 else '', 'passage_prefix':'passage: ' if e5 else '', 'validation_tuned_threshold':t,'results':{s:report(s) for s in ['validation','test']}, 'limitations':['Synthetic questions and facts','No actual STT or face outputs','Checkpoint download / machine-dependent latency']},indent=2))
 print(label,'test',report('test'))
