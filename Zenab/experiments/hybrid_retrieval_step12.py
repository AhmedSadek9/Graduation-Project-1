"""WANAS Step 12: offline BM25 + E5 reciprocal-rank fusion.
Place in experiments/ and run: python experiments/hybrid_retrieval_step12.py
Uses saved prediction files only; no transformers, PyTorch, network or downloads.
Exploratory synthetic benchmark only. Do not use this as a clinical safety gate.
"""
from pathlib import Path
import json, csv, statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments' / 'results'
BM = RESULTS / 'bm25_predictions.jsonl'
E5 = RESULTS / 'multilingual_e5_small_predictions.jsonl'
REPORT = RESULTS / 'multilingual_e5_small_report.json'
OUT = RESULTS / 'hybrid_step12_report.json'
PRED = RESULTS / 'hybrid_step12_predictions.jsonl'


def load(path):
    if not path.is_file():
        raise SystemExit(f'Missing {path}. Run previous comparison and keep its results first.')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

bm = {r['query_id']:r for r in load(BM)}
e5 = {r['query_id']:r for r in load(E5)}
rep = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.is_file() else None
if rep is None:
    raise SystemExit('Missing E5 report; run the semantic comparison first.')
if set(bm) != set(e5):
    raise SystemExit('Query IDs differ between BM25 and E5. Refusing unmatched comparison.')
original_e5_threshold = float(rep['validation_tuned_threshold'])


def fuse(b, e, weight):
    # Each method ranks the SAME person-filtered, verified-in-demo candidate set.
    pool_b = set(b['candidate_ids'])
    pool_e = set(e['candidate_ids'])
    if pool_b != pool_e:
        raise ValueError(f"Different candidate filters for query {b['query_id']}")
    if b['person_id'] != e['person_id'] or b['split'] != e['split'] or b['expected_outcome'] != e['expected_outcome']:
        raise ValueError(f"Mismatched metadata for query {b['query_id']}")
    if set(b['acceptable_fact_ids']) != set(e['acceptable_fact_ids']):
        raise ValueError(f"Mismatched labels for query {b['query_id']}")
    br = {fact:i+1 for i,fact in enumerate(b['ranked_fact_ids'])}
    er = {fact:i+1 for i,fact in enumerate(e['ranked_fact_ids'])}
    # RRF k=60, E5 weight determined ONLY on validation.
    scores = {f: (weight / (60+er[f]) if f in er else 0) + ((1-weight)/(60+br[f]) if f in br else 0) for f in pool_b}
    ranked = sorted(scores, key=lambda fact:(-scores[fact], fact))
    e5_lookup = dict(zip(e['ranked_fact_ids'],e['scores']))
    return ranked, scores, e5_lookup


def decision(b,e,weight,threshold):
    ranked, fused_scores, e5_lookup = fuse(b,e,weight)
    top = ranked[0] if ranked else None
    # Threshold is applied to the E5 score for THE HYBRID-SELECTED fact,
    # not necessarily E5's original top candidate. This is an experimental gate.
    top_e5_score = e5_lookup.get(top, float('-inf'))
    selected = top if top is not None and top_e5_score >= threshold else None
    return {
        'query_id':e['query_id'], 'person_id':e['person_id'], 'split':e['split'],
        'expected_outcome':e['expected_outcome'], 'acceptable_fact_ids':e['acceptable_fact_ids'],
        'candidate_ids':e['candidate_ids'], 'ranked_fact_ids':ranked,
        'selected_fact_id':selected,'top_hybrid_fact_id':top,
        'top_hybrid_e5_score':round(top_e5_score,6) if top is not None else None,
        'hybrid_rrf_scores':[round(fused_scores[x],9) for x in ranked],
        'bm25_top_fact_id':b['ranked_fact_ids'][0] if b['ranked_fact_ids'] else None,
        'e5_top_fact_id':e['ranked_fact_ids'][0] if e['ranked_fact_ids'] else None,
        'threshold':threshold,'e5_weight':weight,'upstream_origin':'oracle_synthetic',
    }


def metrics(rows):
    a=[p for p in rows if p['expected_outcome']=='answerable']
    u=[p for p in rows if p['expected_outcome']!='answerable']
    rank=lambda p:next((i for i,fid in enumerate(p['ranked_fact_ids'],1) if fid in p['acceptable_fact_ids']),None)
    supported=sum(bool(p['selected_fact_id']) and p['selected_fact_id'] in p['acceptable_fact_ids'] for p in a)
    wrong=sum(bool(p['selected_fact_id']) and p['selected_fact_id'] not in p['acceptable_fact_ids'] for p in a)
    unsupported=sum(bool(p['selected_fact_id']) for p in u)
    answered=supported+wrong+unsupported
    return {'n':len(rows),'answerable_n':len(a),'unanswerable_n':len(u),
      'raw_rank1_correct_n':sum(bool(p['ranked_fact_ids']) and p['ranked_fact_ids'][0] in p['acceptable_fact_ids'] for p in a),
      'accepted_correct_n':supported,'accepted_wrong_fact_n':wrong,'unsupported_answer_n':unsupported,
      'correct_abstention_n':sum(not p['selected_fact_id'] for p in u),
      'missed_answerable_n':sum(not p['selected_fact_id'] for p in a),
      'accepted_coverage_all':round(answered/len(rows),4) if rows else None,
      'answer_precision':round(supported/answered,4) if answered else None,
      'answerable_correct_rate':round(supported/len(a),4) if a else None,
      'recall_at_3':round(sum((rank(p) or 999)<=3 for p in a)/len(a),4) if a else None,
      'mrr_at_5':round(sum(1/rank(p) for p in a if rank(p) and rank(p)<=5)/len(a),4) if a else None}

# Validation-only selection: candidates are E5 raw score cutoffs from validation,
# with RRF weights chosen on validation. Penalize wrong / unsupported threefold.
validation_ids=sorted(q for q in bm if bm[q]['split']=='validation')
if not validation_ids:raise SystemExit('No validation rows; cannot tune.')
weights=(0.5,0.65,0.8,0.9)
all_candidates=[]
for w in weights:
    scores=sorted({float(x) for q in validation_ids for x in e5[q]['scores']})
    thresholds=[scores[0]-1e-6]+[(x+y)/2 for x,y in zip(scores,scores[1:])]+[scores[-1]+1e-6]
    for t in thresholds:
        rows=[decision(bm[q],e5[q],w,t) for q in validation_ids]
        m=metrics(rows)
        utility=m['accepted_correct_n']-3*(m['accepted_wrong_fact_n']+m['unsupported_answer_n'])
        all_candidates.append((utility, m['accepted_correct_n'], -(m['accepted_wrong_fact_n']+m['unsupported_answer_n']),w,t,m))
# Prefer E5-heavy weight and higher threshold only as deterministic tie breaks.
best=max(all_candidates,key=lambda r:(r[0],r[1],r[2],r[3],r[4]))
_,_,_,w,t,vmet=best
pred=[decision(bm[q],e5[q],w,t) for q in sorted(bm)]
test=[p for p in pred if p['split']=='test']
test_met=metrics(test)
with PRED.open('w',encoding='utf-8') as f:
    for p in pred:f.write(json.dumps(p,ensure_ascii=False)+'\n')
result={'method':'BM25 + multilingual-e5-small weighted RRF','k':60,'e5_weight':w,'bm25_weight':1-w,
 'retrieval_acceptance_threshold_on_hybrid_fact_e5_score':t,'e5_original_frozen_threshold_for_reference':original_e5_threshold,
 'selection':'RRF weight and E5-score gate tuned ONLY on validation with correct - 3*(wrong_fact+unsupported); NOT tuned on test',
 'validation':vmet,'test_exploratory':test_met,
 'limitations':['All source facts are synthetic, not caregiver verified',
 'Test predictions were inspected in earlier steps, so subsequent model development cannot treat this test as pristine held-out evidence',
 'No actual face recognition or STT predictions',
 'RRF scores are rank-fusion values, NOT probabilities',
 'Does not measure real-world or clinical safety',
 'Latency for fusion is not a fair end-to-end model latency measurement.']}
OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('STEP 12 — OFFLINE HYBRID EVALUATION (no downloads)')
print('Chosen ONLY on validation:',{'e5_weight':w,'threshold':round(t,6)})
print('VALIDATION:',json.dumps(vmet,indent=2))
print('TEST — EXPLORATORY:',json.dumps(test_met,indent=2))
print('Saved:',OUT.relative_to(ROOT),'and',PRED.relative_to(ROOT))
print('IMPORTANT: This test set was previously inspected; confirm any final selection on a new independent split.')
