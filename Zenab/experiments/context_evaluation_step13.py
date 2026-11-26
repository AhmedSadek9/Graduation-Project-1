"""WANAS Step 13: reproducible *synthetic* context ablation using cached E5.
Run: python experiments/context_evaluation_step13.py
Does not edit manifests or freeze a new final threshold. No actual face/STT data used.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / 'benchmark' / 'fact_manifest.csv'
QUESTIONS = ROOT / 'benchmark' / 'question_manifest.csv'
RESULTS = ROOT / 'experiments' / 'results'
MODEL_NAME = 'intfloat/multilingual-e5-small'
PRED_FILE = RESULTS / 'multilingual_e5_small_report.json'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def is_demo_approved(f):
    return (f['source_type'] == 'synthetic_test_data'
            and f['verification_status'] == 'verified_in_fictional_scenario'
            and f['review_status'] == 'draft_for_team_review')


def report(rows):
    answerable = [x for x in rows if x['expected_outcome'] == 'answerable']
    unanswerable = [x for x in rows if x['expected_outcome'] != 'answerable']
    accepted = [x for x in rows if x['selected_fact_id']]
    correct = [x for x in accepted if x['selected_fact_id'] in x['acceptable_fact_ids']]
    raw_correct = [x for x in answerable if x['raw_rank_1_fact_id'] in x['acceptable_fact_ids']]
    ranks = [x['acceptable_rank'] for x in answerable]
    return {
        'n': len(rows), 'answerable_n': len(answerable), 'unanswerable_n': len(unanswerable),
        'raw_rank1_correct_n': len(raw_correct),
        'accepted_n': len(accepted), 'accepted_correct_n': len(correct),
        'accepted_wrong_fact_n': sum(x['expected_outcome']=='answerable' and bool(x['selected_fact_id']) and x['selected_fact_id'] not in x['acceptable_fact_ids'] for x in rows),
        'unsupported_answer_n': sum(x['expected_outcome']!='answerable' and bool(x['selected_fact_id']) for x in rows),
        'correct_abstention_n': sum(x['expected_outcome']!='answerable' and not x['selected_fact_id'] for x in rows),
        'missed_answerable_n': sum(x['expected_outcome']=='answerable' and not x['selected_fact_id'] for x in rows),
        'answer_precision': round(len(correct)/len(accepted),4) if accepted else None,
        'accepted_coverage_all': round(len(accepted)/len(rows),4) if rows else None,
        'answerable_correct_rate': round(len(correct)/len(answerable),4) if answerable else None,
        'recall_at_3': round(sum(r is not None and r<=3 for r in ranks)/len(answerable),4) if answerable else None,
        'mrr_at_5': round(sum(1/r for r in ranks if r is not None and r<=5)/len(answerable),4) if answerable else None,
        'query_latency_ms_median': round(statistics.median(x['query_latency_ms'] for x in rows),3) if rows else None,
    }


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit('Install sentence-transformers in this Python environment first.') from e
    try:
        import numpy as np
    except ImportError as e:
        raise SystemExit('numpy is required.') from e

    facts = read_csv(FACTS)
    questions = [q for q in read_csv(QUESTIONS) if q['split'] in ('validation','test')]
    threshold = json.loads(PRED_FILE.read_text(encoding='utf-8'))['validation_tuned_threshold']
    ids = {f['person_id']: f['person_name'] for f in facts}
    people = sorted(ids)
    pool_by_person = {pid: [f for f in facts if f['person_id']==pid and is_demo_approved(f)] for pid in people}
    print('Loading E5 from installed/cached checkpoint (first load can check cache)...')
    model = SentenceTransformer(MODEL_NAME)
    embeddings = {}
    # Encode each passage once to avoid repeated scoring work across scenarios.
    for pid, pool in pool_by_person.items():
        if pool:
            texts = ['passage: '+f['person_name']+'. '+f['memory_fact'] for f in pool]
            embeddings[pid] = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    scenarios = ['question_only','question_plus_identity','question_identity_last_request',
                 'stale_last_request_guarded','unknown_face_guarded']
    preds=[]
    for q in questions:
        pid=q['person_id']
        expected=json.loads(q['acceptable_fact_ids'])
        name=ids.get(pid,'')
        other=next((p for p in people if p != pid), None)
        for scenario in scenarios:
            # All contexts below are deliberately synthetic, NOT actual audio or recognition outputs.
            if scenario=='question_only':
                text=q['question']; context=''; face_status='recognized'
            elif scenario=='question_plus_identity':
                context=f'Current recognized person: {name} ({pid}).'
                text=q['question']+' '+context; face_status='recognized'
            elif scenario=='question_identity_last_request':
                context=f'Current recognized person: {name} ({pid}). Previous request: Tell me about {name}.'
                text=q['question']+' '+context; face_status='recognized'
            elif scenario=='stale_last_request_guarded':
                context=f'Current recognized person: {name} ({pid}). Previous request: Tell me about {ids[other]} ({other}).'
                text=q['question']+' '+context; face_status='recognized'
            else:
                context='Face recognition returned unknown identity.'
                text=q['question']; face_status='unknown'
            pool=pool_by_person.get(pid,[])
            row={'query_id':q['query_id'],'original_question_id':q['original_question_id'],
                 'split':q['split'],'scenario':scenario,'person_id':pid,'expected_person_id':q['expected_person_id'],
                 'question':q['question'],'expected_outcome':q['expected_outcome'],
                 'acceptable_fact_ids':expected,'upstream_origin':'synthetic_context_oracle_person',
                 'face_status':face_status,'context_text':context,
                 'candidate_fact_ids':[f['fact_id'] for f in pool],
                 'threshold':threshold,'threshold_source':'existing validation-tuned E5 threshold',
                 'model':MODEL_NAME,'ranked_fact_ids':[],'scores':[],
                 'raw_rank_1_fact_id':None,'acceptable_rank':None,'selected_fact_id':None,
                 'abstention_reason':None,'query_latency_ms':0.0}
            start=time.perf_counter()
            # Unknown-face case must not attempt retrieval at all.
            if face_status=='unknown':
                row['abstention_reason']='unknown_face'
            # Guard against irrelevant previous-person context: ignore stale last request, do not use it for query.
            else:
                if scenario=='stale_last_request_guarded':
                    text=q['question']+f' Current recognized person: {name} ({pid}).'
                    row['abstention_reason']='stale_previous_request_ignored'  # diagnostic, overwritten if answerable
                if pool:
                    emb=model.encode(['query: '+text],normalize_embeddings=True,show_progress_bar=False)[0]
                    sims=np.asarray(embeddings[pid]) @ np.asarray(emb)
                    order=sorted(range(len(pool)),key=lambda i:(-float(sims[i]),pool[i]['fact_id']))
                    row['ranked_fact_ids']=[pool[i]['fact_id'] for i in order]
                    row['scores']=[round(float(sims[i]),6) for i in order]
                    row['raw_rank_1_fact_id']=row['ranked_fact_ids'][0]
                    row['acceptable_rank']=next((i for i,fid in enumerate(row['ranked_fact_ids'],1) if fid in expected),None)
                    if float(sims[order[0]])>=threshold:
                        row['selected_fact_id']=row['raw_rank_1_fact_id']
                        row['abstention_reason']=None
                    else:
                        row['abstention_reason']='retrieval_score_below_threshold'
                else:
                    row['abstention_reason']='no_verified_demo_candidates'
            row['query_latency_ms']=round((time.perf_counter()-start)*1000,3)
            preds.append(row)
    RESULTS.mkdir(parents=True,exist_ok=True)
    pred_path=RESULTS/'context_step13_predictions.jsonl'
    pred_path.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds),encoding='utf-8')
    summary={
        'model':MODEL_NAME,'threshold':threshold,'threshold_tuning':'NOT retuned; frozen from prior validation',
        'scenario_origin':'synthetic/constructed examples; not measured upstream predictions',
        'data_scope':'fictional demo data; not caregiver-approved production data',
        'method_notes':['Same IDs and answer labels for all variants','E5 query: and passage: prefixes applied',
        'Candidate pool filtered by person and demo-verification before ranking',
        'Passage embeddings precomputed once; latency is query encode + score + guard, excludes load/index build',
        'Stale context guard discards previous request referencing a different person',
        'Unknown face is rejected without ranking',
        'Test split already inspected in previous steps: test is exploratory, not untouched'],
        'by_scenario':{s:{sp:report([p for p in preds if p['scenario']==s and p['split']==sp]) for sp in ('validation','test')} for s in scenarios}}
    out=RESULTS/'context_step13_report.json';out.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSTEP 13 — CONTEXT ABLATION (SYNTHETIC ONLY)')
    print('Frozen E5 threshold:',round(threshold,6))
    for scenario in scenarios:
        print('\n'+scenario)
        for split in ('validation','test'):
            a=summary['by_scenario'][scenario][split]
            print(split, 'n=',a['n'],'raw_rank1=',a['raw_rank1_correct_n'],'accepted_correct=',a['accepted_correct_n'],
                  'wrong=',a['accepted_wrong_fact_n'],'unsupported=',a['unsupported_answer_n'],
                  'abstained_unanswerable=',a['correct_abstention_n'],'latency_ms=',a['query_latency_ms_median'])
    print('\nSaved:',out.relative_to(ROOT),'and',pred_path.relative_to(ROOT))
    print('IMPORTANT: Stale/unknown cases are SIMULATED; real STT + face predictions still pending.')

if __name__=='__main__':
    main()
