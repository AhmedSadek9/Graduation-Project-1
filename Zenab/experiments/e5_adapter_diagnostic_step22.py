"""WANAS Step 22 — reproduce E5 demo adapter decisions from SAVED rankings.

Place this file in experiments/ and run from project root:
    python experiments/e5_adapter_diagnostic_step22.py

No downloads or model execution. Diagnostic only, using old synthetic questions,
oracle person IDs and previously inspected test set (NOT independent/production).
Matches Step 21 candidate type filtering + per-candidate E5 score acceptance.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fact_type_guard_step15 import allowed_fact_types

RESULTS = ROOT / 'experiments' / 'results'
QFILE = ROOT / 'benchmark' / 'question_manifest.csv'
FFILE = ROOT / 'benchmark' / 'fact_manifest.csv'
PFILE = RESULTS / 'multilingual_e5_small_predictions.jsonl'
CF = ROOT / 'experiments' / 'step16' / 'frozen_e5_config_step16.json'
OUT = RESULTS / 'e5_adapter_step22_diagnostic.json'
PRED_OUT = RESULTS / 'e5_adapter_step22_predictions.jsonl'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    answerable = [r for r in rows if r['expected_outcome'] == 'answerable']
    unanswerable = [r for r in rows if r['expected_outcome'] != 'answerable']
    accepted = [r for r in rows if r['selected_fact_id'] is not None]
    correct = [r for r in answerable if r['selected_fact_id'] in r['acceptable_fact_ids']]
    wrong = [r for r in answerable if r['selected_fact_id'] is not None and r['selected_fact_id'] not in r['acceptable_fact_ids']]
    unsupported = [r for r in unanswerable if r['selected_fact_id'] is not None]
    raw_top1 = sum(bool(r['raw_rank1_fact_id']) and r['raw_rank1_fact_id'] in r['acceptable_fact_ids'] for r in answerable)
    recall3 = sum(bool(set(r['ranked_fact_ids'][:3]) & set(r['acceptable_fact_ids'])) for r in answerable)
    mrr5 = sum(next((1/i for i, fid in enumerate(r['ranked_fact_ids'][:5],1) if fid in r['acceptable_fact_ids']),0) for r in answerable)
    return {
        'total':len(rows), 'answerable_n':len(answerable), 'unanswerable_n':len(unanswerable),
        'raw_rank1_correct_n':raw_top1,
        'recall_at_3':round(recall3/len(answerable),4) if answerable else None,
        'mrr_at_5':round(mrr5/len(answerable),4) if answerable else None,
        'accepted_n':len(accepted), 'accepted_correct_n':len(correct),
        'accepted_wrong_fact_n':len(wrong), 'unsupported_answer_n':len(unsupported),
        'correct_abstentions_n':len(unanswerable)-len(unsupported),
        'missed_answerable_n':len(answerable)-len(correct)-len(wrong),
        'accepted_precision':round(len(correct)/len(accepted),4) if accepted else None,
        'answerable_correct_rate':round(len(correct)/len(answerable),4) if answerable else None,
        'abstention_reasons':dict(Counter(r['abstention_reason'] for r in rows if r['selected_fact_id'] is None)),
    }


def main():
    for path in (QFILE,FFILE,PFILE,CF):
        if not path.is_file():
            raise SystemExit(f'Missing prerequisite: {path}\nCopy Step 22 into experiments/ inside WANAS_Zinab_Final.')
    config=json.loads(CF.read_text(encoding='utf-8'))
    if config.get('model') != 'intfloat/multilingual-e5-small' or config.get('query_variant') != 'question_only':
        raise SystemExit('Unexpected frozen model or query variant; refusing comparison')
    threshold=float(config['acceptance_threshold'])
    if not math.isfinite(threshold): raise SystemExit('Invalid frozen threshold')
    if config.get('acceptance_rule','').find('allowed question-intent fact_type') < 0:
        raise SystemExit('Frozen config changed; audit the acceptance rule first')
    expected_hash=config.get('checksums_sha256',{}).get('fact_manifest')
    actual_hash=hashlib.sha256(FFILE.read_bytes()).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise SystemExit('Fact manifest has changed since Step 16 freeze; cannot compare safely')
    facts={f['fact_id']:f for f in read_csv(FFILE)}
    questions={q['query_id']:q for q in read_csv(QFILE)}
    original=[json.loads(x) for x in PFILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    output=[]
    seen=set()
    for p in original:
        qid=p['query_id']
        if qid in seen: raise SystemExit(f'Duplicate query ID: {qid}')
        seen.add(qid)
        q=questions.get(qid)
        if q is None: raise SystemExit(f'Unknown query ID: {qid}')
        if q['person_id'] != p['person_id'] or q['split'] != p['split']:
            raise SystemExit(f'Question/person/split mismatch: {qid}')
        ranked=p.get('ranked_fact_ids',[])
        scores=p.get('scores',[])
        if len(ranked)!=len(scores): raise SystemExit(f'Rank/score mismatch: {qid}')
        pool=[fid for fid in p.get('candidate_ids',[]) if fid in facts]
        if len(pool)!=len(p.get('candidate_ids',[])) or set(ranked)!=set(pool):
            raise SystemExit(f'Incomplete candidate ranking: {qid}')
        for fid in pool:
            f=facts[fid]
            if f['person_id'] != p['person_id'] or f['source_type'] != 'synthetic_test_data' or f['verification_status'] != 'verified_in_fictional_scenario':
                raise SystemExit(f'Candidate filter violation: {qid}, {fid}')
        allowed, intent_reason=allowed_fact_types(q['question'])
        selected=None
        selected_score=None
        top_type_candidate=None
        top_type_score=None
        if allowed is not None and len(allowed)==0:
            reason=intent_reason
        elif allowed is None:
            reason='unrecognized_question_intent'
        else:
            typed=[(fid,float(score)) for fid,score in zip(ranked,scores) if facts[fid]['fact_type'] in allowed]
            if not typed:
                reason='no_matching_verified_fact_type'
            else:
                top_type_candidate,top_type_score=typed[0]
                if top_type_score >= threshold:
                    selected,selected_score=typed[0]
                    reason=None
                else:
                    reason='below_frozen_retrieval_threshold'
        acceptable=json.loads(q['acceptable_fact_ids'])
        output.append({
            'query_id':qid,'split':q['split'],'person_id':q['person_id'],'question':q['question'],
            'upstream_origin':'oracle_synthetic','expected_outcome':q['expected_outcome'],
            'acceptable_fact_ids':acceptable,'raw_rank1_fact_id':ranked[0] if ranked else None,
            'raw_rank1_score':scores[0] if scores else None,'candidate_ids':pool,
            'ranked_fact_ids':ranked,'scores':scores,'allowed_fact_types':sorted(allowed) if allowed is not None else None,
            'question_intent':intent_reason,'top_type_candidate_id':top_type_candidate,
            'top_type_score':top_type_score,'selected_fact_id':selected,'selected_score':selected_score,
            'threshold':threshold,'abstention_reason':reason,
            'decision_source':'reconstructed_from_saved_e5_rankings; not live face or adapter execution'
        })
    reports={split:summarize([r for r in output if r['split']==split]) for split in ('train','validation','test')}
    obj={
        'scope':'exploratory previously inspected synthetic data; NOT new held-out evaluation',
        'model':config['model'],'threshold':threshold,'query_variant':'question_only',
        'behavior':'Step21 fact-type filter BEFORE acceptance; E5 ranks and scores read from saved prediction file',
        'not_tested':'actual face/STT; live model latency; independent reviewer; real caregiver approval',
        'metrics':reports,
    }
    RESULTS.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    PRED_OUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    print('STEP 22 — E5 ADAPTER DECISION DIAGNOSTIC (OFFLINE / NO MODEL DOWNLOADS)')
    print('Frozen threshold:',threshold,'| Saved ranking rows:',len(output))
    for split in ('validation','test'):
        print('\n'+split.upper()+':',json.dumps(reports[split],ensure_ascii=False))
        for r in [x for x in output if x['split']==split and x['expected_outcome']=='answerable' and x['selected_fact_id'] is None][:3]:
            print('  MISSED:',r['query_id'],'top_type=',r['top_type_candidate_id'],'score=',r['top_type_score'],'reason=',r['abstention_reason'])
    print('\nSaved:',OUT.relative_to(ROOT),'and',PRED_OUT.relative_to(ROOT))
    print('WARNING: Diagnostic replay of previously inspected synthetic results; not independent or actual-upstream evaluation.')

if __name__=='__main__': main()
