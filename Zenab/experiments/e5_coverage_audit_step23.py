"""WANAS Step 23 — read-only coverage/error audit of frozen E5 adapter decisions.

Run from project root:
    python experiments/e5_coverage_audit_step23.py

No ML packages, model loading, internet, or modification of the frozen pipeline.
All results exploratory: predictions were previously inspected, and oracle IDs
and synthetic facts were used.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments' / 'results'
SOURCE = RESULTS / 'e5_adapter_step22_predictions.jsonl'
CONFIG = ROOT / 'experiments' / 'step16' / 'frozen_e5_config_step16.json'
OUT_CSV = RESULTS / 'e5_coverage_audit_step23.csv'
OUT_JSON = RESULTS / 'e5_coverage_audit_step23.json'


def classify(r):
    expected = r['expected_outcome'] == 'answerable'
    selected = r.get('selected_fact_id')
    acceptable = set(r.get('acceptable_fact_ids') or [])
    if not expected:
        return 'unsupported_accepted' if selected else 'correct_abstention'
    if selected:
        return 'correct_answer' if selected in acceptable else 'wrong_fact'
    if r.get('abstention_reason') == 'below_frozen_retrieval_threshold':
        return 'threshold_rejected_correct_candidate' if r.get('top_type_candidate_id') in acceptable else 'threshold_rejected_wrong_candidate'
    if r.get('abstention_reason') == 'unrecognized_question_intent':
        return 'intent_unrecognized'
    if r.get('abstention_reason') == 'no_matching_verified_fact_type':
        return 'no_matching_type'
    return 'other_abstention'


def main():
    if not SOURCE.exists() or not CONFIG.exists():
        raise SystemExit('Missing Step 22 predictions or Step 16 config. Run Step 22 before Step 23.')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
    expected_threshold = float(cfg['acceptance_threshold'])
    if cfg.get('model') != 'intfloat/multilingual-e5-small':
        raise SystemExit('Unexpected model in frozen configuration')
    ids = [(r['split'], r['query_id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('Duplicate split/query records: aborting audit')
    for r in rows:
        if abs(float(r['threshold'])-expected_threshold)>1e-9:
            raise SystemExit('Threshold mismatch; do not mix configurations')
        r['audit_category'] = classify(r)
        r['oracle_candidate_matches'] = r.get('top_type_candidate_id') in set(r.get('acceptable_fact_ids') or [])
    export=[]
    for r in rows:
        if r['split'] not in ('validation','test'):
            continue
        export.append({
            'split': r['split'], 'query_id':r['query_id'], 'person_id':r['person_id'],
            'question':r['question'], 'expected_outcome':r['expected_outcome'],
            'acceptable_fact_ids':json.dumps(r.get('acceptable_fact_ids',[])),
            'top_ranked_fact_id':r.get('raw_rank1_fact_id'),
            'top_allowed_fact_id':r.get('top_type_candidate_id'),
            'top_allowed_score':r.get('top_type_score'),
            'selected_fact_id':r.get('selected_fact_id'),
            'abstention_reason':r.get('abstention_reason'),
            'audit_category':r['audit_category'],
            'oracle_candidate_matches':r['oracle_candidate_matches'],
            'upstream_origin':r.get('upstream_origin'),
        })
    with OUT_CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    stats={}
    for split in ('validation','test'):
        group=[r for r in rows if r['split']==split]
        counts=Counter(r['audit_category'] for r in group)
        stats[split]={'questions':len(group),'answerable':sum(r['expected_outcome']=='answerable' for r in group),
                      'categories':dict(sorted(counts.items())),
                      'threshold_correct_candidate_rejected':counts['threshold_rejected_correct_candidate'],
                      'threshold_wrong_candidate_rejected':counts['threshold_rejected_wrong_candidate'],
                      'intent_unrecognized':counts['intent_unrecognized']}
    OUT_JSON.write_text(json.dumps({'scope':'EXPLORATORY; synthetic fixture, prior viewed test, oracle person IDs',
        'frozen_threshold':expected_threshold,'configuration_id':cfg.get('configuration_id'),
        'important':'Oracle candidate comparison is diagnostic only; not available to runtime.',
        'metrics':stats},ensure_ascii=False,indent=2),encoding='utf-8')
    print('STEP 23 — E5 COVERAGE ERROR AUDIT (offline/read-only)')
    print('Frozen threshold:',expected_threshold)
    for split in ('validation','test'):
        d=stats[split]
        print(f'\n{split.upper()} n={d["questions"]} answerable={d["answerable"]}')
        for cat,n in d['categories'].items(): print(f'  {cat}: {n}')
    print('\nSaved:',OUT_CSV.relative_to(ROOT),'and',OUT_JSON.relative_to(ROOT))
    print('No frozen settings changed. No new held-out accuracy claims.')

if __name__=='__main__':main()
