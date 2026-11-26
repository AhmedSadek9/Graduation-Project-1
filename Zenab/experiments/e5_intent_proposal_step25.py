"""Step25 replay of proposed intents against Step22 saved E5 rankings.
Reads previous synthetic ranking rows; does not change frozen config or Bridge.
Run: python experiments/e5_intent_proposal_step25.py
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from intent_rules_step25 import proposed_allowed_fact_types

SOURCE=ROOT/'experiments/results/e5_adapter_step22_predictions.jsonl'
FACTS=ROOT/'benchmark/fact_manifest.csv'
OUT=ROOT/'experiments/results/e5_intent_proposal_step25.json'
ROWS=ROOT/'experiments/results/e5_intent_proposal_step25_changes.csv'

def metrics(rows, key):
    a=[r for r in rows if r['expected_outcome']=='answerable']
    u=[r for r in rows if r['expected_outcome']!='answerable']
    good=sum(bool(r[key]) and r[key] in r['acceptable_fact_ids'] for r in a)
    wrong=sum(bool(r[key]) and r[key] not in r['acceptable_fact_ids'] for r in a)
    unsafe=sum(bool(r[key]) for r in u)
    return {'n':len(rows),'answerable_n':len(a),'unanswerable_n':len(u),
            'accepted_correct_n':good,'accepted_wrong_fact_n':wrong,
            'unsupported_answer_n':unsafe,'correct_abstentions_n':len(u)-unsafe,
            'missed_answerable_n':len(a)-good-wrong}

def main():
    if not SOURCE.exists() or not FACTS.exists():
        raise SystemExit('Missing saved Step22 predictions or fact manifest')
    facts={r['fact_id']:r for r in csv.DictReader(FACTS.open(encoding='utf-8-sig',newline=''))}
    original=[json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
    output=[]; seen=set(); changed=[]
    for r0 in original:
        r=dict(r0); key=(r['split'],r['query_id'])
        if key in seen: raise RuntimeError(f'Duplicate key: {key}')
        seen.add(key)
        if len(r['ranked_fact_ids']) != len(r['scores']): raise RuntimeError(f'Misaligned ranking: {key}')
        allowed, rule=proposed_allowed_fact_types(r['question'])
        pick=None
        if allowed:
            for fid, score in zip(r['ranked_fact_ids'],r['scores']):
                f=facts.get(fid)
                if f is None or f['person_id'] != r['person_id']:
                    raise RuntimeError(f'Missing/cross-person fact: {key} {fid}')
                if f['fact_type'] in allowed and float(score) >= float(r['threshold']):
                    pick=fid
                    break
        r['proposed_selected_fact_id']=pick
        r['proposed_intent_reason']=rule
        if pick != r['selected_fact_id']:
            changed.append({'split':r['split'],'query_id':r['query_id'],'question':r['question'],
                            'expected_outcome':r['expected_outcome'],
                            'before':r['selected_fact_id'] or '', 'after':pick or '',
                            'new_intent':rule})
        output.append(r)
    summary={}
    for split in ('validation','test'):
        part=[r for r in output if r['split']==split]
        summary[split]={'before':metrics(part,'selected_fact_id'),
                        'after':metrics(part,'proposed_selected_fact_id'),
                        'changed_predictions':sum(r['selected_fact_id'] != r['proposed_selected_fact_id'] for r in part)}
    report={'status':'PROPOSAL ONLY; old synthetic test already inspected; not a held-out evaluation',
            'frozen_threshold_unchanged':True,'production_bridge_unchanged':True,
            'summary':summary,'changed_rows':changed}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with ROWS.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','query_id','question','expected_outcome','before','after','new_intent'])
        w.writeheader();w.writerows(changed)
    print('STEP 25 — PROPOSED INTENT RULES (OFFLINE / EXPLORATORY)')
    for split,item in summary.items():
        print(split.upper(),'BEFORE:',item['before'],'AFTER:',item['after'],'CHANGED:',item['changed_predictions'])
    print('Saved:', OUT.relative_to(ROOT),'and',ROWS.relative_to(ROOT))
    print('NO production rules changed; no independent accuracy claim.')
if __name__=='__main__': main()
