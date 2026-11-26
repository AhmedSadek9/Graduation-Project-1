"""WANAS Step 24 — read-only review of E5 intent coverage.

Run: python experiments/e5_intent_review_step24.py
Requires Step 22 predictions. Never edits model, guard, facts, or threshold.
Reports old synthetic validation/test only; not independent evaluation.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments'/'results'
SOURCE=RESULTS/'e5_adapter_step22_predictions.jsonl'
CSV=RESULTS/'e5_intent_review_step24.csv'
REPORT=RESULTS/'e5_intent_review_step24.json'


def intent_hint(question):
    """Human-review hints only, NOT an automated guard update."""
    q=' '.join(question.casefold().split())
    if 'phone' in q or 'telephone' in q or 'mobile' in q:
        return 'unsupported_contact_information', 'BLOCK; no personal contact disclosure'
    if any(x in q for x in ('who is ', 'who are ', "who's ")):
        return 'identity_relationship', 'REVIEW whether person reference is explicit'
    if any(x in q for x in ('talk to', 'talk about', 'conversation')):
        return 'conversation_starter', 'REVIEW conversational intent'
    if any(x in q for x in ('together','family gathering','trip','picnic','visit together','shared memory','what memory')):
        return 'shared_memory', 'REVIEW shared event vs location'
    if any(x in q for x in ('where does','where did','which city','live in')):
        return 'location_context', 'REVIEW place vs shared event ambiguity'
    if any(x in q for x in ('help','bring','organize','send','exchange')):
        return 'practical_support', 'REVIEW support vs shared event ambiguity'
    if any(x in q for x in ('collection','music','song','favorite','hobby')):
        return 'personal_interest', 'REVIEW personal interest'
    return 'unknown', 'Manual review required; do not auto-accept'


def main():
    if not SOURCE.exists():
        raise SystemExit('Missing Step 22 predictions. Run Step 22 first.')
    rows=[json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len({(r['split'],r['query_id']) for r in rows})!=len(rows):
        raise SystemExit('Duplicate split/query_id: refusing audit.')
    focus=[r for r in rows if r['split'] in ('validation','test') and r.get('abstention_reason') in ('unrecognized_question_intent','no_matching_verified_fact_type')]
    export=[]
    for r in focus:
        hint,note=intent_hint(r['question'])
        export.append({'split':r['split'],'query_id':r['query_id'],'person_id':r['person_id'],
                       'question':r['question'],'expected_outcome':'answerable' if r['expected_outcome']=='answerable' else 'unanswerable',
                       'old_reason':r['abstention_reason'],'review_hint_only':hint,'manual_review_note':note,
                       'current_selected_fact_id':r.get('selected_fact_id'),
                       'top_ranked_fact_id':r.get('raw_rank1_fact_id'),
                       'top_type_candidate_id':r.get('top_type_candidate_id')})
    if export:
        with CSV.open('w',encoding='utf-8-sig',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=export[0].keys());wr.writeheader();wr.writerows(export)
    else:
        CSV.write_text('No flagged rows\n',encoding='utf-8')
    report={'scope':'EXPLORATORY, previously inspected synthetic benchmark',
            'change_to_production_or_frozen_guard':'NONE',
            'caution':'Review hints are not rules; expected labels used only to summarize audited outcomes',
            'flagged_by_split':dict(Counter(r['split'] for r in export)),
            'flagged_by_reason':dict(Counter(r['old_reason'] for r in export)),
            'flagged_answerable_by_split':dict(Counter(r['split'] for r in export if r['expected_outcome']=='answerable')),
            'manual_review_rows':export}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STEP 24 — INTENT REVIEW (READ ONLY)')
    print('Flagged rows:',len(export),'by split:',report['flagged_by_split'])
    print('Answerable flagged:',report['flagged_answerable_by_split'])
    for r in export:
        print(f" {r['split']:10} {r['query_id']:18} {r['old_reason']:31} | {r['question']}")
        print(f"    REVIEW HINT: {r['review_hint_only']} — {r['manual_review_note']}")
    print('Saved:',CSV.relative_to(ROOT),'and',REPORT.relative_to(ROOT))
    print('NO runtime rules or threshold changed. NOT new held-out evidence.')

if __name__=='__main__':main()
