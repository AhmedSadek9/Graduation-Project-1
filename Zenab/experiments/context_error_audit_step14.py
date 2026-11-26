"""WANAS Step 14: inspect context prediction errors without re-running models.
Place in experiments/ and run: python experiments/context_error_audit_step14.py
This is a diagnostic, not a new model/threshold evaluation.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'experiments' / 'results' / 'context_step13_predictions.jsonl'
FACTS = ROOT / 'benchmark' / 'fact_manifest.csv'
OUT = ROOT / 'experiments' / 'results' / 'context_step14_error_audit.json'


def main():
    if not PRED.exists():
        raise SystemExit(f'Missing: {PRED}\nRun Step 13 first.')
    with FACTS.open('r', encoding='utf-8-sig', newline='') as f:
        facts = {r['fact_id']: r for r in csv.DictReader(f)}
    with PRED.open('r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keycounts = Counter((r['split'], r['scenario'], r['query_id']) for r in rows)
    duplicates = {str(k): v for k,v in keycounts.items() if v > 1}
    print('STEP 14 — CONTEXT ERROR AUDIT (no model downloads)')
    print('Prediction rows:', len(rows), 'unique split/scenario/question:', len(keycounts))
    print('Duplicate records within the SAME scenario:', len(duplicates))
    if duplicates:
        print('WARNING: actual duplicates found:', list(duplicates.items())[:5])
    result = {'total_prediction_rows': len(rows), 'unique_scenario_question_keys': len(keycounts),
              'duplicate_scenario_question_keys': duplicates, 'scenarios': {}}
    for scenario in sorted({r['scenario'] for r in rows}):
        result['scenarios'][scenario] = {}
        for split in ['validation', 'test']:
            subset = [r for r in rows if r['scenario'] == scenario and r['split'] == split]
            issues=[]
            for r in subset:
                selected = r.get('selected_fact_id')
                if not selected:
                    continue
                acceptable = r.get('acceptable_fact_ids') or []
                if selected in acceptable:
                    continue
                kind = 'unsupported_answer' if r['expected_outcome'] != 'answerable' else 'wrong_fact'
                top = facts.get(selected, {})
                issues.append({'kind': kind, 'query_id': r['query_id'], 'person_id': r['person_id'],
                    'question': r['question'], 'selected_fact_id': selected,
                    'selected_fact_type': top.get('fact_type'), 'selected_fact_text': top.get('memory_fact'),
                    'acceptable_fact_ids': acceptable, 'top_score': (r.get('scores') or [None])[0]})
            result['scenarios'][scenario][split]={'n':len(subset),'issues':issues}
            print(f'\n{scenario} / {split}: n={len(subset)} wrong_or_unsupported={len(issues)}')
            for issue in issues:
                print(f"  {issue['kind']} | {issue['query_id']} | selected={issue['selected_fact_id']} "
                      f"({issue['selected_fact_type']}) | expected={issue['acceptable_fact_ids']} | score={issue['top_score']}")
                print('    Q:', issue['question'])
                print('    Returned:', issue['selected_fact_text'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSaved:', OUT.relative_to(ROOT))
    print('Synthetic context / previously inspected test: exploratory results only.')

if __name__ == '__main__':
    main()
