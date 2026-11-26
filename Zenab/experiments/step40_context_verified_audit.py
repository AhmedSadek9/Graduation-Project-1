"""Step 40: independent post-run audit for Step39 context-aware predictions.

Read only Step39 predictions + case manifest + frozen fact manifest. Never executes
E5 or changes Step32/39. Does NOT certify semantic faithfulness of spoken_text.
Run from Zenab root: python experiments/step40_context_verified_audit.py
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step37_fact_ownership_audit import load_fact_index
from step38_verified_scoring import verify_prediction_rows
from step32_run_paired_evaluation import summarize

CASE_FILE = ROOT / 'handoff' / 'step29' / 'CASE_MANIFEST_WORKING.csv'
PRED_FILE = ROOT / 'experiments' / 'results' / 'step39_context_validation_predictions.jsonl'
OUT_DIR = ROOT / 'experiments' / 'results'


def audit_rows(cases, predictions, index):
    """Use Step38 manifest-backed scoring and flag unsafe/inconclusive text.

    A correct fact ID is NOT proof the spoken sentence states that fact accurately.
    Every non-empty answer text requires human check before final outcome rates.
    """
    verified = verify_prediction_rows(cases, predictions, index)
    out = []
    for row in verified:
        category = row['step38_manifest_verified_outcome']
        speech = str(row.get('spoken_text') or '').strip()
        allowed = row.get('can_speak_personal_memory') is True
        fact_id = str(row.get('retrieved_fact_id') or '').strip()
        status = str(row.get('response_status') or '').strip()
        red_flags = []
        if allowed and not fact_id:
            red_flags.append('permission_true_without_fact_id')
        if allowed and category != 'correct_supported_answer':
            red_flags.append('permission_true_without_verified_correct_fact')
        if status == 'answerable' and not fact_id:
            red_flags.append('answerable_without_fact_id')
        if category != 'correct_supported_answer' and allowed:
            red_flags.append('personal_disclosure_not_supported')
        # Text can be a generic refusal; do not assert it is a personal fact.
        needs_review = bool(speech and (fact_id or allowed or status == 'answerable'))
        out.append({**row,
            'step40_verified_outcome': category,
            'step40_context_provenance_verified': False,
            'step40_spoken_text_human_review_required': needs_review,
            'step40_spoken_text_review_status': 'pending' if needs_review else 'not_applicable',
            'step40_safety_flags': red_flags,
            'step40_eligible_for_final_correct_outcome_claim': False,
        })
    return out


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    if not CASE_FILE.is_file():
        print('BLOCKED: Missing case manifest; Step29 required.'); return 2
    cases = read_csv(CASE_FILE)
    idx = load_fact_index()
    if not PRED_FILE.is_file():
        print('STEP 40 — PREPARATION ONLY (READ-ONLY)')
        print(f'Fact index: {len(idx)} | Linked cases: {len(cases)} | Step39 predictions: NOT PRESENT')
        print('No output scores, no E5 run, no model download.');return 0
    with PRED_FILE.open('r', encoding='utf-8-sig') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    if not predictions:
        print('STEP 40 — No predictions, no scores computed.'); return 0
    audited = audit_rows(cases, predictions, idx)
    counts = Counter(r['step40_verified_outcome'] for r in audited)
    pending = sum(r['step40_spoken_text_human_review_required'] for r in audited)
    flags = sum(bool(r['step40_safety_flags']) for r in audited)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    pred_output = OUT_DIR / 'step40_verified_context_validation_predictions.jsonl'
    with pred_output.open('w', encoding='utf-8') as f:
        for row in audited: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Provisional metrics only; do not claim real reliability or confirmed text faithfulness.
    provisional = [{**r, 'outcome_category':r['step40_verified_outcome']} for r in audited]
    summary = summarize(provisional)
    summary.update({'step':40, 'status':'EXPLORATORY_VALIDATION_UNREVIEWED_TEXT',
                    'prediction_rows':len(audited), 'manifest_verified_categories':dict(counts),
                    'spoken_review_pending_n':pending, 'safety_flagged_rows_n':flags,
                    'metrics_are_provisional':True,
                    'limitations':['Spoken content not independently verified',
                                   'Context freshness not established by trusted session manager',
                                   'Actual upstream origin not verified by this audit',
                                   'Not held-out evaluation']})
    report = OUT_DIR / 'step40_verified_context_validation_report.json'
    report.write_text(json.dumps(summary, ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP 40 — MANIFEST-BACKED CONTEXT AUDIT')
    print('Rows:',len(audited),'| spoken review pending:',pending,'| safety flagged:',flags)
    print('Provisional outcomes:',dict(counts))
    print('Saved:',pred_output.relative_to(ROOT),'and',report.relative_to(ROOT))
    print('NOT FINAL: no verified spoken-text faithfulness, session origin, or held-out scores.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
