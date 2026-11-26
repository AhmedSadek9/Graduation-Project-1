"""WANAS Step 15: offline, exploratory fact-type safety audit.

Place this script in experiments/ and run from the project root:
  python experiments/fact_type_guard_step15.py

Reads Step 13 ranked predictions and fact manifest; does NOT run/download models,
does NOT edit original predictions or threshold. Rule matches use question text only,
never expected labels, fact IDs, or splits. Synthetic data only.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PREDICTIONS = RESULTS / "context_step13_predictions.jsonl"
FACTS = ROOT / "benchmark" / "fact_manifest.csv"
OUTPUT = RESULTS / "fact_type_guard_step15_report.json"
OUTPUT_ROWS = RESULTS / "fact_type_guard_step15_predictions.jsonl"

# Conservative query-intent map. These are NOT learned from evaluation labels.
# Unrecognized/ambiguous queries are left unchanged; an explicit guard is not a
# claim that arbitrary natural language is accurately classified.
def allowed_fact_types(question: str) -> tuple[set[str] | None, str]:
    q = re.sub(r"\s+", " ", str(question).lower()).strip()
    if not q:
        return set(), "empty_question"
    if re.search(r"\b(phone|telephone|mobile|cell|contact)\s*(number|no\.?|details)?\b", q) or "رقم التليفون" in q or "رقم الهاتف" in q:
        return set(), "unsupported_contact_information"
    if re.search(r"\b(who is|who's|who are|relationship to|related to)\b", q):
        return {"identity_relationship"}, "identity_intent"
    if re.search(r"\b(talk to|talk about|conversation|what should i ask|what can i ask)\b", q):
        return {"conversation_starter"}, "conversation_intent"
    if re.search(r"\b(shared memory|what memory|remember doing|together|first meet)\b", q) or re.search(r"\b(what (meal|game|event|dish) did)\b", q):
        return {"shared_memory"}, "shared_memory_intent"
    if re.search(r"\b(where (does|is|did).*\b(live|based|located)|which city|live in|based in)\b", q):
        return {"location_context"}, "location_intent"
    if re.search(r"\b(favorite|likes|interest|hobby|enjoy|music|song)\b", q):
        return {"personal_interest"}, "interest_intent"
    if re.search(r"\b(helps?|support|organize|bring|share with|send|drop by|visit us)\b", q):
        return {"practical_support"}, "support_intent"
    return None, "no_guard_rule"


def summarize(rows: list[dict], field: str) -> dict:
    answered = [r for r in rows if r.get(field)]
    answerable = [r for r in rows if r['expected_outcome'] == 'answerable']
    unanswerable = [r for r in rows if r['expected_outcome'] != 'answerable']
    correct = sum(r[field] in r['acceptable_fact_ids'] for r in answerable if r.get(field))
    wrong = sum(r[field] not in r['acceptable_fact_ids'] for r in answerable if r.get(field))
    unsupported = sum(bool(r.get(field)) for r in unanswerable)
    return {
        'n': len(rows), 'answerable_n': len(answerable), 'unanswerable_n': len(unanswerable),
        'accepted_n': len(answered), 'accepted_correct_n': correct,
        'accepted_wrong_fact_n': wrong, 'unsupported_answer_n': unsupported,
        'correct_abstention_n': len(unanswerable) - unsupported,
        'missed_answerable_n': len(answerable) - correct - wrong,
        'precision_of_accepted': round(correct / len(answered), 4) if answered else None,
        'correct_answer_rate_answerable': round(correct / len(answerable), 4) if answerable else None,
    }


def main() -> None:
    for path in (PREDICTIONS, FACTS):
        if not path.exists():
            raise SystemExit(f'Missing: {path}\nRun Step 13 or check benchmark paths.')
    with FACTS.open(encoding='utf-8-sig', newline='') as fp:
        facts = {r['fact_id']: r for r in csv.DictReader(fp)}
    with PREDICTIONS.open(encoding='utf-8') as fp:
        source = [json.loads(line) for line in fp if line.strip()]
    if not source:
        raise SystemExit('No prediction rows to evaluate.')
    seen = set()
    output = []
    for original in source:
        r = dict(original)
        key = (r['split'], r['scenario'], r['query_id'])
        if key in seen:
            raise SystemExit(f'Duplicate prediction key: {key}')
        seen.add(key)
        allowed, reason = allowed_fact_types(r['question'])
        old = r.get('selected_fact_id')
        ranked = r.get('ranked_fact_ids') or []
        scores = r.get('scores') or []
        threshold = float(r['threshold'])
        if len(scores) != len(ranked):
            raise SystemExit(f'Rank/score mismatch: {key}')
        for fid in ranked:
            if fid not in facts:
                raise SystemExit(f'Unknown fact ID {fid} for {key}')
            if facts[fid]['person_id'] != r['person_id']:
                raise SystemExit(f'Cross-person candidate in {key}: {fid}')
        new = None
        # Enforce face unknown even if a malformed prediction contained ranked facts.
        if r.get('face_status') == 'unknown':
            guard_reason = 'unknown_face'
        elif allowed == set():
            guard_reason = reason
        else:
            # Select the highest ranked fact of the permitted type whose OWN score
            # exceeds the pre-existing threshold. Never reuse top-1 score for a
            # lower candidate. No labels/expected IDs used to choose a fact.
            for fid, score in zip(ranked, scores):
                if (allowed is None or facts[fid]['fact_type'] in allowed) and float(score) >= threshold:
                    new = fid
                    break
            guard_reason = None if new else ('no_type_compatible_fact_above_threshold' if allowed is not None else 'below_threshold')
        r['step15_original_selected_fact_id'] = old
        r['step15_selected_fact_id'] = new
        r['step15_guard_reason'] = guard_reason
        r['step15_intent_rule'] = reason
        r['step15_allowed_fact_types'] = sorted(allowed) if allowed is not None else None
        r['step15_change'] = 'unchanged' if old == new else ('blocked' if not new else 'reselected')
        output.append(r)

    scenarios = sorted({r['scenario'] for r in output})
    summary = {}
    print('STEP 15 — FACT TYPE GUARD (OFFLINE, SYNTHETIC, EXPLORATORY)')
    print('Prediction rows:', len(output), 'unique:', len(seen))
    for scenario in scenarios:
        summary[scenario] = {}
        for split in ('validation', 'test'):
            subset = [r for r in output if r['scenario'] == scenario and r['split'] == split]
            if not subset: continue
            before = summarize(subset, 'step15_original_selected_fact_id')
            after = summarize(subset, 'step15_selected_fact_id')
            changes = dict(Counter(r['step15_change'] for r in subset))
            summary[scenario][split] = {'before':before, 'after':after, 'changes':changes}
            print(f'\n{scenario} / {split} n={len(subset)}')
            print(' BEF: correct=', before['accepted_correct_n'], 'wrong=', before['accepted_wrong_fact_n'], 'unsupported=', before['unsupported_answer_n'], 'abstentions=', before['correct_abstention_n'])
            print(' AFT: correct=', after['accepted_correct_n'], 'wrong=', after['accepted_wrong_fact_n'], 'unsupported=', after['unsupported_answer_n'], 'abstentions=', after['correct_abstention_n'])
            print(' Changes:', changes)
            for r in subset:
                if r['step15_change'] != 'unchanged':
                    print(' ', r['query_id'], r['step15_change'], str(r['step15_original_selected_fact_id']), '->', str(r['step15_selected_fact_id']), '['+r['step15_intent_rule']+']')
    RESULTS.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps({'evaluation_scope':'synthetic / previously inspected test; exploratory only',
                                  'method':'fact-type guard after E5 ranking; frozen per-row threshold',
                                  'scenario_results':summary}, indent=2),encoding='utf-8')
    OUTPUT_ROWS.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    print('\nSaved:', OUTPUT.relative_to(ROOT), 'and', OUTPUT_ROWS.relative_to(ROOT))
    print('NOTE: fact-type rules do not replace evidence verification or independent held-out testing.')

if __name__=='__main__':
    main()
