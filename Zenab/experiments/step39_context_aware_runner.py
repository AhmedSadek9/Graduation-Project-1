"""Step 39: separate experimental context-aware paired evaluator for fictional WANAS data.

Default: preflight ONLY (no model loading). Run from Zenab root.
Execute ONLY after independent session freshness policy, Ahmed score semantics,
Hanaa ID mapping, and actual case provenance are reviewed by team.
Does not modify Step32, Step35, frozen threshold, or any benchmark.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
from step32_run_paired_evaluation import (rows_from, s, prepare, oracle_face, actual_face,
                                          fact_ids, categorize, summarize, actual_face_ready,
                                          actual_stt_ready)
from step35_context_adapter import SafeContextAdapter, is_pronoun_question

BASE = ROOT / 'handoff' / 'step29'


def parse_context(raw: Any):
    """Strictly parse JSON dict; never interpret arbitrary text as prior identity.

    `fresh` is only a declarative input here, not proof of freshness. The
    caller/session manager must derive it from trusted timestamps before live use.
    """
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if set(('referent_person_id', 'session_id', 'fresh')) - set(obj):
        return None
    if not isinstance(obj['referent_person_id'], str) or not obj['referent_person_id'].strip():
        return None
    if not isinstance(obj['session_id'], str) or not obj['session_id'].strip():
        return None
    if type(obj['fresh']) is not bool:
        return None
    return obj


def load_profile_names(db_path: Path):
    data = json.loads(db_path.read_text(encoding='utf-8'))
    people = data['persons']
    mapping = {}
    for person in people:
        pid = person['person_id']
        if pid in mapping:
            raise ValueError('Duplicate person ID in fictional memory database')
        mapping[pid] = person.get('name')
    return mapping


def evaluate_context(valid, bridge, name_lookup, min_face_score=0.75):
    """Evaluate ready A/B/C/D conditions; support pronouns via safe context adapter.

    The returned outcome is Step32's *provisional* categorization. Fact-owner
    verification with Step37/38 must be applied before final metric reporting.
    """
    adapter = SafeContextAdapter(bridge, name_lookup, min_face_score=min_face_score)
    results = []
    for case, readiness in valid:
        context = parse_context(s(case, 'conversation_context'))
        session = s(case, 'session_id') or None
        for run in 'ABCD':
            if not readiness.get(run, False):
                continue
            face = oracle_face(case) if run in 'AC' else actual_face(case)
            question = s(case, 'reference_transcript') if run in 'AB' else s(case, 'actual_stt_transcript')
            response = adapter.process(face, question, context=context, session_id=session)
            outcome = categorize(case, response)
            results.append({
                'case_id': s(case, 'case_id'), 'run':run, 'split':s(case, 'split'),
                'image_id':s(case,'image_id'),'question_id':s(case,'question_id'),
                'utterance_id':s(case,'utterance_id'),'session_id':session,
                'expected_person_id':s(case,'expected_person_id'),
                'face_decision_used':face.get('status'),
                'face_person_id_used':face.get('person_id'),
                'face_score_used':face.get('recognition_confidence'),
                'reference_transcript':s(case,'reference_transcript'),
                'actual_stt_transcript':s(case,'actual_stt_transcript'),
                'question_used':question,
                'pronoun_question':is_pronoun_question(question),
                'context_parsed':context is not None,
                'context_resolution':response.get('context_resolution','not_applicable'),
                'resolved_question':response.get('resolved_question'),
                'acceptable_fact_ids':fact_ids(s(case,'acceptable_fact_ids')),
                'retrieved_fact_id':response.get('fact_id'),
                'retrieved_person_id':response.get('person_id'),
                'response_status':response.get('status'),
                'spoken_text':response.get('spoken_text'),
                'abstention_reason':response.get('abstention_reason'),
                'can_speak_personal_memory':response.get('can_speak_personal_memory'),
                'outcome_category_provisional':outcome,
                'outcome_category':outcome,
                'context_provenance_note':'fresh flag not independently verified by this runner',
            })
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Run actual eligible validation rows with E5 demo')
    p.add_argument('--face-score-contract-confirmed', action='store_true')
    p.add_argument('--session-freshness-contract-confirmed', action='store_true')
    args = p.parse_args(argv)
    manifest = BASE/'CASE_MANIFEST_WORKING.csv'
    poolfile = BASE/'STEP29_QUESTION_POOL.csv'
    if not manifest.exists() or not poolfile.exists():
        print('ERROR missing Step29 manifest/question pool');return 2
    cols, cases = rows_from(manifest)
    _, questions = rows_from(poolfile)
    pool = {s(q,'question_id'):q for q in questions}
    valid, issues, counts = prepare(cases,pool)
    required = {'session_id','conversation_context','case_review_status'}
    missing = required - cols
    # Step32 prepare checks standard question/pairing validity; Step39 additionally
    # requires explicit case review before any model run.
    blocked_reviews = [s(c,'case_id') for c,_ in valid if s(c,'case_review_status').lower() not in ('approved','reviewed')]
    pronoun_rows = sum(is_pronoun_question(s(c,'reference_transcript')) for c,_ in valid)
    print('STEP 39 — CONTEXT-AWARE PAIRED EVALUATION')
    print('Linked rows:',len(cases),'Valid:',len(valid),'Issues:',len(issues),
          'Run-ready:',dict(counts),'Pronoun questions:',pronoun_rows)
    if missing or issues:
        print('BLOCKED schema/validation issues:',sorted(missing),issues[:2]);return 2
    if not args.execute:
        print('PREFLIGHT ONLY — NO E5 LOADED / NO PREDICTIONS GENERATED')
        print('Context must be a JSON object with referent_person_id, session_id, fresh.')
        print('Current ready cases:', dict(counts), '| Unreviewed pairings:', len(blocked_reviews))
        return 0
    if not args.face_score_contract_confirmed or not args.session_freshness_contract_confirmed:
        print('BLOCKED: Need both Ahmed score contract and trusted session freshness contract');return 2
    if blocked_reviews:
        print('BLOCKED: Unreviewed case pairings:',blocked_reviews[:5]);return 2
    if not valid:
        print('NO ACTUAL LINKED CASES — no model run');return 0
    # The team must separately inspect source authenticity and the image/ID mapping;
    # a CSV value declaring actual is not independently verified by this tool.
    from e5_demo_bridge_step21 import E5DemoBridge, THRESHOLD
    names = load_profile_names(ROOT/'Task1'/'memory_database.json')
    bridge = E5DemoBridge(data_mode='demo',threshold=THRESHOLD)
    rows = evaluate_context(valid,bridge,names.get)
    output = EXPERIMENTS/'results';output.mkdir(parents=True,exist_ok=True)
    predictions = output/'step39_context_validation_predictions.jsonl'
    with predictions.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report = summarize(rows)
    report.update({'step':39,'data':'fictional_demo_facts',
                   'status':'EXPLORATORY_VALIDATION_ONLY_NOT_HELDOUT',
                   'notes':['Step35 pronoun support is narrow (tell me about him/her; who is him/her).',
                            'Context fresh boolean must be produced by a trusted session manager.',
                            'Step32 outcome labels are provisional; verify ownership with Step37/38.',
                            'Actual source declaration is not verified cryptographically.',
                            'Do not infer real Face/STT performance unless upstream rows were supplied.']})
    summary=output/'step39_context_validation_summary.json'
    summary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved rows:',len(rows),'at',predictions)
    print('Paired reliability:',report['paired_A_D'])
    return 0

if __name__=='__main__':
    sys.exit(main())
