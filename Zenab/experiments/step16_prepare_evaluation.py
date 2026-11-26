"""WANAS Step 16 — freeze an exploratory E5 baseline and prepare blind review.
Place in experiments/ and run: python experiments/step16_prepare_evaluation.py
No model loads, downloads, or modification of existing datasets/results.
"""
from __future__ import annotations
import csv, hashlib, json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'benchmark'; RES=ROOT/'experiments'/'results'; HAND=ROOT/'handoff'
OUT=ROOT/'experiments'/'step16'
OUT.mkdir(parents=True,exist_ok=True)

def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(p, rows, columns):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',str(s).lower()).strip().replace('’',"'")

def main():
    already=[OUT/n for n in ('frozen_e5_config_step16.json','blind_question_authoring_template.csv','toqa_handoff_draft.csv')]
    if any(x.exists() for x in already):
        print('STEP 16 already prepared. No files overwritten:')
        for x in already:
            if x.exists(): print(' -',x.relative_to(ROOT))
        print('To restart intentionally, manually archive the entire experiments/step16 folder first.')
        return
    fact_path=BENCH/'fact_manifest.csv';question_path=BENCH/'question_manifest.csv'
    report_path=RES/'multilingual_e5_small_report.json'
    for p in [fact_path,question_path,report_path]:
        if not p.exists():raise SystemExit(f'Missing {p}; verify the extracted project folder.')
    facts=read_csv(fact_path);qs=read_csv(question_path)
    by_pid=defaultdict(list); by_id={}
    for f in facts:
        by_pid[f['person_id']].append(f);by_id[f['fact_id']]=f
    report=json.loads(report_path.read_text(encoding='utf-8'))
    threshold=float(report['validation_tuned_threshold'])
    if not 0<=threshold<=1:raise SystemExit('Invalid E5 threshold')
    config={
       'configuration_id':'zinab-e5-question-only-typeguard-exploratory-v1',
       'status':'PROVISIONAL_FROZEN_FOR_NEW_BLIND_EVALUATION_NOT_PRODUCTION_APPROVED',
       'model':'intfloat/multilingual-e5-small',
       'e5_query_prefix':'query: ', 'e5_passage_prefix':'passage: ',
       'query_variant':'question_only',
       'last_request':'not_appended_to_query',
       'identity':'filter_candidate_facts_by_recognized_person_id_before_ranking',
       'verification_filter':'synthetic_test_data + verified_in_fictional_scenario + draft_for_team_review (DEMO ONLY)',
       'production_mode':'requires independent caregiver approval; synthetic facts always blocked',
       'ranking':'cosine similarity of normalized E5 embeddings',
       'acceptance_threshold':threshold,
       'acceptance_rule':'highest ranked candidate with allowed question-intent fact_type and its own score >= frozen threshold; else abstain',
       'guard_source':'experiments/fact_type_guard_step15.py',
       'selection_basis':'prior exploratory validation; prior test repeatedly inspected',
       'known_limits':['Rule-based English-heavy intent detection','Context/last-request experimental only',
                      'No real face/STT upstream predictions','No untouched final evaluation yet'],
       'checksums_sha256':{},
       'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    for label,p in [('fact_manifest',fact_path),('question_manifest',question_path),('e5_report',report_path),
                    ('guard_script',ROOT/'experiments'/'fact_type_guard_step15.py'),
                    ('context_script',ROOT/'experiments'/'context_evaluation_step13.py')]:
        if p.exists():config['checksums_sha256'][label]=digest(p)
        else:config['checksums_sha256'][label]=None
    cpath=OUT/'frozen_e5_config_step16.json';cpath.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')

    # Authoring slots, NOT actual new test questions. Leave question and labels BLANK.
    types=['identity_relationship','personal_interest','location_context','shared_memory','practical_support','conversation_starter']
    rows=[]
    people=sorted(by_pid)
    for i,pid in enumerate(people):
        eligible=[f for f in by_pid[pid] if f['fact_type']==types[i%len(types)]]
        chosen=eligible[0] if eligible else by_pid[pid][0]
        rows.append({'new_query_id':f'BLIND16-{i+1:03d}','person_id':pid,'authoring_prompt':f'Write an ORIGINAL answerable question about {chosen["fact_type"]}; avoid copying previous questions',
                     'question':'','expected_outcome':'','acceptable_fact_ids':'','original_question_id':'',
                     'context_scenario':'question_only','fact_source_id_for_author':chosen['fact_id'],
                     'independent_author':'','independent_reviewer':'','review_approval':'','origin_id':'step16_independent_authoring_pending','notes':''})
    for i,pid in enumerate(people):
        rows.append({'new_query_id':f'BLIND16-{len(people)+i+1:03d}','person_id':pid,
                     'authoring_prompt':'Write an ORIGINAL unanswerable question; check no demo fact supports it, vary wording and missing-info category',
                     'question':'','expected_outcome':'','acceptable_fact_ids':'','original_question_id':'',
                     'context_scenario':'question_only','fact_source_id_for_author':'',
                     'independent_author':'','independent_reviewer':'','review_approval':'','origin_id':'step16_independent_authoring_pending','notes':''})
    columns=list(rows[0]); template=OUT/'blind_question_authoring_template.csv';write_csv(template,rows,columns)

    # A real source-grounded Toqa handoff draft, with explicit synthetic/demo semantics.
    samples=[]
    known=by_pid[people[0]][0]
    qok=next((q for q in qs if q['person_id']==known['person_id'] and q['expected_outcome']=='answerable'
              and known['fact_id'] in json.loads(q['acceptable_fact_ids'])),None)
    quna=next((q for q in qs if q['person_id']==known['person_id'] and q['expected_outcome']=='unanswerable'),None)
    if qok:
        samples.append({'example_type':'answerable_synthetic_demo','query_id':qok['query_id'],'recognized_person_id':qok['person_id'],
                        'question':qok['question'],'expected_fact_id':known['fact_id'],'expected_spoken_text':known['memory_fact'],
                        'abstention_reason':'','source_origin':known['origin_id'],'verification_status':known['verification_status'],
                        'upstream_origin':'oracle_synthetic','reference_transcript':'','actual_stt_transcript':'','handoff_status':'DRAFT_NOT_SENT'})
    if quna:
        samples.append({'example_type':'missing_fact','query_id':quna['query_id'],'recognized_person_id':quna['person_id'],
                        'question':quna['question'],'expected_fact_id':'','expected_spoken_text':'No verified information available for this request.',
                        'abstention_reason':'unanswerable','source_origin':quna['origin_id'],'verification_status':'not_applicable',
                        'upstream_origin':'oracle_synthetic','reference_transcript':'','actual_stt_transcript':'','handoff_status':'DRAFT_NOT_SENT'})
    samples.append({'example_type':'unknown_person','query_id':'STEP16-UNKNOWN-FACE','recognized_person_id':'unknown',
                    'question':'Who is this person?','expected_fact_id':'','expected_spoken_text':'I cannot identify this person reliably.',
                    'abstention_reason':'unknown_face','source_origin':'step16_synthetic_safety_example','verification_status':'not_applicable',
                    'upstream_origin':'simulated_unknown_face','reference_transcript':'','actual_stt_transcript':'','handoff_status':'DRAFT_NOT_SENT'})
    hp=OUT/'toqa_handoff_draft.csv';write_csv(hp,samples,list(samples[0]))
    np=OUT/'prior_question_norms.json';np.write_text(json.dumps(sorted({norm(q['question']) for q in qs}),ensure_ascii=False,indent=2),encoding='utf-8')
    print('STEP 16 — PREPARATION ONLY')
    print('Facts:',len(facts),'Persons:',len(people),'Existing questions:',len(qs))
    print('Frozen E5 threshold:',threshold)
    print('Blind authoring slots:',len(rows),'(UNFILLED — not evaluated)')
    print('Toqa handoff examples:',len(samples),'(DRAFT — not sent)')
    print('Wrote:',cpath.relative_to(ROOT),template.relative_to(ROOT),hp.relative_to(ROOT),np.relative_to(ROOT),sep='\n - ')
    print('No independent held-out score claimed; no upstream outputs used.')
if __name__=='__main__':main()
