"""WANAS Step 27: prepare independent-review paperwork; never run retrieval.
Usage: python experiments/step27_prepare_review.py
Writes to experiments/step27/. Does not change the Step16 questions or frozen config.
"""
from __future__ import annotations
import csv, hashlib, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/'experiments/step16/blind_question_authoring_template.csv'
OLD=ROOT/'benchmark/question_manifest.csv'
FACT=ROOT/'benchmark/fact_manifest.csv'
FROZEN=ROOT/'experiments/step16/frozen_e5_config_step16.json'
OUT=ROOT/'experiments/step27'

def load(p):
    with p.open('r', encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(s): return re.sub(r'\s+', ' ', (s or '').casefold().replace('’', "'")).strip(' ?!.')

def main():
    for p in (Q,OLD,FACT,FROZEN):
        if not p.exists(): raise SystemExit('Missing: '+str(p))
    qs=load(Q); old=load(OLD); facts=load(FACT)
    cfg=json.loads(FROZEN.read_text(encoding='utf-8'))
    if cfg.get('model')!='intfloat/multilingual-e5-small' or abs(float(cfg['acceptance_threshold'])-0.8824205)>1e-9:
        raise SystemExit('Unexpected frozen E5 configuration; review manually')
    fact_index={f['fact_id']:f for f in facts}
    existing={normalize(q.get('question')) for q in old}
    seen=set(); errors=[]; warning=[]; review_rows=[]
    for r in qs:
        qid=r.get('new_query_id',''); question=r.get('question','').strip(); pid=r.get('person_id','')
        label=r.get('expected_outcome','').strip(); origin=r.get('origin_id','')
        if not question: errors.append(qid+': blank question')
        if label not in ('answerable','unanswerable'):errors.append(qid+': unexpected label '+label)
        n=normalize(question)
        if n in existing:errors.append(qid+': exact prior normalized duplicate')
        if n in seen:errors.append(qid+': duplicate among new items')
        seen.add(n)
        try: acceptable=json.loads(r.get('acceptable_fact_ids',''))
        except (ValueError,TypeError): acceptable=None; errors.append(qid+': invalid JSON acceptable_fact_ids')
        if not isinstance(acceptable,list): acceptable=[]
        if label=='answerable' and not acceptable:errors.append(qid+': answerable but no acceptable fact')
        if label=='unanswerable' and acceptable:errors.append(qid+': unanswerable with acceptable fact')
        for fid in acceptable:
            if fid not in fact_index:errors.append(qid+': unknown fact '+fid)
            elif fact_index[fid]['person_id']!=pid: errors.append(qid+': cross-person fact '+fid)
        if 'ai_assisted' in origin.lower(): warning.append(qid+': AI-assisted draft; cannot claim independently authored blind evaluation')
        # A named human reviewer is insufficient to prove independent authoring.
        approved=(r.get('review_approval','').strip().lower()=='approved'
             and bool(r.get('independent_author','').strip())
             and bool(r.get('independent_reviewer','').strip())
             and r.get('independent_author','').strip()!=r.get('independent_reviewer','').strip())
        if approved and 'ai_assisted' in origin.lower():
            warning.append(qid+': marked approved but source remains AI-assisted; must not claim true blind holdout')
        review_rows.append({
          'new_query_id':qid,'person_id':pid,'question':question,'expected_outcome':label,
          'acceptable_fact_ids':json.dumps(acceptable,ensure_ascii=False),
          'origin_id':origin,'authoring_status':'AI_ASSISTED_DRAFT_NOT_BLIND' if 'ai_assisted' in origin.lower() else 'CHECK_PROVENANCE',
          'human_check_fact_correctness':'PENDING','human_check_unanswerability':'PENDING',
          'human_check_paraphrase_leakage':'PENDING','human_check_ambiguity':'PENDING',
          'human_check_person_id':'PENDING','reviewer_name':'', 'reviewer_notes':'',
          'review_result':'PENDING',
        })
    if len(qs)!=40:errors.append(f'expected 40 questions; observed {len(qs)}')
    if len(set(r['new_query_id'] for r in qs)) != len(qs):errors.append('duplicate query IDs')
    OUT.mkdir(parents=True,exist_ok=True)
    review=OUT/'STEP27_HANAA_REVIEW_QUEUE.csv'
    with review.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review_rows[0]));w.writeheader();w.writerows(review_rows)
    policy={
      'policy_id':'step27_candidate_e5_080_EXPLORATORY_ONLY',
      'status':'LOCKED_FOR_FUTURE_HUMAN_REVIEW_NOT_DEPLOYABLE',
      'model':'intfloat/multilingual-e5-small',
      'candidate_threshold':0.80,
      'baseline_threshold':float(cfg['acceptance_threshold']),
      'query_variant':'question_only','identity_filter':'enforce recognized person_id before retrieval',
      'fact_type_guard':'existing Step15 rules; Step25 is proposal ONLY',
      'verification':'approved facts only; synthetic blocked in production',
      'safety':'reject unknown/low-confidence/mismatched identity and unsupported information',
      'selection_note':'0.80 described on previously inspected validation; NOT independently validated',
      'do_not_change':['frozen_e5_config_step16.json','production bridge','120 original synthetic facts'],
      'inputs_sha256':{'step16_draft':sha(Q),'old_questions':sha(OLD),'fact_manifest':sha(FACT),'frozen_config':sha(FROZEN)},
    }
    (OUT/'STEP27_EXPERIMENTAL_POLICY.json').write_text(json.dumps(policy,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    ready={
      'step':'27','status':'REVIEW_PREPARATION_ONLY_NOT_READY_FOR_BLIND_TEST',
      'generated_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'question_count':len(qs),'answerable':sum(r.get('expected_outcome')=='answerable' for r in qs),
      'unanswerable':sum(r.get('expected_outcome')=='unanswerable' for r in qs),
      'unique_persons':len({r.get('person_id') for r in qs}),
      'structural_errors':errors,'provenance_warning_count':len(warning),
      'human_review_completed':0,
      'blind_holdout_claim_permitted':False,
      'next_action':'Hanaa: review draft source facts/IDs/ambiguity/leakage. For true independent holdout, separately author fresh unseen questions and have a different reviewer approve before scoring.',
      'files_created':['STEP27_HANAA_REVIEW_QUEUE.csv','STEP27_EXPERIMENTAL_POLICY.json','STEP27_READINESS_REPORT.json'],
    }
    (OUT/'STEP27_READINESS_REPORT.json').write_text(json.dumps(ready,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('STEP 27 — REVIEW READINESS (NO MODEL LOADING, NO EVALUATION)')
    print('Questions:',len(qs),'| Answerable:',ready['answerable'],'| Unanswerable:',ready['unanswerable'],'| Persons:',ready['unique_persons'])
    print('Structural errors:',len(errors),'| AI-assisted origins:',sum('ai_assisted' in r.get('origin_id','').lower() for r in qs))
    print('Reviewer approval: PENDING | True blind held-out evaluation: NOT READY')
    print('Frozen baseline:',policy['baseline_threshold'],'| Exploratory candidate:',policy['candidate_threshold'])
    for f in ready['files_created']: print('Saved:',(OUT/f).relative_to(ROOT))
    if errors:
        for e in errors[:10]:print('STRUCTURAL ERROR:',e)
    print('No frozen threshold, runtime guard, facts, or questions changed.')
if __name__=='__main__':main()
