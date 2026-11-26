"""Validate new blind authoring questions before ANY final evaluation.
Run: python experiments/step16_validate_blind_questions.py
Does NOT grade a retrieval model or modify submitted questions.
"""
import csv,json,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'experiments'/'step16'/'blind_question_authoring_template.csv'
Q=ROOT/'benchmark'/'question_manifest.csv'; F=ROOT/'benchmark'/'fact_manifest.csv'
def load(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def norm(s):return re.sub(r'\s+',' ',str(s).lower()).strip().replace('’',"'")
def main():
    a=load(T); old=load(Q); facts=load(F); byid={f['fact_id']:f for f in facts}
    prior={norm(q['question']) for q in old}; seen=set(); errors=[]; completed=0; people=set(); types=set()
    for r in a:
        q=r['question'].strip()
        if not q:continue
        completed+=1;pid=r['person_id']; people.add(pid)
        if norm(q) in prior:errors.append(f'{r["new_query_id"]}: reuses exact existing question')
        if norm(q) in seen:errors.append(f'{r["new_query_id"]}: duplicate new question')
        seen.add(norm(q))
        if r['review_approval'].strip().lower()!='approved':errors.append(f'{r["new_query_id"]}: independent review not approved')
        if not r['independent_author'].strip() or not r['independent_reviewer'].strip():errors.append(f'{r["new_query_id"]}: missing author/reviewer')
        if r['independent_author'].strip()==r['independent_reviewer'].strip():errors.append(f'{r["new_query_id"]}: author must not review own question')
        label=r['expected_outcome'].strip()
        if label not in ('answerable','unanswerable'):errors.append(f'{r["new_query_id"]}: invalid expected_outcome');continue
        types.add(label)
        try: ids=json.loads(r['acceptable_fact_ids'])
        except Exception:errors.append(f'{r["new_query_id"]}: acceptable_fact_ids must be JSON list');continue
        if not isinstance(ids,list):errors.append(f'{r["new_query_id"]}: acceptable_fact_ids not a list');continue
        if label=='answerable' and not ids:errors.append(f'{r["new_query_id"]}: answerable requires acceptable fact')
        if label=='unanswerable' and ids:errors.append(f'{r["new_query_id"]}: unanswerable must have []')
        for fid in ids:
            if fid not in byid:errors.append(f'{r["new_query_id"]}: unknown fact {fid}')
            elif byid[fid]['person_id']!=pid:errors.append(f'{r["new_query_id"]}: cross-person fact {fid}')
        if not r['original_question_id'].strip():errors.append(f'{r["new_query_id"]}: original_question_id required to group paraphrases')
    print(f'STEP 16 BLIND-SET CHECK: filled={completed}/{len(a)}, people={len(people)}, labels={sorted(types)}')
    if completed!=len(a):errors.append(f'Only {completed}/{len(a)} slots completed')
    if types != {'answerable','unanswerable'}:errors.append('Need answerable and unanswerable questions')
    if errors:
        for e in errors[:60]:print('ERROR:',e)
        print('NOT READY for independent evaluation:',len(errors),'issues')
        sys.exit(1)
    print('STRUCTURAL CHECKS PASSED. Independent authorship must still be verified outside this script.')
if __name__=='__main__':main()
