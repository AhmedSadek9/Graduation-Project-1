"""Investigate a labeled illustrative music query; not claimed to be the original bug's exact query."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from bm25_benchmark import FACTS,candidates,bm25_rank
root=Path(__file__).resolve().parents[1]
conf=json.loads((root/'experiments/results/bm25_frozen_configuration.json').read_text())
q={'person_id':'person-020','question':'Does Nada record music clips for the family?'}
pool=candidates(q); rank=bm25_rank(q['question'],pool)
target='person-020-F05'
row={'query_text':q['question'],'query_origin':'constructed_diagnostic_example_not_original_reported_query','person_id':q['person_id'],'target_fact_id':target,'target_fact_text':next(f['memory_fact'] for f in FACTS if f['fact_id']==target),'filter_survived':any(f['fact_id']==target for f in pool),'candidate_fact_ids':[f['fact_id'] for f in pool],'ranked_fact_ids':[f['fact_id'] for f,s in rank],'ranked_scores':[round(s,6) for f,s in rank],'target_rank':next((i for i,(f,s) in enumerate(rank,1) if f['fact_id']==target),None),'winning_score':rank[0][1] if rank else None,'threshold':conf['acceptance_threshold'],'accepted':bool(rank and rank[0][1]>conf['acceptance_threshold']), 'checkpoint':'filter -> rank -> threshold','original_not_found_bug_repaired':'NOT_VERIFIED_original_query_missing'}
(root/'experiments/results/music_diagnostic.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
print(json.dumps(row,indent=2))
