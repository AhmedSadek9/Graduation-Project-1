from pathlib import Path
import csv
p=Path(__file__).resolve().parent/'TOQA_EXAMPLES_STEP19.csv'
with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==5
assert sum(r['can_speak_personal_memory']=='true' for r in rows)==1
assert all(r['source_origin'].startswith('synthetic') and r['upstream_type']=='mock' for r in rows)
assert {r['example_type'] for r in rows}=={'verified_answer','missing_fact','unresolved_person','person_mismatch','production_blocks_synthetic'}
print('STEP 19: 5/5 example records checked (CSV shape; NOT live integration)')
