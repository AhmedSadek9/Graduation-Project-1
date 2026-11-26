"""Evaluate real upstream runs. Requires filled actual_person_id and actual_stt_transcript.
No simulated prediction is accepted. Run BM25 script first.
"""
import json,csv,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(root/'handoff/upstream_predictions_template.csv',encoding='utf-8-sig')))
if not rows or any(not r['actual_person_id'] or not r['actual_stt_transcript'] or r['data_origin']!='actual_model_output' for r in rows):
 raise SystemExit('PENDING: actual Hanaa/Ahmed face decisions + Toqa STT transcripts and data_origin=actual_model_output required. No end-to-end metric calculated.')
raise SystemExit('Actual predictions supplied; now adapt frozen model runner to rerun queries before computing loss. Do not substitute oracle scores.')
