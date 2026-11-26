"""WANAS Step 28: local document/readiness check. No models, no live integration."""
from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'experiments' / 'step27'

checks = {
    'core_memory_bridge': ROOT / 'memory_integration.py',
    'synthetic_memory_db': ROOT / 'Task1' / 'memory_database.json',
    'question_manifest': ROOT / 'benchmark' / 'question_manifest.json',
    'step21_e5_demo': ROOT / 'experiments' / 'e5_demo_bridge_step21.py',
    'step27_policy': REVIEW / 'STEP27_EXPERIMENTAL_POLICY.json',
    'step27_readiness': REVIEW / 'STEP27_READINESS_REPORT.json',
    'step27_review_queue': REVIEW / 'STEP27_HANAA_REVIEW_QUEUE.csv',
    'step28_protocol': ROOT / 'handoff' / 'step28' / 'STEP28_INTEGRATION_PROTOCOL.md',
    'face_input_template': ROOT / 'handoff' / 'step28' / 'FACE_OUTPUT_TEMPLATE.csv',
    'stt_input_template': ROOT / 'handoff' / 'step28' / 'STT_TRANSCRIPT_TEMPLATE.csv',
}
print('STEP 28 — LOCAL READINESS CHECK (NO MODEL LOADING OR UPSTREAM EXECUTION)')
missing = []
for key, path in checks.items():
    found = path.is_file()
    print(f"{'PASS' if found else 'MISSING'} {key}: {path.relative_to(ROOT)}")
    if not found: missing.append(key)

policy_ok = False
policy_path = checks['step27_policy']
if policy_path.exists():
    try:
        policy = json.loads(policy_path.read_text(encoding='utf-8-sig'))
        policy_ok = (float(policy.get('baseline_threshold', -1)) == 0.8824205
                     and float(policy.get('candidate_threshold', -1)) == 0.8
                     and 'NOT_DEPLOYABLE' in str(policy.get('status', '')))
    except (ValueError, TypeError, json.JSONDecodeError):
        policy_ok = False
print(('PASS' if policy_ok else 'FAIL') + ' step27 candidate policy is exploratory, baseline is frozen')

# This is a structural check of locally supplied artifacts, not a claim of independent review.
queue_rows = None
queue_path = checks['step27_review_queue']
if queue_path.exists():
    try:
        with queue_path.open(encoding='utf-8-sig', newline='') as f:
            queue_rows = sum(1 for _ in csv.DictReader(f))
    except Exception:
        pass
print('REVIEW QUEUE ROWS:', queue_rows if queue_rows is not None else 'unavailable')

status = {
    'local_document_check': 'PASS' if not missing and policy_ok and queue_rows == 40 else 'INCOMPLETE',
    'independent_evaluation': 'PENDING_HUMAN_REVIEW_AND_NEW_UNSEEN_SET',
    'actual_face_integration': 'PENDING_AHMED_INPUT',
    'actual_stt_integration': 'PENDING_TOQA_INPUT',
    'production_e5': 'NOT_APPROVED',
    'missing_files': missing,
    'step27_policy_verified': policy_ok,
    'step27_review_queue_rows': queue_rows,
}
output = ROOT / 'handoff' / 'step28' / 'STEP28_LOCAL_READINESS_RESULT.json'
output.write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Saved:', output.relative_to(ROOT))
print('Local readiness:', status['local_document_check'])
print('No real Face/STT integration or independent evaluation was executed.')
