"""Execute every WANAS memory regression suite from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TEST_FILES = ['test_memory_suite.py', 'test_identity_mismatch_step3.py', 'test_verification_step4.py', 'test_spoken_cues_step5.py', 'test_memory_safety_step6.py', 'test_database_integrity_step7.py', 'test_schema_validation_step8.py', 'test_face_memory_integration_step9.py']


def main():
    failed = []
    for filename in TEST_FILES:
        print("\n" + "=" * 65, flush=True)
        print("RUNNING " + filename, flush=True)
        print("=" * 65, flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / 'Task4' / filename)], cwd=str(ROOT))
        if proc.returncode:
            failed.append(filename)
    print("\n" + "=" * 65)
    print(f"SUITES PASSED: {len(TEST_FILES)-len(failed)}/{len(TEST_FILES)}")
    if failed:
        print("FAILED SUITES: " + ", ".join(failed))
        return 1
    print("ALL SUITES PASSED — synthetic fixture tests only")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
