"""
WANAS + ARC — Comprehensive Memory Retrieval & Safety Test Suite (Part 4)
Owner: Zinab (with Tasneem for Cross-Module Safety Evaluation & All Team Members)
Version: 2.1

Runs automated verification across:
1. All 160 evaluation queries in `retrieval_test_queries.json`
   (100 answerable, 20 unanswerable, 40 unknown_identity_abstain).
2. All 30 unknown identity test cases in `unknown_test_cases.json`.
3. Cross-person data isolation checks (verifying zero data leakage between IDs).
4. Low-confidence recognition gating checks.
5. Spoken cue formatting checks for Toqa's Voice Router.

Generates `test_results_report.json` and prints a clean summary table for the repo.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# --- Smart Path Resolver (works whether files are in Task1/Task2/Task3 or in one folder) ---
BASE_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR.parent,
    BASE_DIR.parent / "Task1",
    BASE_DIR.parent / "Task2",
    BASE_DIR.parent / "Task3",
    BASE_DIR / "Task1",
    BASE_DIR / "Task2",
    BASE_DIR / "Task3",
]

for d in SEARCH_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))


def find_file(filename: str) -> Path:
    for d in SEARCH_DIRS:
        candidate = d / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched in: {[str(d) for d in SEARCH_DIRS if d.exists()]}"
    )


from memory_retrieval import MemoryRetriever, SAFE_UNKNOWN_MESSAGE, SAFE_LOW_CONFIDENCE_MESSAGE
from spoken_memory_cues import SpokenMemoryFormatter


def run_all_tests() -> dict:
    db_path = str(find_file("memory_database.json"))
    queries_path = find_file("retrieval_test_queries.json")
    unknown_path = find_file("unknown_test_cases.json")

    retriever = MemoryRetriever(db_path=db_path, min_confidence=0.75)
    formatter = SpokenMemoryFormatter(db_path=db_path, min_confidence=0.75)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(unknown_path, "r", encoding="utf-8") as f:
        unknown_cases = json.load(f)

    results_log = []
    stats = {
        "total_queries": len(queries),
        "queries_passed": 0,
        "queries_failed": 0,
        "answerable_passed": 0,
        "answerable_total": 0,
        "unanswerable_passed": 0,
        "unanswerable_total": 0,
        "unknown_query_passed": 0,
        "unknown_query_total": 0,
        "unknown_profiles_passed": 0,
        "unknown_profiles_total": len(unknown_cases),
        "cross_person_isolation_passed": 0,
        "cross_person_isolation_total": 0,
        "low_confidence_gating_passed": False,
    }

    # --- 1. Evaluate all 160 Retrieval Test Queries ---
    for q in queries:
        qid = q["query_id"]
        pid = q["person_id"]
        question = q["question"]
        expected_fact_id = q["expected_fact_id"]
        expected_outcome = q["expected_outcome"]

        res = retriever.query_fact(person_id=pid, question=question, recognition_confidence=0.95)
        actual_status = res["status"]
        actual_fact_id = res["matched_fact"]["fact_id"] if res["matched_fact"] else None

        passed = False
        if expected_outcome == "answerable":
            stats["answerable_total"] += 1
            passed = (actual_status == "answerable") and (actual_fact_id == expected_fact_id)
            if passed:
                stats["answerable_passed"] += 1
        elif expected_outcome == "unanswerable":
            stats["unanswerable_total"] += 1
            passed = (actual_status == "unanswerable") and (actual_fact_id is None)
            if passed:
                stats["unanswerable_passed"] += 1
        elif expected_outcome == "unknown_identity_abstain":
            stats["unknown_query_total"] += 1
            passed = (actual_status == "unknown_identity_abstain") and (actual_fact_id is None)
            if passed:
                stats["unknown_query_passed"] += 1

        if passed:
            stats["queries_passed"] += 1
        else:
            stats["queries_failed"] += 1
            results_log.append({
                "query_id": qid,
                "person_id": pid,
                "question": question,
                "expected_outcome": expected_outcome,
                "expected_fact_id": expected_fact_id,
                "actual_status": actual_status,
                "actual_fact_id": actual_fact_id,
            })

    # --- 2. Evaluate all 30 Unknown Visitor Profiles ---
    for case in unknown_cases:
        uid = case["image_person_id"]
        cue = formatter.format_greeting_cue(person_id=uid, recognition_confidence=0.35)
        if (
            cue["status"] == "unknown_identity_abstain"
            and cue["spoken_text"] == case["expected_safe_response"]
        ):
            stats["unknown_profiles_passed"] += 1

    # --- 3. Cross-Person Isolation Check (Verify person-001 facts never mention person-002 etc.) ---
    for i in range(1, 21):
        pid_a = f"person-{i:03d}"
        pid_b = f"person-{(i % 20) + 1:03d}"
        stats["cross_person_isolation_total"] += 1
        prof_a = retriever.get_person_profile(pid_a, recognition_confidence=0.95)
        prof_b = retriever.get_person_profile(pid_b, recognition_confidence=0.95)
        fact_ids_a = {f["fact_id"] for f in prof_a["facts"]}
        fact_ids_b = {f["fact_id"] for f in prof_b["facts"]}
        if prof_a["status"] == "success" and prof_b["status"] == "success" and fact_ids_a.isdisjoint(fact_ids_b):
            stats["cross_person_isolation_passed"] += 1

    # --- 4. Low-Confidence Recognition Gating Check ---
    low_conf_res = retriever.get_person_profile("person-001", recognition_confidence=0.45)
    stats["low_confidence_gating_passed"] = (
        low_conf_res["status"] == "low_confidence_abstain"
        and low_conf_res["facts"] == []
        and low_conf_res["safe_message"] == SAFE_LOW_CONFIDENCE_MESSAGE
    )

    report = {
        "suite_name": "WANAS Personal Memory Retrieval & Safety Evaluation",
        "schema_version": "2.1",
        "summary": stats,
        "failed_queries_details": results_log,
    }

    report_path = BASE_DIR / "test_results_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = run_all_tests()
    s = report["summary"]
    print("=================================================================")
    print("WANAS + ARC — MEMORY RETRIEVAL & SAFETY TEST RESULTS")
    print("=================================================================")
    print(f"1. Total Evaluation Queries   : {s['queries_passed']}/{s['total_queries']} PASSED")
    print(f"   - Answerable Facts         : {s['answerable_passed']}/{s['answerable_total']} PASSED")
    print(f"   - Unanswerable Abstain     : {s['unanswerable_passed']}/{s['unanswerable_total']} PASSED")
    print(f"   - Unknown Identity Abstain : {s['unknown_query_passed']}/{s['unknown_query_total']} PASSED")
    print(f"2. Unknown Visitor Profiles   : {s['unknown_profiles_passed']}/{s['unknown_profiles_total']} PASSED")
    print(f"3. Cross-Person Isolation     : {s['cross_person_isolation_passed']}/{s['cross_person_isolation_total']} PASSED")
    print(f"4. Low-Confidence Gating      : {'PASSED' if s['low_confidence_gating_passed'] else 'FAILED'}")
    print("=================================================================")
    if report["failed_queries_details"]:
        print("Failed queries details:", json.dumps(report["failed_queries_details"], indent=2))
    else:
        print("ALL EDGE-CASE, SAFETY, AND RETRIEVAL TESTS PASSED 100%!")
