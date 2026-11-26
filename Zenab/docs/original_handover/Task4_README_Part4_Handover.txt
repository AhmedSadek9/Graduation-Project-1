========================================================================
WANAS + ARC — Personal Memory Retrieval (Zinab's Task)
PART 4 HANDOVER: Edge-Case, Safety & Cross-Person Test Suite
========================================================================

1. WHAT IS INCLUDED IN THIS HANDOVER:
- test_memory_suite.py     : Automated verification suite testing 160 queries, 30 unknown identities, cross-person isolation, and low-confidence gating.
- test_results_report.json : Official evaluation output showing 160/160 queries passed (100%).
- requirements_Part4.txt   : Dependencies list (Zero third-party libraries; standard Python 3.8+).
- README_Part4_Handover.txt: Test methodology, results summary, and run instructions.

2. WHO RECEIVES THIS PART & WHY:
- Tasneem (Cross-Module Safety Evaluation & Dashboard Lead) & All Six Team Members:
  * Satisfies the October 31 Definition of Done: normal cases, edge cases, unknown rejection, and cross-person isolation verified and saved in the repository.

3. HOW TO RUN ON YOUR LAPTOP:
1. Ensure the file in `Task2` is named `memory_retrieval.py` (remove any ` (1)` suffix from browser download).
2. Place `test_memory_suite.py` in `Task4` (or in the main `Wanas31` folder).
3. Run:
   python Task4/test_memory_suite.py
   (It automatically locates `Task1`, `Task2`, and `Task3` files and generates `test_results_report.json`.)

4. SUMMARY OF VERIFIED RESULTS:
- Answerable Queries (F01, F03–F06 across 20 people): 100 / 100 PASSED
- Unanswerable / Missing Info Queries (e.g., phone numbers): 20 / 20 PASSED (safely abstained)
- Unknown Identity Queries (unknown-001 .. unknown-030): 40 / 40 PASSED (safely abstained)
- Unknown Visitor Profiles Check: 30 / 30 PASSED
- Cross-Person Data Isolation (zero cross-ID leakage): 20 / 20 PASSED
- Low-Confidence Recognition Gating (< 0.75 threshold): PASSED
========================================================================
