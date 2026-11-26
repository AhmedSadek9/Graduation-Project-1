"""WANAS + ARC — Step 6 additional safety regression tests.

Place in Wanas31/Task4 and run from the Wanas31 directory:
    python Task4/test_memory_safety_step6.py

Requires the reviewed Step 4 MemoryRetriever and Step 5 SpokenMemoryFormatter.
Uses synthetic test data only; does not edit the database.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Task2"))
sys.path.insert(0, str(ROOT / "Task3"))

from memory_retrieval import MemoryRetriever
from spoken_memory_cues import SpokenMemoryFormatter

DB = ROOT / "Task1" / "memory_database.json"
assert DB.exists(), f"Missing database: {DB}"

retriever = MemoryRetriever(str(DB), data_mode="demo")
formatter = SpokenMemoryFormatter(str(DB), data_mode="demo")
production = MemoryRetriever(str(DB), data_mode="production")
production_formatter = SpokenMemoryFormatter(str(DB), data_mode="production")

cases = [
    ("Known profile allowed in demo", lambda: retriever.get_person_profile("person-001", 0.95)["status"] == "success"),
    ("Missing person safely rejected", lambda: retriever.get_person_profile(None, 0.95)["status"] == "unknown_identity_abstain"),
    ("Unknown person safely rejected", lambda: retriever.get_person_profile("unknown-001", 0.95)["status"] == "unknown_identity_abstain"),
    ("Unknown identifier safely rejected", lambda: retriever.get_person_profile("person-999", 0.95)["status"] == "not_found_abstain"),
    ("Missing confidence rejected", lambda: retriever.get_person_profile("person-001", None)["status"] == "invalid_confidence_abstain"),
    ("NaN confidence rejected", lambda: retriever.get_person_profile("person-001", float('nan'))["status"] == "invalid_confidence_abstain"),
    ("Infinite confidence rejected", lambda: retriever.get_person_profile("person-001", float('inf'))["status"] == "invalid_confidence_abstain"),
    ("Boolean confidence rejected", lambda: retriever.get_person_profile("person-001", True)["status"] == "invalid_confidence_abstain"),
    ("Out of range confidence rejected", lambda: retriever.get_person_profile("person-001", 1.5)["status"] == "invalid_confidence_abstain"),
    ("Low confidence rejected", lambda: retriever.get_person_profile("person-001", 0.50)["status"] == "low_confidence_abstain"),
    ("Empty question abstains", lambda: retriever.query_fact("person-001", "", 0.95)["status"] == "unanswerable"),
    ("Unsupported favorite color abstains", lambda: retriever.query_fact("person-001", "What is Omar Hassan favorite color?", 0.95)["status"] == "unanswerable"),
    ("Different known person name rejected", lambda: retriever.query_fact("person-001", "Who is Salma Nabil?", 0.95)["status"] == "person_mismatch_abstain"),
    ("Different explicit ID rejected", lambda: retriever.query_fact("person-001", "Who is person-002?", 0.95)["status"] == "person_mismatch_abstain"),
    ("Correct known person allowed", lambda: retriever.query_fact("person-001", "Who is Omar Hassan?", 0.95)["status"] == "answerable"),
    ("Production rejects synthetic profile", lambda: production.get_person_profile("person-001", 0.95)["status"] == "no_verified_facts"),
    ("Production does not expose synthetic name", lambda: production.get_person_profile("person-001", 0.95)["name"] is None),
    ("Production never returns synthetic matched fact", lambda: production.query_fact("person-001", "Who is Omar Hassan?", 0.95)["matched_fact"] is None),
    ("Demo voice cue needs confidence", lambda: formatter.format_greeting_cue("person-001")["status"] == "invalid_confidence_abstain"),
    ("Demo voice cue rejects unknown identity", lambda: formatter.format_greeting_cue("unknown-001", 0.95)["status"] == "unknown_identity_abstain"),
    ("Production voice does not speak synthetic identity", lambda: "Omar" not in production_formatter.format_greeting_cue("person-001", 0.95)["spoken_text"]),
    ("Voice question refuses mismatched identity", lambda: formatter.format_question_cue("person-001", "Who is Salma Nabil?", 0.95)["status"] == "person_mismatch_abstain"),
    ("Voice conversation starter requires confidence", lambda: formatter.format_conversation_starter("person-001")["status"] == "invalid_confidence_abstain"),
    ("Production conversation starter blocks synthetic data", lambda: production_formatter.format_conversation_starter("person-001", 0.95)["status"] == "no_verified_facts"),
]

passed = 0
for index, (name, check) in enumerate(cases, 1):
    try:
        ok = check()
    except Exception as exc:
        print(f"{index:02}. FAIL  {name}: {type(exc).__name__}: {exc}")
        continue
    print(f"{index:02}. {'PASS' if ok else 'FAIL'}  {name}")
    passed += bool(ok)

print(f"\nSTEP 6: {passed}/{len(cases)} PASSED")
if passed != len(cases):
    raise SystemExit(1)
