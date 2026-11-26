"""WANAS Step 34 — context and stale-identity safety regression.

Execute from project root (Zenab):
    python experiments/test_step34_stale_identity_safety.py

Pure local demo tests with fictional profiles. No real face outputs, no STT,
no model download. Checks the Step 9 Rule-Based bridge and Step 21 E5 demo
face gate; does NOT certify context-aware retrieval or E2E safety.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from memory_integration import FaceMemoryBridge
from e5_demo_bridge_step21 import E5DemoBridge

KNOWN = {"status": "recognized", "person_id": "person-001", "recognition_confidence": 0.95}
QUESTION = "Tell me about him"

def assert_safe(result, title):
    assert isinstance(result, dict), f"{title}: non-dict result"
    assert result.get("can_speak_personal_memory") is False, f"{title}: speech permitted"
    assert not result.get("fact_id"), f"{title}: returned fact_id"
    assert result.get("person_id") != "person-001", f"{title}: stale identity exposed"
    assert result.get("status") not in ("answerable", "success"), f"{title}: answered personal memory"

class FixedModel:
    """Tiny deterministic vector stub; tests safety gates, never semantic accuracy."""
    def encode(self, texts, **kwargs):
        return [[1.0, 0.0] for _ in texts]


def main():
    passes = 0
    for label, bridge in (("RULE_BASED", FaceMemoryBridge(data_mode="demo")),
                          ("E5_DEMO", E5DemoBridge(data_mode="demo", model=FixedModel()))):
        # Establish prior known identity. This must not leak into a later failed face result.
        prior = bridge.process(KNOWN, "retrieve_memory", "Who is Omar Hassan?") if label == "RULE_BASED" else bridge.process(KNOWN, "Who is Omar Hassan?")
        assert isinstance(prior, dict)
        for status in ("unknown", "no_face", "no_valid_face", "low_confidence"):
            face = {"status": status, "person_id": "person-001", "recognition_confidence": 0.95}
            result = bridge.process(face, "retrieve_memory", QUESTION) if label == "RULE_BASED" else bridge.process(face, QUESTION)
            assert_safe(result, f"{label}:{status}")
            passes += 1
            print(f"PASS {label} stale-recognition then {status}: no disclosure")
        for name, face in (
            ("missing_result", None),
            ("missing_score", {"status": "recognized", "person_id": "person-001"}),
            ("invalid_score", {"status": "recognized", "person_id": "person-001", "recognition_confidence": float("nan")}),
            ("unknown_id", {"status": "recognized", "person_id": "unknown-visitor", "recognition_confidence": 0.95}),
        ):
            result = bridge.process(face, "retrieve_memory", QUESTION) if label == "RULE_BASED" else bridge.process(face, QUESTION)
            assert isinstance(result, dict)
            assert not result.get("can_speak_personal_memory") and not result.get("fact_id"), f"{label}:{name}: disclosure"
            passes += 1
            print(f"PASS {label} {name}: no disclosure")
    print(f"STEP 34 — STALE IDENTITY SAFETY: {passes}/{passes} PASSED (fictional demo, unit-level)")
    print("NOTE: E5 was NOT loaded for rejected faces. No real face/STT/context-aware E2E evaluation.")

if __name__ == "__main__":
    main()
