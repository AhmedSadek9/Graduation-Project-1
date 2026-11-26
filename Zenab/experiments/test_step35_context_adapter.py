"""Run from project root: python experiments/test_step35_context_adapter.py"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from step35_context_adapter import SafeContextAdapter

VALID_FACE = {"status": "recognized", "person_id": "person-001", "recognition_confidence": 0.95}
SESSION = "fictional-session-1"
CONTEXT = {"referent_person_id": "person-001", "session_id": SESSION, "fresh": True}
QUESTION = "Tell me about him"

class StubBridge:
    def __init__(self): self.calls = []
    def process(self, face_result, question):
        self.calls.append((face_result, question))
        return {"status":"answerable", "person_id":face_result.get('person_id') if isinstance(face_result,dict) else None,
                "fact_id":"person-001-F01", "spoken_text":"Fictional demonstration only",
                "can_speak_personal_memory":True}


def main():
    stub = StubBridge()
    adapter = SafeContextAdapter(stub, lambda pid: "Omar Hassan" if pid == "person-001" else None)
    cases = [
        ("unknown", {**VALID_FACE, "status":"unknown"}, CONTEXT, SESSION),
        ("no_face", {**VALID_FACE, "status":"no_face"}, CONTEXT, SESSION),
        ("no_valid_face", {**VALID_FACE, "status":"no_valid_face"}, CONTEXT, SESSION),
        ("low_score", {**VALID_FACE, "recognition_confidence":0.3}, CONTEXT, SESSION),
        ("missing_score", {"status":"recognized", "person_id":"person-001"}, CONTEXT, SESSION),
        ("nan_score", {**VALID_FACE, "recognition_confidence":float('nan')}, CONTEXT, SESSION),
        ("unknown_id", {**VALID_FACE, "person_id":"unknown-visitor"}, CONTEXT, SESSION),
        ("missing_session", VALID_FACE, CONTEXT, None),
        ("missing_context", VALID_FACE, None, SESSION),
        ("stale_context", VALID_FACE, {**CONTEXT, "fresh":False}, SESSION),
        ("wrong_session", VALID_FACE, {**CONTEXT, "session_id":"old-session"}, SESSION),
        ("other_person", VALID_FACE, {**CONTEXT, "referent_person_id":"person-002"}, SESSION),
        ("other_current_face", {**VALID_FACE,"person_id":"person-002"}, CONTEXT, SESSION),
        ("empty_referent", VALID_FACE, {**CONTEXT,"referent_person_id":None}, SESSION),
    ]
    passed = 0
    for label,face,ctx,sid in cases:
        n = len(stub.calls)
        result = adapter.process(face,QUESTION,ctx,sid)
        assert result["can_speak_personal_memory"] is False, label
        assert not result["fact_id"], label
        assert len(stub.calls) == n, f"retrieval happened during {label}"
        print("PASS BLOCK",label,result["abstention_reason"])
        passed += 1
    result=adapter.process(VALID_FACE,QUESTION,CONTEXT,SESSION)
    assert result["resolved_question"] == "Who is Omar Hassan?"
    assert stub.calls[-1][1]=="Who is Omar Hassan?"
    assert result['context_resolution'].startswith('same_session')
    print("PASS valid matched context delegates to bridge")
    passed+=1
    result=adapter.process(VALID_FACE,"Who is Omar Hassan?",None,None)
    assert stub.calls[-1][1]=="Who is Omar Hassan?"
    print("PASS explicit question delegated unmodified")
    passed+=1
    print(f"STEP 35 UNIT TESTS: {passed}/{passed} PASSED (stub only, not E2E)")

if __name__ == '__main__':main()
