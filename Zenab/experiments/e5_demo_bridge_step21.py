"""Step 21: E5 experimental retrieval bridge (synthetic demo ONLY).

DO NOT use for patient data. No trained/calibrated recognition probability is assumed.
The production FaceMemoryBridge remains unchanged. For the semantic demo:
  python experiments/e5_demo_bridge_step21.py

Loads the cached multilingual-e5-small on first use, not on every request.
Uses the Step16 frozen threshold and Step15 question-type rules.
"""
from __future__ import annotations
import sys
import math
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Task2"))
sys.path.insert(0, str(ROOT / "experiments"))
from memory_retrieval import MemoryRetriever, SAFE_NO_MEMORY_MESSAGE, SAFE_PERSON_MISMATCH_MESSAGE
from fact_type_guard_step15 import allowed_fact_types

THRESHOLD = 0.8824205
CHECKPOINT = "intfloat/multilingual-e5-small"


class E5DemoBridge:
    """Explicit demo-only bridge. No actual face model, microphone or TTS attached."""
    def __init__(self, model=None, db_path=None, threshold=THRESHOLD, data_mode="production", min_face_score=0.75):
        if data_mode not in ("demo", "production"):
            raise ValueError("data_mode must be demo or production")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("invalid threshold")
        self.data_mode = data_mode
        self.threshold = threshold
        self.retriever = MemoryRetriever(str(db_path or ROOT / "Task1" / "memory_database.json"), min_confidence=min_face_score, data_mode=data_mode)
        self.model = model
        self._cache = {}

    def _model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers in this Python environment") from exc
            self.model = SentenceTransformer(CHECKPOINT)
        return self.model

    @staticmethod
    def _abstain(status, reason, pid=None, question=None):
        return {"intent": "retrieve_memory", "status": status, "person_id": pid,
                "question": question, "fact_id": None, "fact_type": None,
                "spoken_text": SAFE_NO_MEMORY_MESSAGE, "can_speak_personal_memory": False,
                "abstention_reason": reason, "retrieval_engine": "e5_demo_step21"}

    def process(self, face_result, question):
        if self.data_mode != "demo":
            return self._abstain("production_not_approved", "semantic_demo_not_approved_for_production")
        if not isinstance(face_result, dict) or face_result.get("status") != "recognized":
            return self._abstain("face_not_recognized_abstain", "missing_or_unrecognized_face")
        pid = face_result.get("person_id")
        confidence = face_result.get("recognition_confidence")
        if not isinstance(pid, str) or not pid.strip():
            return self._abstain("invalid_person_id_abstain", "missing_person_id")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return self._abstain("invalid_confidence_abstain", "invalid_face_score", pid)
        if not isinstance(question, str) or not question.strip():
            return self._abstain("unanswerable", "empty_question", pid, question)
        profile = self.retriever.get_person_profile(pid, recognition_confidence=confidence)
        if profile["status"] != "success":
            return self._abstain(profile["status"], profile["status"], pid, question)
        mentioned = self.retriever._mentioned_person_ids(question.lower())
        if mentioned and mentioned != {pid}:
            return self._abstain("person_mismatch_abstain", "question_identity_mismatch", pid, question)
        allowed, reason = allowed_fact_types(question)
        # Unrecognized intent must abstain rather than retrieve an arbitrary personal fact.
        if allowed is None or not allowed:
            return self._abstain("unanswerable", reason, pid, question)
        facts = sorted((f for f in profile["facts"] if f.get("fact_type") in allowed), key=lambda f:f["fact_id"])
        if not facts:
            return self._abstain("unanswerable", "no_matching_verified_fact_type", pid, question)
        # E5 prefixes; same query/passage text convention as evaluation code.
        passages = ["passage: " + profile["name"] + ". " + f["memory_fact"] for f in facts]
        model = self._model()
        cache_key = (pid, tuple((f["fact_id"], f["memory_fact"]) for f in facts))
        if cache_key not in self._cache:
            self._cache[cache_key] = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = model.encode(["query: " + question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = [(sum(float(a)*float(b) for a,b in zip(query_embedding, vector)), f)
                  for vector, f in zip(self._cache[cache_key], facts)]
        scores.sort(key=lambda sf: (-sf[0], sf[1]["fact_id"]))
        score, fact = scores[0]
        if score < self.threshold:
            result = self._abstain("unanswerable", "below_frozen_retrieval_threshold", pid, question)
            result["top_candidate_score"] = round(score, 6)
            return result
        return {"intent":"retrieve_memory", "status":"answerable", "person_id":pid,
                "question":question,"fact_id":fact["fact_id"],"fact_type":fact["fact_type"],
                "spoken_text":fact["memory_fact"], "can_speak_personal_memory":True,
                "retrieval_score":round(score, 6),"retrieval_engine":"e5_demo_step21",
                "abstention_reason": None, "upstream_origin":"simulated_face_result"}


if __name__ == "__main__":
    bridge = E5DemoBridge(data_mode="demo")
    face = {"status": "recognized", "person_id": "person-001", "recognition_confidence": 0.95}
    for question in ("Who is Omar Hassan?", "What is Omar Hassan's phone number?", "What memory do I share with Omar Hassan?"):
        print(json.dumps(bridge.process(face, question), indent=2))
