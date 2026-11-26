"""WANAS Step 9: explicit, safe FACE -> MEMORY contract adapter.

This is a local mock/API boundary, NOT a connection to the actual face model.
Call with face_result={'status':'recognized','person_id':'person-001',
                       'recognition_confidence':0.95}.
IMPORTANT: recognition_confidence must be an explicitly provided, validated
0..1 score on an agreed threshold scale; a raw model cosine similarity is NOT
necessarily a calibrated probability. Never fabricate 1.0 on missing score.
"""
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Task2'))
sys.path.insert(0, str(ROOT / 'Task3'))
from spoken_memory_cues import SpokenMemoryFormatter

RECOGNIZED_STATUS = 'recognized'
SAFE_FACE_REJECTION = 'Identity not confirmed — no personal memory available.'


class FaceMemoryBridge:
    def __init__(self, db_path: Optional[str] = None, data_mode: str = 'production',
                 min_confidence: float = 0.75):
        path = db_path or str(ROOT / 'Task1' / 'memory_database.json')
        self.formatter = SpokenMemoryFormatter(path, min_confidence=min_confidence, data_mode=data_mode)

    @staticmethod
    def _reject(status: str, intent: str) -> Dict[str, Any]:
        return {'status': status, 'intent': intent, 'person_id': None,
                'spoken_text': SAFE_FACE_REJECTION, 'can_speak_personal_memory': False}

    def process(self, face_result: Any, intent: str = 'identify_person',
                question: Optional[str] = None) -> Dict[str, Any]:
        """Only accepted intents: identify_person, retrieve_memory, conversation_starter."""
        valid_intents = {'identify_person', 'retrieve_memory', 'conversation_starter'}
        if intent not in valid_intents:
            return self._reject('invalid_intent_abstain', str(intent))
        if not isinstance(face_result, dict):
            return self._reject('invalid_face_result_abstain', intent)
        if face_result.get('status') != RECOGNIZED_STATUS:
            return self._reject('face_not_recognized_abstain', intent)
        pid = face_result.get('person_id')
        if not isinstance(pid, str) or not pid.strip() or pid.lower().startswith('unknown'):
            return self._reject('invalid_person_id_abstain', intent)
        score = face_result.get('recognition_confidence')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            return self._reject('invalid_score_abstain', intent)
        if intent == 'identify_person':
            result = self.formatter.format_greeting_cue(pid, recognition_confidence=score)
        elif intent == 'retrieve_memory':
            if not isinstance(question, str) or not question.strip():
                return self._reject('invalid_question_abstain', intent)
            result = self.formatter.format_question_cue(pid, question, recognition_confidence=score)
        else:
            result = self.formatter.format_conversation_starter(pid, recognition_confidence=score)
        return {**result, 'can_speak_personal_memory': result.get('status') in ('success', 'answerable')}


if __name__ == '__main__':
    import json
    bridge = FaceMemoryBridge(data_mode='demo')  # Synthetic demonstration ONLY
    sample_face = {'status':'recognized','person_id':'person-001','recognition_confidence':0.95}
    print(json.dumps(bridge.process(sample_face), indent=2))
    print(json.dumps(bridge.process(sample_face, 'retrieve_memory', 'Who is Omar Hassan?'), indent=2))
