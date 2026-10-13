"""Mock services for early WANAS integration tests."""


class MockFaceService:
    def __init__(self, results=None):
        self.results = results or [
            {
                "person_id": "person-001",
                "name": "Ahmed",
                "relationship": "Son",
                "confidence": 0.91,
                "status": "known",
            }
        ]

    def recognize(self, image):
        return self.results


class MockSTTService:
    def __init__(self, transcript="Tell me about him"):
        self.transcript = transcript

    def transcribe(self, audio_path):
        return self.transcript


class MockIntentService:
    def __init__(self, intent="GET_MEMORY", confidence=0.95):
        self.intent = intent
        self.confidence = confidence

    def classify(self, text):
        return {
            "text": text,
            "intent": self.intent,
            "confidence": self.confidence,
        }


class MockMemoryRetriever:
    def __init__(self, result=None):
        self.calls = []

        self.result = result or {
            "query": "Tell me about him",
            "memory_text": "Ahmed visits every Friday.",
            "verified": True,
            "retrieval_score": 0.88,
            "status": "found",
        }

    def __call__(self, person_id, query):
        self.calls.append({
            "person_id": person_id,
            "query": query,
        })

        return self.result