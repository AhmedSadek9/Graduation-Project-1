"""Logic-only tests: fake embeddings; NO model load / no claim of E5 performance."""
from e5_demo_bridge_step21 import E5DemoBridge

class FakeModel:
    def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
        values=[]
        for sentence in sentences:
            values.append([1.0,0.0] if "Omar Hassan is your son" in sentence or sentence.startswith("query: Who is") else [0.0,1.0])
        return values

face={"status":"recognized","person_id":"person-001","recognition_confidence":0.95}
b=E5DemoBridge(model=FakeModel(),data_mode="demo")
cases=[
    (face,"Who is Omar Hassan?","answerable"),
    (face,"What is Omar Hassan's phone number?","unanswerable"),
    ({**face,"status":"unknown"},"Who is Omar Hassan?","face_not_recognized_abstain"),
    ({**face,"recognition_confidence":None},"Who is Omar Hassan?","invalid_confidence_abstain"),
    (face,"Who is Salma Nabil?","person_mismatch_abstain"),
    ({**face,"person_id":"person-999"},"Who is Omar Hassan?","not_found_abstain"),
]
for i,(f,q,expected) in enumerate(cases,1):
    got=b.process(f,q)
    assert got["status"]==expected, (i,got,expected)
    if expected=="answerable": assert got["fact_id"]=="person-001-F01"
    else: assert not got["can_speak_personal_memory"]
    print(f"{i:02d} PASS {expected}")
prod=E5DemoBridge(model=FakeModel())
assert prod.process(face,"Who is Omar Hassan?")["status"] == "production_not_approved"
print("07 PASS production fails closed")
print("STEP 21 logic-only tests: 7/7 PASSED")
