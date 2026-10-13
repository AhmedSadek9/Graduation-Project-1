
from core.context import SystemContext

context = SystemContext()

context.update_person({
    "person_id": "person-001",
    "name": "Ahmed",
    "relationship": "Son",
    "confidence": 0.91,
    "status": "known"
})

print(context.current_person["person_id"])
