
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ai.memory.memory import get_memories


FALLBACK_MESSAGE = "لا تتوفر لدي معلومات موثقة كافية"

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


def _keyword_score(query, memory_text):
    """Calculate query-word overlap."""
    query_words = set(query.lower().split())
    memory_words = set(memory_text.lower().split())

    if not query_words:
        return 0.0

    return len(query_words & memory_words) / len(query_words)


def _semantic_score(query, memory_text):
    """Calculate semantic similarity using sentence embeddings."""
    model = _load_model()

    vectors = model.encode(
        [query, memory_text],
        normalize_embeddings=True,
    )

    score = cosine_similarity(
        [vectors[0]],
        [vectors[1]],
    )[0][0]

    # Cosine similarity can be negative; keep score in [0, 1].
    return max(0.0, float(score))


def retrieve_memory(
    person_id,
    query,
    min_score=0.45,
    keyword_weight=0.3,
):
    """
    Retrieve a verified memory using hybrid keyword and semantic search.
    """
    if not person_id or not str(person_id).strip():
        return {
            "query": query,
            "memory_text": FALLBACK_MESSAGE,
            "verified": False,
            "retrieval_score": 0.0,
            "status": "no_person",
        }

    if not query or not str(query).strip():
        return {
            "query": query,
            "memory_text": FALLBACK_MESSAGE,
            "verified": False,
            "retrieval_score": 0.0,
            "status": "empty_query",
        }

    memories = get_memories(person_id, verified_only=True)

    if not memories:
        return {
            "query": query,
            "memory_text": FALLBACK_MESSAGE,
            "verified": False,
            "retrieval_score": 0.0,
            "status": "not_found",
        }

    best_memory = None
    best_score = 0.0
    query_text = str(query).strip()

    for memory in memories:
        keyword_score = _keyword_score(
            query_text,
            memory["memory_text"],
        )

        semantic_score = _semantic_score(
            query_text,
            memory["memory_text"],
        )

        hybrid_score = (
            keyword_weight * keyword_score
            + (1.0 - keyword_weight) * semantic_score
        )

        if hybrid_score > best_score:
            best_score = hybrid_score
            best_memory = memory

    if best_memory is None or best_score < min_score:
        return {
            "query": query,
            "memory_text": FALLBACK_MESSAGE,
            "verified": False,
            "retrieval_score": round(best_score, 3),
            "status": "not_found",
        }

    return {
        "query": query,
        "memory_text": best_memory["memory_text"],
        "verified": True,
        "retrieval_score": round(best_score, 3),
        "status": "found",
    }
