
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. Paths
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "faces"

OLD_FILE = DATA_DIR / "enrollment_embeddings.npz"
NEW_FILE = DATA_DIR / "enrollment_embeddings_candidate.npz"

OUTPUT_FILE = DATA_DIR / "enrollment_comparison.csv"


# ==========================================
# 2. Load Embeddings
# ==========================================

old_data = np.load(OLD_FILE)
new_data = np.load(NEW_FILE)

old_ids = old_data["person_ids"]
new_ids = new_data["person_ids"]

old_embeddings = old_data["embeddings"]
new_embeddings = new_data["embeddings"]


# ==========================================
# 3. Validate Data
# ==========================================

if not np.array_equal(old_ids, new_ids):
    raise ValueError("Person IDs do not match!")

if old_embeddings.shape != new_embeddings.shape:
    raise ValueError("Embedding shapes do not match!")

print("=" * 65)
print("WANAS - ENROLLMENT COMPARISON")
print("=" * 65)

print("Old embeddings shape:", old_embeddings.shape)
print("New embeddings shape:", new_embeddings.shape)


# ==========================================
# 4. Compare Old and New Prototypes
# ==========================================

results = []

for person_id, old_emb, new_emb in zip(
    old_ids,
    old_embeddings,
    new_embeddings
):

    old_emb = old_emb.astype(np.float32)
    new_emb = new_emb.astype(np.float32)

    old_norm = old_emb / (np.linalg.norm(old_emb) + 1e-10)
    new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-10)

    similarity = float(np.dot(old_norm, new_norm))

    difference = float(np.linalg.norm(old_norm - new_norm))

    results.append({
        "person_id": str(person_id),
        "old_new_similarity": similarity,
        "embedding_difference": difference
    })


# ==========================================
# 5. Save Comparison
# ==========================================

comparison_df = pd.DataFrame(results)

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("COMPARISON RESULTS")
print("=" * 65)

print(comparison_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nComparison completed!")
