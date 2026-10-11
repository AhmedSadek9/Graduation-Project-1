
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"

EMBEDDINGS_FILE = DATA_DIR / "face_embeddings.json"

# Face recognition model
MODEL_NAME = "buffalo_l"
MODEL_PROVIDER = "CUDAExecutionProvider"

# Recognition settings
SIMILARITY_THRESHOLD = 0.65
MIN_FACE_SIZE = 20

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Standard person identification statuses
STATUS_KNOWN = "known"
STATUS_UNKNOWN = "unknown"
STATUS_LOW_CONFIDENCE = "low_confidence"
