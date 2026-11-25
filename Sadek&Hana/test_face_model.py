
import onnxruntime as ort
from insightface.app import FaceAnalysis

print("ONNX Runtime:", ort.__version__)

# Load CUDA and cuDNN DLLs from installed NVIDIA packages
ort.preload_dlls()

print("Available Providers:", ort.get_available_providers())

print("\nLoading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("\nModel loaded successfully!")

for name, model in app.models.items():
    print(f"{name}: {model.session.get_providers()}")
