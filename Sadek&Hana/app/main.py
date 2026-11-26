
from fastapi import FastAPI

from app.routers.face_router import router as face_router

app = FastAPI(
    title="WANAS API",
    description="Backend API for WANAS Assistive System",
    version="1.0.0"
)

# Register Face Recognition Router
app.include_router(face_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to WANAS Backend",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "WANAS"
    }
