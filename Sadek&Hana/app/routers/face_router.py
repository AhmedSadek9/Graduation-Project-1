
from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

from app.services.face_recognition import face_service
from app.schemas.face_schema import FaceRecognitionResponse

router = APIRouter(
    prefix="/face",
    tags=["Face Recognition"]
)


@router.post(
    "/recognize",
    response_model=FaceRecognitionResponse
)
async def recognize_face(file: UploadFile = File(...)):

    # 1. Validate uploaded file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        # 2. Read uploaded image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # 3. Decode image using OpenCV
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode the uploaded image."
            )

        # 4. Initialize Face Recognition Service
        face_service.initialize()

        # 5. Recognize face
        result = face_service.recognize(image)

        # 6. Return structured response
        return FaceRecognitionResponse(
            success=True,
            filename=file.filename or "uploaded_image",
            result=result
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Face recognition failed: {str(e)}"
        )
