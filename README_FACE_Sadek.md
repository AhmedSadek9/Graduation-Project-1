# WANAS — Face Recognition Module

## Overview
This module detects faces, extracts face embeddings, and identifies enrolled people using InsightFace.

## Main Files
- `ai/face/config.py`: Model configuration, file paths, and recognition thresholds.
- `ai/face/enrollment.py`: Registers people and stores their face embeddings.
- `ai/face/face_recognition.py`: Detects faces and matches embeddings against registered people.
- `demo_face_recognition.py`: Runs face recognition on an input image and prints JSON output.
- `test_recognition.py`: Evaluates recognition results on the configured test images.
- `test_unknown.py`: Tests recognition on an image containing an unknown person.

## Requirements
Run the commands from the project root with the project's virtual environment activated.

Required Python packages include OpenCV, NumPy, ONNX Runtime, and InsightFace.

## Run the Demo
```powershell
python demo_face_recognition.py
```

Enter an image path when prompted, for example:
`data/ahmed_test2.jpg`

## Output
The demo returns a JSON object containing:
- `success`: Whether the image was loaded and processed.
- `faces_detected`: Number of face results returned.
- `results`: Recognition results for each detected face.

Each recognition result can contain:
- `person_id`
- `name`
- `relationship`
- `confidence`
- `status`
- `bbox`

Possible statuses currently include `known`, `unknown`, and `low_confidence`.

## Run Evaluation
```powershell
python test_recognition.py
```

The evaluation report is written to `evaluation_results.csv`.

## Current Test Results
The current test set contains three enrolled-person images, and all three passed in the latest recorded run. The unknown-person, no-face, and multiple-face scenarios were also tested successfully.

These are preliminary test results, not a general accuracy guarantee. More images and conditions are needed for a reliable evaluation.

## Important Notes
- Do not run `test_face_pipeline.py` against the live embeddings file unless you intend to add a test person.
- Do not run `register_person.py` repeatedly without checking whether the people already have saved embeddings.
- The recognition threshold is configured in `ai/face/config.py`. Do not change it without evaluating the results.
- Confirm the final JSON contract with the WANAS team before integration.
