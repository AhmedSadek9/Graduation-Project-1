# Basmala OCR and Document Risk Warning

An incremental OCR and Document Warning module for the Basmala project. Existing datasets and CSV experiment outputs are retained.

## Quick Use

### PaddleOCR

```python
from src.pipeline.ocr_pipeline import OCRPipeline

pipeline = OCRPipeline(
    engine="paddle",
    engine_config={"lang": "en"},
    preprocess=False
)

result = pipeline.run(
    "path/to/image.jpg",
    ground_truth="optional reference"
)

print(result)
```

### Tesseract

```python
pipeline = OCRPipeline(
    engine="tesseract",
    engine_config={"lang": "ara+eng"}
)
```

Set the Tesseract executable using the `TESSERACT_CMD` environment variable or `executable=` in the engine configuration.

Do not commit machine-specific executable paths. Language availability depends on the installed Tesseract model data.

## Output

The standard OCR output includes:

- `extracted_text`
- `confidence`
- `confidence_scale`
- `status`
- `engine`
- `processing_time`
- `error`

Optional evaluation fields:

- `cer`
- `wer`

These are included when ground truth is supplied.

### Confidence Scales

PaddleOCR confidence is the mean recognition score on a 0–1 scale.

Tesseract confidence is the mean valid word confidence on a 0–100 scale.

These scales are engine-specific and are not directly comparable.

## Preprocessing

Existing image-processing operations remain available.

Pipeline preprocessing is opt-in:

```python
preprocess=True
```

The pipeline does not overwrite the source image.

`preprocess_array` accepts an already-loaded image.

## Evaluation

Text normalization currently:

- Removes punctuation.
- Removes Arabic diacritics and tatweel.
- Removes Unicode format marks.
- Maps Arabic and Persian digits to ASCII.
- Applies case-folding.
- Collapses whitespace.

CER ignores spaces, while WER retains token boundaries.

`calculate_corpus_cer` and `calculate_corpus_wer` aggregate edit counts over reference character and word totals, respectively, rather than averaging per-image percentages.

These normalization choices affect evaluation scores and should be reported when presenting experimental results.

## Dataset Configuration

Use a different image directory and matching reference files in your evaluation script.

The OCR pipeline itself does not depend on dataset-specific paths.

Ground-truth filenames should match image stems when using the existing `main.py` evaluator.

## OCR Baseline Results

Evaluation was conducted on an experimental subset of 44 English medication-label images.

| Metric | PaddleOCR | Tesseract |
|---|---:|---:|
| Successful images | 44/44 (100%) | 38/44 (86.36%) |
| Failed images | 0 | 6 |
| Average per-image CER | 14.85% | 79.1% |
| Average per-image WER | 36.36% | 102.63% |
| Corpus CER | 15.05% | 78.75% |
| Corpus WER | 39.13% | 102.5% |
| Average processing time | 2.642 sec/image | 0.856 sec/image |

PaddleOCR achieved better text extraction accuracy and a higher success rate on the evaluated subset.

Tesseract processed images faster but produced substantially higher CER and WER.

PaddleOCR was selected as the primary OCR engine based on the comparative evaluation. Tesseract remains available as an alternative engine.

These results apply only to the current experimental subset and should not be generalized to Arabic documents or other datasets.

## Document Warning CLI

Run the warning CLI on a document image:

```powershell
python warning_cli.py data/raw_images/real_samples/P0001.jpg
```

Use Tesseract instead of PaddleOCR:

```powershell
python warning_cli.py data/raw_images/real_samples/P0001.jpg --engine tesseract
```

The CLI displays:

- OCR status and extracted text.
- OCR confidence.
- Document warning status.
- Warning reasons.
- Whether manual review is required.

## Warning Rules

The warning module identifies OCR extraction-quality issues using transparent rule-based checks.

| Warning | Condition |
|---|---|
| `ocr_failed` | OCR status is not `success` |
| `empty_text` | Extracted text is empty |
| `low_ocr_confidence` | Normalized confidence is below 0.6 |
| `insufficient_text` | Extracted text contains fewer than 10 characters |

### Confidence Threshold

The warning confidence threshold is:

```python
confidence_threshold = 0.6
```

Confidence values on a 0–100 scale are normalized before comparison.

The threshold is an experimental prototype setting and has not been scientifically calibrated against a large representative dataset.

### Warning Output

- `warning_status`: `clear` or `warning`
- `warning_category`: a warning reason, `multiple_warnings`, or `null`
- `warnings`: list of detected warning reasons
- `review_required`: whether manual review is required

When multiple warning conditions are detected, `warning_category` is set to `multiple_warnings`.

These rules assess OCR extraction quality only. They do not independently assess medical, clinical, legal, or document authenticity risks.

## Shared Data Contract

The OCR and Document Warning modules return a shared document result.

Example:

```json
{
  "document_type": "unknown",
  "extracted_text": "Nexcitual",
  "confidence": 0.913,
  "confidence_scale": "0-1",
  "status": "success",
  "engine": "paddle",
  "warning_status": "warning",
  "warning_category": "insufficient_text",
  "warnings": ["insufficient_text"],
  "review_required": true,
  "processing_time": 17.19,
  "error": null
}
```

### Contract Fields

| Field | Description |
|---|---|
| `document_type` | Document type; defaults to `unknown` |
| `extracted_text` | Text extracted by OCR |
| `confidence` | OCR confidence score |
| `confidence_scale` | Confidence scale used by the OCR engine |
| `status` | OCR execution status |
| `engine` | OCR engine used |
| `warning_status` | `clear` or `warning` |
| `warning_category` | Main warning category, `multiple_warnings`, or `null` |
| `warnings` | List of detected warning reasons |
| `review_required` | Whether manual review is required |
| `processing_time` | OCR processing duration |
| `error` | OCR error details, if any |

`document_type` currently defaults to `unknown`; automatic document classification is not implemented.

`warning_category` is derived from the warning list.

These fields support downstream API and database integration.

## Limitations

- OCR accuracy depends on image quality, resolution, lighting, text size, orientation, and document layout.
- Handwriting and complex layouts may produce inaccurate extraction.
- Ground truth is only as reliable as its transcription; verify samples before making evaluation claims.
- Confidence is engine-specific and is not calibrated accuracy.
- CER and WER depend on the selected normalization and tokenization rules.
- The experimental evaluation subset contains English medication-label images and does not establish performance on Arabic documents or other document types.
- Selective preprocessing has not been proven to improve OCR universally. The previous contrast<23 trial was exploratory and only slightly changed CER.
- The warning threshold and minimum text length are heuristic prototype settings and require further evaluation.
- Automatic document classification is not implemented.
- OCR is text extraction only. Document Warning rules are a separate downstream module.
- Warnings indicate a need for manual review and do not replace medical, clinical, legal, or authenticity verification.

## Offline Operation

The OCR pipeline is designed for local execution using PaddleOCR or Tesseract.

Offline execution requires:

- Python dependencies to be installed.
- PaddleOCR model files to be available locally.
- Tesseract executable and required language data to be installed locally when using Tesseract.

Initial dependency or model downloads may require internet access. Offline operation should be verified after installation and model preparation.

## Testing

Run the project tests:

```powershell
pytest -q
```

The test suite covers OCR configuration, evaluation, pipeline behavior, standard output, and warning rules.

