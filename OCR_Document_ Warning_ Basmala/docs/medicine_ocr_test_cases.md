# Medicine OCR Test Cases

## Purpose
Evaluate OCR text extraction from medicine packaging and labels. These tests assess text extraction only and do not validate medication safety or medical decisions.

## Test Cases

| ID | Test Scenario | Expected OCR Behavior | Review Required |
|---|---|---|---|
| MED-01 | Clear medicine package | Extract visible text accurately | Depends on warning rules |
| MED-02 | Blurry package | Extract available text; low confidence may trigger warning | Yes if warning triggered |
| MED-03 | Partially visible label | Extract visible text without inventing missing text | Yes |
| MED-04 | Glare or reflection | Extract readable text; report uncertainty | Yes if incomplete |
| MED-05 | Small printed text | Extract text if readable; flag low confidence | Yes if warning triggered |
| MED-06 | Empty or unreadable image | Return no_text or error | Yes |
| MED-07 | Short extracted text | Extract available text and apply insufficient_text rule | Yes |
| MED-08 | Mixed Arabic and English | Extract both languages when supported | Review extraction quality |

## Evaluation Criteria
- OCR status
- Extracted text
- Confidence score and scale
- Processing time
- Warning category
- Warnings list
- Review required
- Correct handling of unreadable or incomplete inputs

## Limitations
- OCR output is not medical advice.
- OCR does not verify medicine authenticity or correctness.
- Extracted medicine information must be reviewed by a qualified human.
- Test cases require actual sample images and team review.
- These test cases are a draft, not clinically validated.

## Review Status
Draft — requires review and approval by the team, including Tasneem's medicine test-case review.
