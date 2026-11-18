# Basmala – Document Risk Warning Rules

## 1. Purpose
This document describes the first transparent, rule-based warning prototype used by the Basmala OCR module.

The warning system identifies OCR output conditions that require manual review. It does not make autonomous medical or legal decisions.

## 2. Warning Rules

| Rule ID | Condition | Warning Message | Review Required |
|---|---|---|---|
| ocr_failed | OCR status is not `success` | OCR processing failed | Yes |
| empty_text | Extracted text is empty | No text was extracted | Yes |
| low_ocr_confidence | OCR confidence is below 0.60 | OCR confidence is low | Yes |
| insufficient_text | Extracted text length is below 10 characters | Extracted text may be incomplete | Yes |

## 3. Warning Classification

- No warnings: `warning_category = null`
- One warning: `warning_category` is the corresponding rule ID
- Multiple warnings: `warning_category = multiple_warnings`
- Any warning requires manual review.
- `review_required` indicates that the result should be reviewed by a human.

## 4. Example Results

### Successful OCR with insufficient text
- OCR status: `success`
- Extracted text: `Nexcitual`
- Confidence: `0.913`
- Warning: `insufficient_text`
- Review required: `true`

### OCR failure
- OCR status: `error`
- Extracted text: empty
- Error: `Unreadable or corrupt image`
- Warnings: `ocr_failed`, `empty_text`
- Warning category: `multiple_warnings`
- Review required: `true`

## 5. Safety and Limitations

- These rules evaluate OCR output quality, not medication safety or document authenticity.
- A high OCR confidence score does not guarantee that extracted text is correct.
- The current thresholds (0.60 confidence and 10 characters) are prototype values and require team/supervisor review.
- Document type is currently returned as `unknown` when it cannot be identified.
- Warnings flag the need for human review; they do not replace professional judgment.
- Medicine-specific warning rules have not been approved or validated.

## 6. Review and Approval

This is a draft prototype rule table. The examples, thresholds, and document types should be reviewed with the team and supervisor before being treated as approved rules.
