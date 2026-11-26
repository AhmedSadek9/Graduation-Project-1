# Step 27 — Preparing for independent review (WANAS)

## Install

1. Copy `step27_prepare_review.py` to `C:\Users\Zenab\Desktop\WANAS_Zinab_Final\experiments\`.
2. Open PowerShell in `WANAS_Zinab_Final` and run:

```powershell
python experiments/step27_prepare_review.py
```

3. Outputs will be generated in `experiments/step27/`:
   - `STEP27_HANAA_REVIEW_QUEUE.csv`: 40 drafts with review columns, **not signed-off**.
   - `STEP27_EXPERIMENTAL_POLICY.json`: exploratory 0.80 candidate versus 0.8824205 baseline, **not deployable**.
   - `STEP27_READINESS_REPORT.json`: readiness and structural checks.

## Hanaa's review (later)

Hanaa needs to check person IDs versus face recognition stable IDs, acceptable fact IDs, factual answerability, actual absence of facts for unanswerable examples, ambiguous phrasing, sensitive personal data, contradictions, paraphrase leakage across existing train/validation/test, and record notes/results. The author must not certify their own question. AI-assisted drafts are never automatically an *independently authored blind holdout* merely because they receive review. For a genuinely blind final benchmark, use newly authored unseen questions, a separate reviewer, and lock labels and model settings before final scoring.

## Important

- The script deliberately does **not** call E5, change thresholds, run final evaluation, modify the original 120 facts or 40 questions, or replace the runtime Bridge.
- The candidate `0.80` was observed on previously inspected synthetic Validation. It is not proven safe or ready for production, especially with only six unanswerable validation questions.
- Wait for real face outputs from Ahmed and real STT transcripts from Toqa for upstream reliability testing.
