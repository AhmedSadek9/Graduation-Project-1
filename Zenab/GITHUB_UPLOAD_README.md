# GitHub release checklist — WANAS Memory Retrieval

This repo is **synthetic-demo-only**. The original FaceMemoryBridge remains rule-based; the E5 adapter is experimental. `experiments/evaluate_upstream_actual.py` is a placeholder, **not** a working A/B/C/D evaluator.

## Safe to commit

`Task1/` to `Task4/` (fictional fixtures), `benchmark/`, `experiments/` scripts and documented synthetic results, `reports/`, `handoff/` **blank templates and synthetic examples**, `docs/`, `README.md`, `requirements.txt`, `.gitignore`.

## Never commit without explicit consent and an approved secure workflow

Real face images, raw audio, patient identities, actual participant-level face/STT predictions, private mappings, tokens, credentials, HF model cache, environments and archived ZIP backups. `.gitignore` blocks common examples; **always inspect `git status` and staged changes manually**. If a sensitive file was already tracked, `.gitignore` alone cannot untrack it.

## Quick smoke tests

```powershell
python run_all_tests.py
python experiments/test_e5_demo_bridge_step21.py
python experiments/test_intent_proposal_step25.py
python experiments/step28_readiness_check.py
```

## Git workflow

From the project root, run `git status`, `git remote -v`, and `git branch --show-current` first. If this folder is not already a clone of the team repository, **clone the team repo and copy the reviewed contents into your new branch** rather than initializing a different unrelated repo. Never force push. Create a PR for team review.

See `PROGRESS_AND_GAPS.md` for actual metrics and open tasks.
