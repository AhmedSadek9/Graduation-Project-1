# WANAS Step 25 — Intent proposal (EXPLORATORY)

## Important
- 19 flagged rows in Step 24 = 14 unknown-identity questions that must stay abstentions + 5 answerable intent gaps.
- Proposes **shared_memory** for picnic/family-gathering/family-trip event queries, and **personal_interest** for collection queries.
- **Does not replace** `fact_type_guard_step15.py`, `e5_demo_bridge_step21.py`, or `frozen_e5_config_step16.json`.
- Does **not** lower acceptance threshold 0.8824205 or use expected fact IDs for selection.
- Replay on old test/validation is **not independent evidence**. Real Face/STT and blind evaluation remain pending.

## Windows
Put the three `.py` files into `WANAS_Zinab_Final/experiments/`. Then run from root:

```powershell
python experiments/test_intent_proposal_step25.py
python experiments/e5_intent_proposal_step25.py
```

New outputs are created under `experiments/results/`, containing before/after decision comparisons. Keep this proposal separate until independent review and validation.
