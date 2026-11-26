# Step 17 — What is complete and what remains

**Complete now:** 120 synthetic facts and 140 benchmark questions; BM25, MiniLM, E5 comparison; validation thresholds; RRF hybrid experiment; synthetic context ablation; fact-type guard retrospective analysis; provisional E5 configuration recorded.

**Partially complete:** candidate ranking traces (inspect whether verification-filter trace is preserved for each question); latency (one-run model-specific timings, no robust repeated benchmark, no hybrid incremental time); music diagnosis (constructed query only); Step 16 (40 draft questions, not independently reviewed).

**Not complete:** 150–250 facts (scope deferred); genuinely independent held-out evaluation; actual person-ID mapping confirmation from Hanaa; real STT transcripts from تقى; real face decisions from Ahmed; quantification of end-to-end reliability loss; final independent acceptance of a retrieval method.

**Hanaa review later:** (a) cross-check all 20 IDs against face enrollment/ground truth, (b) inspect each drafted question vs its acceptable fact(s) or missing fact, (c) flag similar names, stale facts, conflicting context, (d) audit paraphrase-group splits and accidental leakage, (e) fill actual reviewer and approval records only after genuinely checking the rows.

**Next local action:** Inspect Step 17 report, save it in `reports/` of the project, and then prepare the final reproducibility checks and the three scenario handoff rows for تقى. No new model downloads are needed to read this report.
