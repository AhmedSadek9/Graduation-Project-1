# WANAS Face Recognition Handoff

**Prepared:** 25 November 2026  
**From:** Ahmed  
**To:** Hanaa  
**Status:** Candidate configuration for independent review

## Candidate configuration

| Setting | Value |
|---|---|
| Recognition model | InsightFace `buffalo_s` |
| Enrollment | Normalized mean of two enrollment embeddings per person |
| Gallery | 20 trial identities, 40 enrollment images |
| Similarity | Cosine similarity of L2-normalized embeddings |
| Decision threshold | `0.41`, selected on the 50-image validation split |
| Detection resolution | `640 × 640` |
| Minimum valid face box | 20 pixels in both width and height |
| Runtime providers | CUDAExecutionProvider with CPU fallback |

The four InsightFace configurations tied at 95% validation balanced accuracy. The candidate uses both available enrollment photos and has a smaller reported model size than `buffalo_l`. Treat this as a candidate for independent review, not a claim that it won the validation comparison.

## Evidence

### Validation

- 50 probes total, with 20 known and 30 unknown probes.
- 18/20 known identities were correct, with 0 wrong known identities.
- One known probe was rejected and one known probe had a detection failure.
- 30/30 unknown probes were rejected.
- Balanced accuracy was 95%.

### Held-out evaluation already run by Ahmed

- 59 probes total, with 29 known and 30 unknown probes.
- `buffalo_s` with two-image mean identified 29/29 known probes and rejected 30/30 unknown probes.
- Balanced accuracy was 100% on this set. These results are recorded as observed and were not used to tune the threshold.

The held-out set contains 59 images, so report the counts with the percentages. The team has already inspected these results. Hanaa can independently verify the pipeline and traces, while the report should not describe this set as unseen by the whole team.

## Comparison with trained models

The F1–F4 evaluation was rerun with a complete gallery after an earlier run used an incomplete 9-person gallery. Use only the rerun results below.

| Model | Validation balanced accuracy | Held-out balanced accuracy | Known correct on held-out | Security errors |
|---|---:|---:|---:|---:|
| F1 ResNet-18 pretrained | 57.50% | 63.79% | 8/29 | 0 |
| F2 MobileNetV3-Small pretrained | 52.50% | 58.62% | 5/29 | 0 |
| F3 ResNet-18 scratch | 57.50% | 56.90% | 4/29 | 0 |
| F4 MobileNetV3-Small scratch | 50.00% | 50.00% | 0/29 | 1 |

F4's consequential error was `person-002_test_lighting.jpg`, predicted as `person-009` with cosine score `0.908622` at threshold `0.87`.

## Timing and environment notes

The existing latency figures are not directly comparable. F1–F4 report embedding-only latency. InsightFace reports a wider pipeline latency. Memory figures also use different measurement methods. Keep those fields separate until all pipelines are timed using the same boundaries and method.

The evaluation console reported a PyTorch CUDA 12.x versus ONNX Runtime CUDA 13.x compatibility warning. The run completed, but record the warning and verify the environment before relying on performance measurements.

## Files to include in the handoff

- `data/november/enrollment_face_audit_full.csv`
- `data/november/validation_results/predictions.csv`
- `data/november/validation_results/summary.csv`
- `data/november/test_results/predictions.csv`
- `data/november/test_results/summary.csv`
- `data/face_recognition/wanas_evaluation_F1_F4/full_comparison_F1_F4_vs_InsightFace.csv`
- The corrected evaluation script and the runnable recognition service/configuration

Before handoff, confirm each listed file exists in Ahmed's project folder and record the exact reproduction command. Hanaa's quality scorer, feature procedure, score interpretation, and failure examples are still pending from her side. Video and temporal evaluation remain pending until clips are available.
