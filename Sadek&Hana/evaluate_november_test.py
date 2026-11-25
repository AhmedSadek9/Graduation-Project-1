from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from compare_november import ROOT, MODELS, evaluate

OUT = ROOT / "test_results"
THRESHOLDS = {
    ("buffalo_l", "single"): 0.45,
    ("buffalo_l", "two_image_mean"): 0.54,
    ("buffalo_s", "single"): 0.36,
    ("buffalo_s", "two_image_mean"): 0.41,
}

def apply_fixed_threshold(group, threshold, load_ms):
    known = group.role.eq("known_probe")
    scored = group.status.eq("scored")
    accepted = scored & group.score.ge(threshold)
    correct = accepted & known & group.top1.eq(group.expected)
    rejected_unknown = ~known & scored & ~accepted
    balanced = (correct.sum() / known.sum() + rejected_unknown.sum() / (~known).sum()) / 2
    decisions = group.status.copy()
    decisions.loc[scored] = "unknown"
    decisions.loc[accepted] = group.loc[accepted, "top1"]
    return dict(
        threshold=threshold,
        balanced_accuracy=balanced,
        known_correct=int(correct.sum()),
        known_wrong_id=int((known & accepted & ~correct).sum()),
        known_rejected=int((known & scored & ~accepted).sum()),
        unknown_accepted=int((~known & accepted).sum()),
        unknown_rejected=int((~known & scored & ~accepted).sum()),
        known_failures=int((known & ~scored).sum()),
        unknown_failures=int((~known & ~scored).sum()),
        median_ms=group.recognition_ms.median(),
        p95_ms=group.recognition_ms.quantile(0.95),
        model_load_ms=load_ms,
    ), decisions

def main():
    manifest = pd.read_csv(ROOT / "WANAS_Face_Recognition_Split_Manifest.csv")
    cases = manifest.loc[manifest.split.eq("held_out_test")]
    OUT.mkdir(exist_ok=True)
    ort.preload_dlls()
    frames, times = [], {}
    for model, folder in MODELS.items():
        print("Evaluating on TEST:", model, flush=True)
        frame, times[model] = evaluate(model, folder, cases)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    summaries = []
    for (model, mode), group in results.groupby(["model", "enrollment"]):
        th = THRESHOLDS[(model, mode)]
        summary, decisions = apply_fixed_threshold(group, th, times[model])
        results.loc[group.index, "decision"] = decisions
        summaries.append(dict(model=model, enrollment=mode, **summary))
    results.to_csv(OUT / "predictions.csv", index=False)
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / "summary.csv", index=False)
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
