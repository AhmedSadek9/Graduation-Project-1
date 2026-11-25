from pathlib import Path
from time import perf_counter
import json

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parent / "data" / "november"
OUT = ROOT / "validation_results"
MODELS = {"buffalo_l": ROOT, "buffalo_s": ROOT / "buffalo_s"}


def evaluate(model, folder, cases):
    with np.load(folder / "enrollment_embeddings_candidate.npz") as gallery:
        ids = gallery["person_ids"].astype(str)
        templates = gallery["embeddings"]
        singles = gallery["individual_embeddings"][:, 0]
    if set(ids) != {f"person-{i:03}" for i in range(1, 21)} or len(ids) != 20:
        raise ValueError(f"Unexpected gallery IDs for {model}")
    start = perf_counter()
    app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    load_ms = (perf_counter() - start) * 1000
    rows = []
    for case in cases.itertuples():
        filename = Path(case.image_path.replace("\\", "/")).name
        path = ROOT / "faces_new" / "faces" / case.person_id / filename
        image = cv2.imread(str(path))
        start = perf_counter()
        common = dict(model=model, image=filename, expected=case.person_id, role=case.role,
                      all_boxes="[]", selected_box="", status="missing_or_unreadable")
        candidates = {}
        if image is not None:
            faces = app.get(image)
            common["all_boxes"] = json.dumps([f.bbox.tolist() for f in faces])
            valid = [f for f in faces if min(f.bbox[2:] - f.bbox[:2]) >= 20]
            common["status"] = "no_valid_face" if faces else "no_face"
            if valid:
                face = max(valid, key=lambda f: float(f.det_score))
                common.update(selected_box=json.dumps(face.bbox.tolist()), status="scored")
                for mode, matrix in (("single", singles), ("two_image_mean", templates)):
                    scores = matrix @ face.normed_embedding
                    first, second = np.argsort(-scores, kind="stable")[:2]
                    candidates[mode] = dict(top1=ids[first], score=float(scores[first]),
                                            top2=ids[second], score2=float(scores[second]))
        common["recognition_ms"] = (perf_counter() - start) * 1000 if image is not None else np.nan
        for mode in ("single", "two_image_mean"):
            rows.append(dict(common, enrollment=mode, **candidates.get(mode, {})))
    return pd.DataFrame(rows).reindex(columns=list(common) + ["enrollment","top1","score","top2","score2"]), load_ms


def summarize(group, load_ms):
    known = group.role.eq("known_probe")
    scored = group.status.eq("scored")
    if not scored.any():
        raise ValueError("No usable predictions; inspect predictions_raw.csv")
    choices = []
    for threshold in np.linspace(0, 1, 101):
        accepted = scored & group.score.ge(threshold)
        correct = accepted & known & group.top1.eq(group.expected)
        rejected_unknown = ~known & scored & ~accepted
        balanced = (correct.sum()/known.sum() + rejected_unknown.sum()/(~known).sum()) / 2
        choices.append((balanced, -int((accepted & ~known).sum()), threshold))
    balanced, _, threshold = max(choices)
    accepted = scored & group.score.ge(threshold)
    correct = accepted & known & group.top1.eq(group.expected)
    decisions = group.status.copy()
    decisions.loc[scored] = "unknown"
    decisions.loc[accepted] = group.loc[accepted, "top1"]
    return dict(threshold=threshold, balanced_accuracy=balanced,
                known_correct=int(correct.sum()), known_wrong_id=int((known & accepted & ~correct).sum()),
                known_rejected=int((known & scored & ~accepted).sum()),
                unknown_accepted=int((~known & accepted).sum()), unknown_rejected=int((~known & scored & ~accepted).sum()),
                known_failures=int((known & ~scored).sum()), unknown_failures=int((~known & ~scored).sum()),
                median_ms=group.recognition_ms.median(), p95_ms=group.recognition_ms.quantile(.95),
                model_load_ms=load_ms), decisions


def main():
    manifest = pd.read_csv(ROOT / "WANAS_Face_Recognition_Split_Manifest.csv")
    cases = manifest.loc[manifest.split.eq("validation")]
    if cases.role.value_counts().to_dict() != {"known_probe":20,"unknown_probe":30} or cases.image_path.duplicated().any():
        raise ValueError("Unexpected validation split")
    OUT.mkdir(exist_ok=True)
    ort.preload_dlls()
    frames, times = [], {}
    for model, folder in MODELS.items():
        print("Evaluating", model, flush=True)
        frame, times[model] = evaluate(model, folder, cases)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUT / "predictions_raw.csv", index=False)
    summaries = []
    for (model, mode), group in results.groupby(["model", "enrollment"]):
        summary, decisions = summarize(group, times[model])
        results.loc[group.index, "decision"] = decisions
        summaries.append(dict(model=model, enrollment=mode, **summary))
    results.to_csv(OUT / "predictions.csv", index=False)
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / "summary.csv", index=False)
    print(table.to_string(index=False))
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
