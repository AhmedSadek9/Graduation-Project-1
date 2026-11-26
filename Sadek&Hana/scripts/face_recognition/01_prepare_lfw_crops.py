import json, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis

t0 = time.time()
ort.preload_dlls()
app = FaceAnalysis(
    name="buffalo_s",
    allowed_modules=["detection"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(320, 320))

PUBLIC_DIR = Path("data/face_recognition/public_dataset")
LFW_MANIFEST = PUBLIC_DIR / "lfw_manifest.csv"
CROPS_DIR = PUBLIC_DIR / "lfw_crops_112"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

df_lfw = pd.read_csv(LFW_MANIFEST)
total = len(df_lfw)
crop_records = []
dropped = 0

for idx, row in enumerate(df_lfw.itertuples(), 1):
    if idx % 500 == 0 or idx == total:
        print(f"Processed {idx}/{total} ({time.time() - t0:.1f}s)...", flush=True)
    img = cv2.imread(row.image_path)
    if img is None:
        dropped += 1
        continue
    faces = app.get(img)
    valid = [
        f for f in faces
        if (f.bbox[2] - f.bbox[0]) >= 20 and (f.bbox[3] - f.bbox[1]) >= 20
    ]
    if not valid:
        dropped += 1
        continue
    best = max(valid, key=lambda f: float(f.det_score))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = best.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = 0.10 * bw, 0.10 * bh
    cx1, cy1 = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
    cx2, cy2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    crop = cv2.resize(img[cy1:cy2, cx1:cx2], (112, 112), interpolation=cv2.INTER_AREA)

    out_sub = CROPS_DIR / row.identity
    out_sub.mkdir(exist_ok=True)
    out_path = out_sub / Path(row.image_path).name
    cv2.imwrite(str(out_path), crop)
    crop_records.append({
        "identity": row.identity,
        "source_path": row.image_path,
        "crop_path": str(out_path),
        "det_score": round(float(best.det_score), 4),
    })

df_crops = pd.DataFrame(crop_records)
counts = df_crops["identity"].value_counts()
keep_ids = sorted(counts[counts >= 8].index.tolist())
df_crops = df_crops[df_crops["identity"].isin(keep_ids)].copy()
id_to_label = {ident: idx for idx, ident in enumerate(keep_ids)}
df_crops["label"] = df_crops["identity"].map(id_to_label)

rng = np.random.default_rng(42)
splits = []
for ident, grp in df_crops.groupby("identity", sort=True):
    idxs = grp.index.to_numpy().copy()
    rng.shuffle(idxs)
    n = len(idxs)
    n_test = max(1, int(round(0.10 * n)))
    n_val = max(1, int(round(0.10 * n)))
    n_train = n - n_val - n_test
    for i, ridx in enumerate(idxs):
        if i < n_train:
            splits.append((ridx, "train"))
        elif i < n_train + n_val:
            splits.append((ridx, "val"))
        else:
            splits.append((ridx, "test"))

df_crops["split"] = df_crops.index.map(dict(splits))

wanas_names = {p.name for p in Path("data/november/faces_new/faces").iterdir() if p.is_dir()}
overlap = wanas_names.intersection(set(keep_ids))
assert len(overlap) == 0, f"Identity overlap found: {overlap}"

df_crops.to_csv(PUBLIC_DIR / "lfw_prepared_splits.csv", index=False)

summary = {
    "dataset": "LFW (min_faces_per_person=10, face-cropped 112x112 via buffalo_s det_500m)",
    "dropped_no_face_or_small": dropped,
    "retained_identities": len(keep_ids),
    "retained_images": len(df_crops),
    "split_counts": df_crops["split"].value_counts().to_dict(),
    "wanas_identity_overlap": len(overlap),
}
(PUBLIC_DIR / "dataset_prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
