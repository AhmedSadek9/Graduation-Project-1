import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import models, transforms


class CosineMarginHead(nn.Module):
    def __init__(self, emb_dim, num_classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        self.scale = scale

    def forward(self, emb):
        return self.scale * F.linear(
            emb,
            F.normalize(self.weight, dim=1)
        )


class FaceEmbeddingNet(nn.Module):
    def __init__(self, arch: str, num_classes: int, emb_dim: int = 512):
        super().__init__()

        if arch == "resnet18":
            backbone = models.resnet18(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=None)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        self.backbone = backbone
        self.emb_head = nn.Sequential(
            nn.Linear(in_features, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.classifier = CosineMarginHead(
            emb_dim,
            num_classes,
            scale=16.0
        )

    def extract_embedding(self, x):
        feat = self.backbone(x)
        emb = self.emb_head(feat)
        return F.normalize(emb, dim=1)


def crop_with_margin(img_bgr, box):
    height, width = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    margin_x, margin_y = 0.10 * box_width, 0.10 * box_height

    crop_x1 = max(0, int(round(x1 - margin_x)))
    crop_y1 = max(0, int(round(y1 - margin_y)))
    crop_x2 = min(width, int(round(x2 + margin_x)))
    crop_y2 = min(height, int(round(y2 + margin_y)))

    crop_bgr = cv2.resize(
        img_bgr[crop_y1:crop_y2, crop_x1:crop_x2],
        (112, 112),
        interpolation=cv2.INTER_AREA
    )
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


def extract_all_wanas_crops(faces_root: Path):
    ort.preload_dlls()

    det_app = FaceAnalysis(
        name="buffalo_s",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    det_app.prepare(ctx_id=0, det_size=(640, 640))

    # Enrollment crops use the audited buffalo_s boxes.
    df_audit = pd.read_csv(
        "data/november/enrollment_face_audit_full.csv"
    )
    sub_audit = df_audit[df_audit["model"] == "buffalo_s"].copy()
    enroll_crops = {}

    for row in sub_audit.itertuples():
        img_name = row.image
        person_id = img_name.split("_")[0]
        img = cv2.imread(str(faces_root / person_id / img_name))
        box = json.loads(row.selected_box)

        enroll_crops.setdefault(person_id, {})[img_name] = (
            crop_with_margin(img, box)
        )

    expected_people = {
        f"person-{number:03d}"
        for number in range(1, 21)
    }
    if set(enroll_crops) != expected_people:
        raise ValueError(
            f"Expected 20 gallery people, found {len(enroll_crops)}."
        )
    wrong_image_counts = {
        person_id: len(images)
        for person_id, images in enroll_crops.items()
        if len(images) != 2
    }
    if wrong_image_counts:
        raise ValueError(
            f"Expected two enrollment images per person: "
            f"{wrong_image_counts}"
        )

    def get_split_crops(pred_csv: Path):
        df = pd.read_csv(pred_csv)
        sub = df[
            (df["model"] == "buffalo_s")
            & (df["enrollment"] == "two_image_mean")
        ].copy()
        records = {}

        for row in sub.itertuples():
            img_name = row.image
            person_folder = img_name.split("_")[0]

            if row.status == "no_face":
                records[img_name] = {
                    "crop": None,
                    "role": row.role,
                    "expected": row.expected,
                }
                continue

            img = cv2.imread(str(faces_root / person_folder / img_name))
            faces = det_app.get(img)
            valid = [
                face for face in faces
                if (face.bbox[2] - face.bbox[0]) >= 20
                and (face.bbox[3] - face.bbox[1]) >= 20
            ]

            if not valid:
                records[img_name] = {
                    "crop": None,
                    "role": row.role,
                    "expected": row.expected,
                }
                continue

            best = max(valid, key=lambda face: float(face.det_score))
            records[img_name] = {
                "crop": crop_with_margin(img, best.bbox),
                "role": row.role,
                "expected": row.expected,
            }

        return records

    val_items = get_split_crops(
        Path("data/november/validation_results/predictions.csv")
    )
    test_items = get_split_crops(
        Path("data/november/test_results/predictions.csv")
    )

    del det_app
    return enroll_crops, val_items, test_items


def tune_threshold_on_val(val_rows):
    best_th = 0.50
    best_key = (-1, -1.0, -1.0)

    for th in np.round(np.arange(0.10, 0.96, 0.01), 2):
        known_tot = known_corr = known_wrong = 0
        unk_tot = unk_rej = unk_acc = 0

        for row in val_rows:
            if row["status"] == "no_face":
                if row["role"] == "known_probe":
                    known_tot += 1
                else:
                    unk_tot += 1
                continue

            pred = row["top1_id"] if row["score"] >= th else "unknown"

            if row["role"] == "known_probe":
                known_tot += 1
                if pred == row["expected"]:
                    known_corr += 1
                elif pred != "unknown":
                    known_wrong += 1
            else:
                unk_tot += 1
                if pred == "unknown":
                    unk_rej += 1
                else:
                    unk_acc += 1

        sec_err = known_wrong + unk_acc
        k_acc = known_corr / known_tot if known_tot else 0.0
        u_rej = unk_rej / unk_tot if unk_tot else 0.0
        bal = 0.5 * (k_acc + u_rej)

        key = (
            1 if sec_err == 0 else 0,
            round(bal, 6),
            round(k_acc, 6),
        )
        if key > best_key:
            best_key = key
            best_th = float(th)

    return best_th


def evaluate_split(rows, threshold):
    known_tot = known_corr = known_wrong = known_unk = known_nf = 0
    unk_tot = unk_rej = unk_acc = unk_nf = 0
    pred_records = []

    for row in rows:
        if row["status"] == "no_face":
            decision = "no_face"
            if row["role"] == "known_probe":
                known_tot += 1
                known_nf += 1
            else:
                unk_tot += 1
                unk_nf += 1
        else:
            decision = (
                row["top1_id"]
                if row["score"] >= threshold
                else "unknown"
            )

            if row["role"] == "known_probe":
                known_tot += 1
                if decision == row["expected"]:
                    known_corr += 1
                elif decision == "unknown":
                    known_unk += 1
                else:
                    known_wrong += 1
            else:
                unk_tot += 1
                if decision == "unknown":
                    unk_rej += 1
                else:
                    unk_acc += 1

        pred_records.append({
            "image": row["image"],
            "role": row["role"],
            "expected": row["expected"],
            "top1_id": row["top1_id"],
            "score": (
                round(row["score"], 6)
                if row["score"] is not None
                else None
            ),
            "threshold": threshold,
            "decision": decision,
            "status": row["status"],
        })

    known_acc_rate = known_corr / known_tot if known_tot else 0.0
    unknown_rej_rate = unk_rej / unk_tot if unk_tot else 0.0

    k_acc = round(known_acc_rate * 100, 2)
    u_rej = round(unknown_rej_rate * 100, 2)
    bal = round(
        0.5 * (known_acc_rate + unknown_rej_rate) * 100,
        2,
    )

    metrics = {
        "threshold": threshold,
        "known_total": known_tot,
        "known_correct": known_corr,
        "known_wrong_id": known_wrong,
        "known_as_unknown": known_unk,
        "known_no_face": known_nf,
        "known_acc_pct": k_acc,
        "unknown_total": unk_tot,
        "unknown_rejected": unk_rej,
        "unknown_accepted": unk_acc,
        "unknown_no_face": unk_nf,
        "unknown_rej_pct": u_rej,
        "balanced_acc_pct": bal,
    }
    return metrics, pd.DataFrame(pred_records)


def main():
    faces_root = Path("data/november/faces_new/faces")
    enroll_crops, val_items, test_items = extract_all_wanas_crops(
        faces_root
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    eval_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_sum = pd.read_csv(
        "data/face_recognition/trained_models/"
        "training_summary_F1_F4.csv"
    )
    eval_out_dir = Path(
        "data/face_recognition/wanas_evaluation_F1_F4"
    )
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    sample_crop = next(
        iter(next(iter(enroll_crops.values())).values())
    )
    sample_tensor = (
        eval_tf(sample_crop).unsqueeze(0).to(device)
    )

    for trow in train_sum.itertuples():
        exp_id = trow.exp_id
        ckpt = torch.load(
            trow.checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model = FaceEmbeddingNet(
            ckpt["arch"],
            num_classes=ckpt["num_classes"],
            emb_dim=ckpt["emb_dim"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # This measures embedding inference only.
        with torch.no_grad():
            for _ in range(5):
                model.extract_embedding(sample_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                model.extract_embedding(sample_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                times.append((time.perf_counter() - t0) * 1000)

            gpu_peak_mb = (
                round(
                    torch.cuda.max_memory_allocated(device)
                    / (1024 * 1024),
                    2,
                )
                if device.type == "cuda"
                else 0.0
            )

        med_emb_ms = round(float(np.median(times)), 2)

        def embed_crop(pil_img):
            x = eval_tf(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                return model.extract_embedding(x).squeeze(0).cpu().numpy()

        gallery_ids = sorted(enroll_crops.keys())
        gallery_mat = []

        for pid in gallery_ids:
            embs = [
                embed_crop(crop)
                for _, crop in sorted(enroll_crops[pid].items())
            ]
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            gallery_mat.append(mean_emb)

        gallery_mat = np.stack(gallery_mat, axis=0)

        def score_split(items_dict):
            rows = []

            for img_name, info in items_dict.items():
                if info["crop"] is None:
                    rows.append({
                        "image": img_name,
                        "role": info["role"],
                        "expected": info["expected"],
                        "top1_id": None,
                        "score": None,
                        "status": "no_face",
                    })
                    continue

                query_emb = embed_crop(info["crop"])
                similarities = gallery_mat @ query_emb
                best_idx = int(np.argmax(similarities))

                rows.append({
                    "image": img_name,
                    "role": info["role"],
                    "expected": info["expected"],
                    "top1_id": gallery_ids[best_idx],
                    "score": float(similarities[best_idx]),
                    "status": "scored",
                })

            return rows

        val_scored = score_split(val_items)
        test_scored = score_split(test_items)

        best_th = tune_threshold_on_val(val_scored)
        val_metrics, val_df = evaluate_split(val_scored, best_th)
        test_metrics, test_df = evaluate_split(test_scored, best_th)

        val_df.to_csv(
            eval_out_dir / f"{exp_id}_val_predictions.csv",
            index=False,
        )
        test_df.to_csv(
            eval_out_dir / f"{exp_id}_test_predictions.csv",
            index=False,
        )

        comparison_rows.append({
            "exp_id": exp_id,
            "model_name": trow.model_name,
            "params_m": round(trow.params / 1e6, 2),
            "size_mb": trow.checkpoint_mb,
            "lfw_test_acc_pct": trow.lfw_test_acc_pct,
            "wanas_val_threshold": best_th,
            "wanas_val_bal_acc_pct": val_metrics["balanced_acc_pct"],
            "wanas_test_known_acc_pct": test_metrics["known_acc_pct"],
            "wanas_test_unknown_rej_pct": test_metrics["unknown_rej_pct"],
            "wanas_test_bal_acc_pct": test_metrics["balanced_acc_pct"],
            "wanas_test_security_errors": (
                test_metrics["known_wrong_id"]
                + test_metrics["unknown_accepted"]
            ),
            "embedding_only_latency_ms": med_emb_ms,
            "torch_gpu_peak_mb": gpu_peak_mb,
        })

    ins_val = pd.read_csv(
        "data/november/validation_results/summary.csv"
    )
    ins_test = pd.read_csv(
        "data/november/test_results/summary.csv"
    )
    test_predictions = pd.read_csv(
        "data/november/test_results/predictions.csv"
    )

    for _, test_row in ins_test[
        ins_test["enrollment"] == "two_image_mean"
    ].iterrows():
        model_name = test_row["model"]

        validation_matches = ins_val[
            (ins_val["model"] == model_name)
            & (ins_val["enrollment"] == "two_image_mean")
        ]
        if len(validation_matches) != 1:
            raise ValueError(
                f"Expected one validation summary row for {model_name} "
                f"with two_image_mean enrollment, "
                f"found {len(validation_matches)}."
            )

        validation_row = validation_matches.iloc[0]

        model_test_predictions = test_predictions[
            (test_predictions["model"] == model_name)
            & (test_predictions["enrollment"] == "two_image_mean")
        ]
        known_total = (
            model_test_predictions["role"] == "known_probe"
        ).sum()
        unknown_total = (
            model_test_predictions["role"] == "unknown_probe"
        ).sum()
        if known_total == 0 or unknown_total == 0:
            raise ValueError(
                f"Missing test predictions for {model_name}."
            )
        known_acc_rate = (
            float(test_row["known_correct"]) / known_total
        )
        unknown_rej_rate = (
            float(test_row["unknown_rejected"]) / unknown_total
        )
        test_balanced_accuracy = round(
            0.5 * (known_acc_rate + unknown_rej_rate) * 100,
            2,
        )

        comparison_rows.append({
            "exp_id": f"Baseline ({model_name})",
            "model_name": (
                f"InsightFace {model_name} (two_image_mean)"
            ),
            "params_m": (
                24.9 if model_name == "buffalo_l" else 13.0
            ),
            "size_mb": (
                275.0 if model_name == "buffalo_l" else 159.0
            ),
            "lfw_test_acc_pct": None,
            "wanas_val_threshold": float(validation_row["threshold"]),
            "wanas_val_bal_acc_pct": (
                float(validation_row["balanced_accuracy"]) * 100
            ),
            "wanas_test_known_acc_pct": round(
                known_acc_rate * 100,
                2,
            ),
            "wanas_test_unknown_rej_pct": round(
                unknown_rej_rate * 100,
                2,
            ),
            "wanas_test_bal_acc_pct": test_balanced_accuracy,
            "wanas_test_security_errors": int(
                test_row["known_wrong_id"]
                + test_row["unknown_accepted"]
            ),
            "pipeline_latency_ms": (
                42.74 if model_name == "buffalo_s" else 44.10
            ),
            "python_tracemalloc_peak_mb": 137.01,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(
        eval_out_dir / "full_comparison_F1_F4_vs_InsightFace.csv",
        index=False,
    )

    print(
        "\n=== Unified Benchmark: F1-F4 vs InsightFace "
        "on WANAS Held-Out Test Set ==="
    )
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()