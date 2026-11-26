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
    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = 0.10 * box_width
    margin_y = 0.10 * box_height

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

    detector_app = FaceAnalysis(
        name="buffalo_s",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector_app.prepare(ctx_id=0, det_size=(640, 640))

    # Enrollment crops use the audited buffalo_s boxes.
    audit_df = pd.read_csv("data/november/enrollment_face_audit.csv")
    audit_rows = audit_df[audit_df["model"] == "buffalo_s"].copy()

    enrollment_crops = {}

    for row in audit_rows.itertuples():
        image_name = row.image
        person_id = image_name.split("_")[0]
        image = cv2.imread(
            str(faces_root / person_id / image_name)
        )
        box = json.loads(row.selected_box)

        enrollment_crops.setdefault(person_id, {})[image_name] = (
            crop_with_margin(image, box)
        )

    def get_split_crops(prediction_path: Path):
        predictions = pd.read_csv(prediction_path)
        selected_rows = predictions[
            (predictions["model"] == "buffalo_s")
            & (predictions["enrollment"] == "two_image_mean")
        ].copy()

        records = {}

        for row in selected_rows.itertuples():
            image_name = row.image
            person_folder = image_name.split("_")[0]

            if row.status == "no_face":
                records[image_name] = {
                    "crop": None,
                    "role": row.role,
                    "expected": row.expected,
                }
                continue

            image = cv2.imread(
                str(faces_root / person_folder / image_name)
            )
            faces = detector_app.get(image)

            valid_faces = [
                face for face in faces
                if (face.bbox[2] - face.bbox[0]) >= 20
                and (face.bbox[3] - face.bbox[1]) >= 20
            ]

            if not valid_faces:
                records[image_name] = {
                    "crop": None,
                    "role": row.role,
                    "expected": row.expected,
                }
                continue

            best_face = max(
                valid_faces,
                key=lambda face: float(face.det_score)
            )

            records[image_name] = {
                "crop": crop_with_margin(image, best_face.bbox),
                "role": row.role,
                "expected": row.expected,
            }

        return records

    validation_items = get_split_crops(
        Path("data/november/validation_results/predictions.csv")
    )
    test_items = get_split_crops(
        Path("data/november/test_results/predictions.csv")
    )

    del detector_app

    return enrollment_crops, validation_items, test_items


def tune_threshold_on_val(validation_rows):
    best_threshold = 0.50
    best_key = (-1, -1.0, -1.0)

    for threshold in np.round(np.arange(0.10, 0.96, 0.01), 2):
        known_total = 0
        known_correct = 0
        known_wrong = 0
        unknown_total = 0
        unknown_rejected = 0
        unknown_accepted = 0

        for row in validation_rows:
            if row["status"] == "no_face":
                if row["role"] == "known_probe":
                    known_total += 1
                else:
                    unknown_total += 1
                continue

            prediction = (
                row["top1_id"]
                if row["score"] >= threshold
                else "unknown"
            )

            if row["role"] == "known_probe":
                known_total += 1

                if prediction == row["expected"]:
                    known_correct += 1
                elif prediction == "unknown":
                    pass
                else:
                    known_wrong += 1
            else:
                unknown_total += 1

                if prediction == "unknown":
                    unknown_rejected += 1
                else:
                    unknown_accepted += 1

        security_errors = known_wrong + unknown_accepted

        known_accuracy = (
            known_correct / known_total if known_total else 0.0
        )
        unknown_rejection = (
            unknown_rejected / unknown_total if unknown_total else 0.0
        )
        balanced_accuracy = 0.5 * (
            known_accuracy + unknown_rejection
        )

        score_key = (
            1 if security_errors == 0 else 0,
            round(balanced_accuracy, 6),
            round(known_accuracy, 6),
        )

        if score_key > best_key:
            best_key = score_key
            best_threshold = float(threshold)

    return best_threshold


def evaluate_split(rows, threshold):
    known_total = 0
    known_correct = 0
    known_wrong = 0
    known_as_unknown = 0
    known_no_face = 0

    unknown_total = 0
    unknown_rejected = 0
    unknown_accepted = 0
    unknown_no_face = 0

    prediction_records = []

    for row in rows:
        if row["status"] == "no_face":
            decision = "no_face"

            if row["role"] == "known_probe":
                known_total += 1
                known_no_face += 1
            else:
                unknown_total += 1
                unknown_no_face += 1
        else:
            decision = (
                row["top1_id"]
                if row["score"] >= threshold
                else "unknown"
            )

            if row["role"] == "known_probe":
                known_total += 1

                if decision == row["expected"]:
                    known_correct += 1
                elif decision == "unknown":
                    known_as_unknown += 1
                else:
                    known_wrong += 1
            else:
                unknown_total += 1

                if decision == "unknown":
                    unknown_rejected += 1
                else:
                    unknown_accepted += 1

        prediction_records.append({
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

    known_accuracy = round(
        known_correct / known_total * 100,
        2
    )
    unknown_rejection = round(
        unknown_rejected / unknown_total * 100,
        2
    )
    balanced_accuracy = round(
        0.5 * (known_accuracy + unknown_rejection),
        2
    )

    metrics = {
        "threshold": threshold,
        "known_total": known_total,
        "known_correct": known_correct,
        "known_wrong_id": known_wrong,
        "known_as_unknown": known_as_unknown,
        "known_no_face": known_no_face,
        "known_acc_pct": known_accuracy,
        "unknown_total": unknown_total,
        "unknown_rejected": unknown_rejected,
        "unknown_accepted": unknown_accepted,
        "unknown_no_face": unknown_no_face,
        "unknown_rej_pct": unknown_rejection,
        "balanced_acc_pct": balanced_accuracy,
    }

    return metrics, pd.DataFrame(prediction_records)


def main():
    faces_root = Path("data/november/faces_new/faces")

    enrollment_crops, validation_items, test_items = (
        extract_all_wanas_crops(faces_root)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluation_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    training_summary = pd.read_csv(
        "data/face_recognition/trained_models/"
        "training_summary_F1_F4.csv"
    )

    output_dir = Path(
        "data/face_recognition/wanas_evaluation_F1_F4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    sample_crop = next(
        iter(next(iter(enrollment_crops.values())).values())
    )
    sample_tensor = (
        evaluation_transform(sample_crop)
        .unsqueeze(0)
        .to(device)
    )

    for training_row in training_summary.itertuples():
        experiment_id = training_row.exp_id

        checkpoint = torch.load(
            training_row.checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model = FaceEmbeddingNet(
            checkpoint["arch"],
            num_classes=checkpoint["num_classes"],
            emb_dim=checkpoint["emb_dim"],
        ).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # This measures embedding-network inference only.
        with torch.no_grad():
            for _ in range(5):
                model.extract_embedding(sample_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            embedding_times = []

            for _ in range(50):
                start_time = time.perf_counter()
                model.extract_embedding(sample_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                embedding_times.append(
                    (time.perf_counter() - start_time) * 1000
                )

            torch_gpu_peak_mb = (
                round(
                    torch.cuda.max_memory_allocated(device)
                    / (1024 * 1024),
                    2,
                )
                if device.type == "cuda"
                else 0.0
            )

        embedding_only_latency_ms = round(
            float(np.median(embedding_times)),
            2,
        )

        def embed_crop(pil_image):
            tensor = (
                evaluation_transform(pil_image)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                embedding = model.extract_embedding(tensor)

            return embedding.squeeze(0).cpu().numpy()

        gallery_ids = sorted(enrollment_crops.keys())
        gallery_embeddings = []

        for person_id in gallery_ids:
            person_embeddings = [
                embed_crop(crop)
                for _, crop in sorted(
                    enrollment_crops[person_id].items()
                )
            ]

            mean_embedding = np.mean(person_embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(
                mean_embedding
            )
            gallery_embeddings.append(mean_embedding)

        gallery_matrix = np.stack(gallery_embeddings, axis=0)

        def score_split(items):
            scored_rows = []

            for image_name, info in items.items():
                if info["crop"] is None:
                    scored_rows.append({
                        "image": image_name,
                        "role": info["role"],
                        "expected": info["expected"],
                        "top1_id": None,
                        "score": None,
                        "status": "no_face",
                    })
                    continue

                query_embedding = embed_crop(info["crop"])
                similarities = gallery_matrix @ query_embedding
                best_index = int(np.argmax(similarities))

                scored_rows.append({
                    "image": image_name,
                    "role": info["role"],
                    "expected": info["expected"],
                    "top1_id": gallery_ids[best_index],
                    "score": float(similarities[best_index]),
                    "status": "scored",
                })

            return scored_rows

        validation_scored = score_split(validation_items)
        test_scored = score_split(test_items)

        best_threshold = tune_threshold_on_val(validation_scored)

        validation_metrics, validation_df = evaluate_split(
            validation_scored,
            best_threshold,
        )
        test_metrics, test_df = evaluate_split(
            test_scored,
            best_threshold,
        )

        validation_df.to_csv(
            output_dir / f"{experiment_id}_val_predictions.csv",
            index=False,
        )
        test_df.to_csv(
            output_dir / f"{experiment_id}_test_predictions.csv",
            index=False,
        )

        comparison_rows.append({
            "exp_id": experiment_id,
            "model_name": training_row.model_name,
            "params_m": round(training_row.params / 1e6, 2),
            "size_mb": training_row.checkpoint_mb,
            "lfw_test_acc_pct": training_row.lfw_test_acc_pct,
            "wanas_val_threshold": best_threshold,
            "wanas_val_bal_acc_pct": (
                validation_metrics["balanced_acc_pct"]
            ),
            "wanas_test_known_acc_pct": (
                test_metrics["known_acc_pct"]
            ),
            "wanas_test_unknown_rej_pct": (
                test_metrics["unknown_rej_pct"]
            ),
            "wanas_test_bal_acc_pct": (
                test_metrics["balanced_acc_pct"]
            ),
            "wanas_test_security_errors": (
                test_metrics["known_wrong_id"]
                + test_metrics["unknown_accepted"]
            ),
            "embedding_only_latency_ms": (
                embedding_only_latency_ms
            ),
            "torch_gpu_peak_mb": torch_gpu_peak_mb,
        })

    validation_summary = pd.read_csv(
        "data/november/validation_results/summary.csv"
    )
    test_summary = pd.read_csv(
        "data/november/test_results/summary.csv"
    )

    test_known_column = next(
        column for column in test_summary.columns
        if "known" in column and "acc" in column
    )
    test_unknown_column = next(
        column for column in test_summary.columns
        if "unknown" in column and "rej" in column
    )
    test_balanced_column = next(
        column for column in test_summary.columns
        if "bal" in column
    )

    for _, test_row in test_summary[
        test_summary["enrollment"] == "two_image_mean"
    ].iterrows():
        model_name = test_row["model"]

        validation_matches = validation_summary[
            (validation_summary["model"] == model_name)
            & (
                validation_summary["enrollment"]
                == "two_image_mean"
            )
        ]

        if len(validation_matches) != 1:
            raise ValueError(
                f"Expected one validation summary row for {model_name} "
                f"with two_image_mean enrollment, "
                f"found {len(validation_matches)}."
            )

        validation_row = validation_matches.iloc[0]

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
            "wanas_val_threshold": float(
                validation_row["threshold"]
            ),
            "wanas_val_bal_acc_pct": (
                float(validation_row["balanced_accuracy"]) * 100
            ),
            "wanas_test_known_acc_pct": test_row[
                test_known_column
            ],
            "wanas_test_unknown_rej_pct": test_row[
                test_unknown_column
            ],
            "wanas_test_bal_acc_pct": test_row[
                test_balanced_column
            ],
            "wanas_test_security_errors": int(
                test_row["known_wrong_id"]
                + test_row["unknown_accepted"]
            ),
            "pipeline_latency_ms": (
                42.74 if model_name == "buffalo_s" else 44.10
            ),
            "python_tracemalloc_peak_mb": 137.01,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = (
        output_dir
        / "full_comparison_F1_F4_vs_InsightFace.csv"
    )
    comparison_df.to_csv(comparison_path, index=False)

    print(
        "\n=== Unified Benchmark: F1-F4 vs InsightFace "
        "on WANAS Held-Out Test Set ==="
    )
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison to: {comparison_path}")


if __name__ == "__main__":
    main()