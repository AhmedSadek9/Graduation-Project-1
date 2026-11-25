import json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FaceCropDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["crop_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class CosineMarginHead(nn.Module):
    def __init__(self, emb_dim, num_classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, emb):
        w_norm = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(emb, w_norm)

class FaceEmbeddingNet(nn.Module):
    def __init__(self, arch: str, pretrained: bool, num_classes: int, emb_dim: int = 512):
        super().__init__()
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        self.backbone = backbone
        self.emb_head = nn.Sequential(
            nn.Linear(in_features, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.classifier = CosineMarginHead(emb_dim, num_classes, scale=16.0)

    def extract_embedding(self, x):
        feat = self.backbone(x)
        emb = self.emb_head(feat)
        return F.normalize(emb, dim=1)

    def forward(self, x):
        emb = self.extract_embedding(x)
        logits = self.classifier(emb)
        return logits, emb

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training F1-F4 on device: {device}")

    df = pd.read_csv("data/face_recognition/public_dataset/lfw_prepared_splits.csv")
    num_classes = int(df["label"].nunique())

    train_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FaceCropDataset(df[df["split"] == "train"], transform=train_tf)
    val_ds = FaceCropDataset(df[df["split"] == "val"], transform=eval_tf)
    test_ds = FaceCropDataset(df[df["split"] == "test"], transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    experiments = [
        {"exp_id": "F1", "name": "ResNet-18 (Pretrained)", "arch": "resnet18", "pretrained": True, "lr": 1e-3, "epochs": 25},
        {"exp_id": "F2", "name": "MobileNetV3-Small (Pretrained)", "arch": "mobilenet_v3_small", "pretrained": True, "lr": 1.5e-3, "epochs": 25},
        {"exp_id": "F3", "name": "ResNet-18 (Scratch)", "arch": "resnet18", "pretrained": False, "lr": 2e-3, "epochs": 30},
        {"exp_id": "F4", "name": "MobileNetV3-Small (Scratch)", "arch": "mobilenet_v3_small", "pretrained": False, "lr": 2.5e-3, "epochs": 30},
    ]

    out_dir = Path("data/face_recognition/trained_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for exp in experiments:
        seed_everything(42)
        exp_id = exp["exp_id"]
        print(f"\n=== [{exp_id}] {exp['name']} ===")
        model = FaceEmbeddingNet(exp["arch"], exp["pretrained"], num_classes=num_classes, emb_dim=512).to(device)
        total_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp["epochs"])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = -1.0
        best_ckpt = out_dir / f"{exp_id}_{exp['arch']}_{'pretrained' if exp['pretrained'] else 'scratch'}.pt"
        t_start = time.perf_counter()

        for epoch in range(1, exp["epochs"] + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += imgs.size(0)

            scheduler.step()

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits, _ = model(imgs)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += imgs.size(0)

            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "exp_id": exp_id,
                    "name": exp["name"],
                    "arch": exp["arch"],
                    "pretrained": exp["pretrained"],
                    "num_classes": num_classes,
                    "emb_dim": 512,
                    "state_dict": model.state_dict(),
                }, best_ckpt)

            if epoch % 5 == 0 or epoch == exp["epochs"]:
                print(f"  Epoch {epoch:02d}/{exp['epochs']} | Train Acc: {train_correct/train_total*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)", flush=True)

        train_time_sec = round(time.perf_counter() - t_start, 2)

        # Evaluate best checkpoint on internal LFW test split
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += imgs.size(0)

        lfw_test_acc = round(test_correct / test_total * 100, 2)
        model_size_mb = round(best_ckpt.stat().st_size / (1024 * 1024), 2)

        row = {
            "exp_id": exp_id,
            "model_name": exp["name"],
            "arch": exp["arch"],
            "pretrained": exp["pretrained"],
            "epochs": exp["epochs"],
            "lr": exp["lr"],
            "params": total_params,
            "checkpoint_mb": model_size_mb,
            "train_time_sec": train_time_sec,
            "lfw_best_val_acc_pct": round(best_val_acc * 100, 2),
            "lfw_test_acc_pct": lfw_test_acc,
            "checkpoint_path": str(best_ckpt),
        }
        results.append(row)
        print(f"  -> Saved {best_ckpt.name} ({model_size_mb} MB) | LFW Test Acc: {lfw_test_acc}% | Time: {train_time_sec}s")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "training_summary_F1_F4.csv", index=False)
    print("\n=== Training Summary (F1-F4) ===")
    print(summary_df[["exp_id", "model_name", "params", "checkpoint_mb", "train_time_sec", "lfw_best_val_acc_pct", "lfw_test_acc_pct"]].to_string(index=False))

if __name__ == "__main__":
    main()
