import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from data import VideoDataset, emotion_names
from model import CNNLSTM


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all = []
    labels_all = []
    for frames, labels in tqdm(loader, desc="train", leave=False):
        frames = frames.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        preds_all.extend(preds)
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="val", leave=False):
            frames = frames.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            preds_all.extend(preds)
            labels_all.extend(labels.detach().cpu().numpy().tolist())
    acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM FER on RAVDESS")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory of RAVDESS videos")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--face-crop", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = VideoDataset(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        face_crop=args.face_crop,
        grayscale=args.grayscale,
    )

    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = CNNLSTM(
        num_classes=8,
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.3,
        pretrained_cnn=False,
        freeze_cnn=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "emotion_names": emotion_names(),
                "config": {
                    "num_frames": args.num_frames,
                    "image_size": args.image_size,
                    "grayscale": args.grayscale,
                }
            }, best_path)
            print(f"Saved new best model to: {best_path} (val_acc={best_val_acc:.4f})")

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()