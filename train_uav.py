from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    if os.environ.get("UAV123_UV_REEXEC"):
        raise SystemExit("PyTorch is missing from the uv environment; run `uv sync` first") from error
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("PyTorch is missing; install uv, then run `uv sync`") from error
    env = {**os.environ, "UAV123_UV_REEXEC": "1"}
    os.execvpe(uv, [uv, "run", "train-uav", *sys.argv[1:]], env)
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

from uav_data import CLASSES, FRAMES, box_for_frame, crop_target, sequences
from uav_model import create_model


class TargetCrops(Dataset):
    def __init__(self, samples, training=False):
        self.samples, self.training = samples, training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, box, label = self.samples[index]
        with Image.open(path) as source:
            image = crop_target(source.convert("RGB"), box)
        if self.training and random.random() < 0.5:
            image = ImageOps.mirror(image)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, label


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on UAV123 annotated target crops")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth annotated frame")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.stride < 1:
        parser.error("--epochs and --stride must be positive")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    records = sequences()
    records_by_class = {}
    for record in records:
        label = record.name.split("_")[0].rstrip("0123456789")
        records_by_class.setdefault(label, []).append(record)
    validation_folders, validation_names = set(), set()
    for label, class_records in records_by_class.items():
        folders = sorted({record.folder for record in class_records})
        if len(folders) > 1:
            validation_folders.add((label, folders[-1]))
        elif len(class_records) > 1:
            validation_names.add(sorted(class_records, key=lambda record: record.name)[-1].name)

    train_samples, validation_samples = [], []
    for record in records:
        label = record.name.split("_")[0].rstrip("0123456789")
        if label not in CLASSES:
            continue
        for offset in range(0, len(record.boxes), args.stride):
            box = record.boxes[offset]
            frame_number = record.start + offset
            path = FRAMES / record.folder / f"{frame_number:06d}.jpg"
            if box is None or not path.is_file() or min(box[2:]) <= 2:
                continue
            sample = (path, box, CLASSES.index(label))
            held_out = (label, record.folder) in validation_folders or record.name in validation_names
            bucket = validation_samples if held_out else train_samples
            bucket.append(sample)
    if not train_samples or not validation_samples:
        raise SystemExit("Could not make train/validation samples from UAV123 annotations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_counts = np.bincount([sample[2] for sample in train_samples], minlength=len(CLASSES))
    sampler = WeightedRandomSampler([1 / train_counts[sample[2]] for sample in train_samples], len(train_samples), replacement=True)
    train_loader = DataLoader(TargetCrops(train_samples, True), batch_size=args.batch_size, sampler=sampler, num_workers=4)
    validation_loader = DataLoader(TargetCrops(validation_samples), batch_size=args.batch_size, num_workers=4)
    model = create_model(pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for layer in model.features[14:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    criterion = torch.nn.CrossEntropyLoss()
    validation_counts = np.bincount([sample[2] for sample in validation_samples], minlength=len(CLASSES))
    print(f"validation majority-class baseline: {validation_counts.max() / validation_counts.sum():.1%}")
    best_balanced_accuracy = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = seen = 0
        correct_by_class = np.zeros(len(CLASSES), dtype=np.int64)
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(1).cpu()
                correct += int((predictions == labels).sum())
                seen += len(labels)
                for label in labels.unique():
                    mask = labels == label
                    correct_by_class[label] += int((predictions[mask] == labels[mask]).sum())
        present = validation_counts > 0
        balanced_accuracy = np.mean(correct_by_class[present] / validation_counts[present])
        print(f"epoch {epoch + 1}/{args.epochs}: loss={total_loss / len(train_loader):.4f}, accuracy={correct / seen:.1%}, balanced accuracy={balanced_accuracy:.1%}")
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    output = Path(__file__).resolve().parent / "models/uav123_mobilenetv2.pth"
    output.parent.mkdir(exist_ok=True)
    torch.save({"classes": CLASSES, "state_dict": best_state}, output)
    print(f"Saved best checkpoint ({best_balanced_accuracy:.1%} balanced accuracy) to {output} using {len(train_samples)} train and {len(validation_samples)} held-out crops on {device}")


if __name__ == "__main__":
    main()
