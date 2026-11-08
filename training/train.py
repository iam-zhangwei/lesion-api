"""Small CPU-friendly transfer-learning run, logged to MLflow.

Expects an ImageFolder layout:  data/train/<class>/*.jpg  and  data/val/<class>/*.jpg
Example (fast, CPU):
  python training/train.py --data-dir data --epochs 3 --subset 2000 --freeze-backbone
"""
import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--subset", type=int, default=0, help="use only N training images (0 = all)")
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--freeze-backbone", action="store_true")
p.add_argument("--out", default="models/model.pt")
args = p.parse_args()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = transforms.Compose(
    [
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        norm,
    ]
)
train_ds = datasets.ImageFolder(Path(args.data_dir) / "train", tf)
val_ds = datasets.ImageFolder(Path(args.data_dir) / "val", tf)
if args.subset:
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_ds), generator=g)[: args.subset].tolist()
    train_ds = Subset(train_ds, idx)
classes = val_ds.classes

train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, args.batch_size)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
if args.freeze_backbone:
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith("fc")
opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


mlflow.set_experiment("lesion-classifier")
with mlflow.start_run():
    mlflow.log_params({**vars(args), "model": "resnet18", "n_classes": len(classes)})
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        train_loss = running / len(train_dl.dataset)
        val_acc = evaluate()
        mlflow.log_metrics({"train_loss": train_loss, "val_accuracy": val_acc}, step=epoch)
        print(f"epoch {epoch + 1}/{args.epochs} loss={train_loss:.4f} val_acc={val_acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out)
    mlflow.log_artifact(str(out))
    print(f"Saved {out}")
