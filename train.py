#!/usr/bin/env python3
import os, argparse, random, numpy as np, torch
import torch.optim as opt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet

# ───────────────────────────── Loss functions ──────────────────────────────
class FocalTversky(nn.Module):
    """Focal-Tversky loss: good for tiny, imbalanced structures."""
    def __init__(self, α=0.3, β=0.7, γ=0.75, eps=1e-6):
        super().__init__()
        self.α, self.β, self.γ, self.eps = α, β, γ, eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        tp = (prob * target).sum((2, 3))
        fp = (prob * (1 - target)).sum((2, 3))
        fn = ((1 - prob) * target).sum((2, 3))
        tv = (tp + self.eps) / (tp + self.α * fp + self.β * fn + self.eps)
        return ((1 - tv) ** self.γ).mean()

bce_fn = nn.BCEWithLogitsLoss()

def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((2, 3))
    union = prob.sum((2, 3)) + target.sum((2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()

# combine Focal-Tversky + BCE + Dice

def combined_loss_fn(logits, target, focal):
    return (focal(logits, target)
            + 0.5 * bce_fn(logits, target)
            + 0.5 * dice_loss(logits, target))

# ───────────────────────────── CLI args ─────────────────────────────────────
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path",       required=True)
parser.add_argument("--train_image_path", required=True)
parser.add_argument("--train_mask_path",  required=True)
parser.add_argument("--save_path",        required=True)
parser.add_argument("--epoch", type=int,   default=40)
parser.add_argument("--lr",    type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=1e-4)
args = parser.parse_args()

# ───────────────────────────── main train loop ──────────────────────────────
def main(cfg):
    # dataset @ 512² tiles
    ds = FullDataset(cfg.train_image_path, cfg.train_mask_path, 512, mode="train")
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=min(4, os.cpu_count()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAM2-UNet with deep supervision
    model = SAM2UNet(checkpoint_path=cfg.hiera_path, deep_sup=True).to(device)

    focal = FocalTversky()
    opti  = opt.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # warmup + cosine scheduler
    warmup_epochs = 5
    cosine_epochs = cfg.epoch - warmup_epochs
    sched = SequentialLR(
        opti,
        schedulers=[
            LinearLR(opti, start_factor=1e-3, total_iters=warmup_epochs),
            CosineAnnealingLR(opti, T_max=cosine_epochs, eta_min=1e-6)
        ],
        milestones=[warmup_epochs]
    )

    os.makedirs(cfg.save_path, exist_ok=True)

    λ_main, λ_side2, λ_side1 = 1.0, 0.3, 0.1

    for ep in range(cfg.epoch):
        model.train()
        # unfreeze last two ViT blocks after 10 epochs
        if ep == 10:
            for p in list(model.encoder.blocks)[-2:].parameters():
                p.requires_grad = True
            print("⚡ Unfroze last two ViT blocks for fine‑tuning")

        for step, batch in enumerate(loader):
            img  = batch["image"].to(device)
            mask = batch["label"].to(device)

            opti.zero_grad()
            pred, side2, side1 = model(img)

            loss_main  = combined_loss_fn(pred,  mask, focal)
            loss_side2 = combined_loss_fn(side2, mask, focal)
            loss_side1 = combined_loss_fn(side1, mask, focal)
            loss = (λ_main  * loss_main
                  + λ_side2 * loss_side2
                  + λ_side1 * loss_side1)
            loss.backward()
            opti.step()

            if step % 50 == 0:
                print(f"Epoch {ep+1}/{cfg.epoch}  iter {step:03d}  loss {loss.item():.4f}")

        sched.step()

        if (ep + 1) % 5 == 0 or (ep + 1) == cfg.epoch:
            ckpt = os.path.join(cfg.save_path, f"SAM2-UNet-{ep+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print("✔ saved", ckpt)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0); np.random.seed(0)
    main(args)
