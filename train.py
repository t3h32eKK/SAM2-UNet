#!/usr/bin/env python3
import os, argparse, random, numpy as np, torch
import torch.optim as opt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet


# ───────────────────────────── Loss: Focal-Tversky ──────────────────────────
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


# ───────────────────────────── CLI args ─────────────────────────────────────
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path",        required=True)
parser.add_argument("--train_image_path",  required=True)
parser.add_argument("--train_mask_path",   required=True)
parser.add_argument("--save_path",         required=True)
parser.add_argument("--epoch", type=int,   default=40)
parser.add_argument("--lr",    type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=1e-4)
args = parser.parse_args()


# ───────────────────────────── main train loop ──────────────────────────────
def main(cfg):
    # dataset @ 512²
    ds = FullDataset(cfg.train_image_path, cfg.train_mask_path, 512, mode="train")
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=min(4, os.cpu_count()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAM2-UNet with deep supervision
    model = SAM2UNet(checkpoint_path=cfg.hiera_path, deep_sup=True).to(device)

    loss_fn = FocalTversky()                      # ← new loss
    opti    = opt.AdamW(model.parameters(),
                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched   = CosineAnnealingLR(opti, cfg.epoch, eta_min=1e-6)

    os.makedirs(cfg.save_path, exist_ok=True)

    λ_main, λ_side2, λ_side1 = 1.0, 0.3, 0.1     # loss weights

    for ep in range(cfg.epoch):
        model.train()
        for step, batch in enumerate(loader):
            img   = batch["image"].to(device)
            mask  = batch["label"].to(device)

            opti.zero_grad()
            pred, side2, side1 = model(img)

            loss  = (λ_main  * loss_fn(pred,  mask)
                   + λ_side2 * loss_fn(side2, mask)
                   + λ_side1 * loss_fn(side1, mask))
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
