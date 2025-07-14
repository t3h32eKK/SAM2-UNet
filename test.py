#!/usr/bin/env python3
import os, argparse, numpy as np, torch
import torch.nn.functional as F
import imageio
from SAM2UNet import SAM2UNet
from dataset import TestDataset

# ───────────────────────────── CLI args ─────────────────────────────────────
parser = argparse.ArgumentParser("SAM2-UNet Test")
parser.add_argument("--checkpoint",     type=str, required=True,
                    help="path to your trained SAM2-UNet weights (.pth)")
parser.add_argument("--test_image_path",type=str, required=True,
                    help="directory of test images")
parser.add_argument("--test_gt_path",   type=str, required=True,
                    help="directory of test masks (for naming)")
parser.add_argument("--save_path",      type=str, required=True,
                    help="directory to save predicted masks")
args = parser.parse_args()



# ────────────────────────────── setup device & model ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()

# ────────────────────────────── prepare dataloader ───────────────────────────────
test_loader = TestDataset(args.test_image_path, args.test_gt_path, size=512)
os.makedirs(args.save_path, exist_ok=True)

# ─────────────────────────────── inference & stitching ───────────────────────────
# number of original images
n_images = len(test_loader) // 16
for img_idx in range(n_images):
    # collect per-tile predictions
    tile_preds = []
    base_name = None
    for _ in range(16):
        img_tile, _, name = test_loader.load_data()
        base_name = name
        with torch.no_grad():
            pred, _, _ = model(img_tile.to(device))
            pred = F.interpolate(pred,
                                 size=(512,512),
                                 mode='bilinear', align_corners=False)
            prob = pred.sigmoid().cpu().numpy().squeeze()
        tile_preds.append(prob)

    # stitch to 2048×2048
    canvas = np.zeros((2048,2048), dtype=np.float32)
    count  = np.zeros_like(canvas)
    for idx, p in enumerate(tile_preds):
        row, col = divmod(idx, 4)
        y, x = row*512, col*512
        canvas[y:y+512, x:x+512] += p
        count [y:y+512, x:x+512] += 1
    canvas /= count

    # downsample to original 512×512
    canvas_tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
    small = F.interpolate(canvas_tensor, size=(512,512),
                          mode='bilinear', align_corners=False)
    out = small.numpy().squeeze()
    # normalize to [0,255]
    out = (out - out.min())/(out.max()-out.min()+1e-8)
    out = (out*255).astype(np.uint8)

    # save
    save_name = os.path.join(args.save_path, base_name)
    imageio.imsave(save_name, out)
    print(f"Saved {save_name}")
