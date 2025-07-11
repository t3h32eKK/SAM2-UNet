import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

"""
Re‑implemented decoder for SAM2‑UNet
-----------------------------------
*   **Gradual up‑sampling**: x4 → x3 → x2 → x1 → full‑res (no single 16 × jump).
*   **Deep supervision (optional)**: side‑outputs are kept, up‑sampled to the input
    size, but can be **disabled** at loss time by setting their loss weight = 0.
*   **Resolution‑agnostic**: final `F.interpolate` uses the *original* H × W of the
    input tensor instead of a hard‑coded (512, 512).
*   **`up4` now used**: final 2 × up‑sampling stage so the decoder path is
    32 → 64 → 128 → 256 → 512 (for a 512² input).
"""


class DoubleConv(nn.Module):
    """(Conv→BN→ReLU) × 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample by 2 and fuse with skip."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x_low, x_skip=None):
        x = self.upsample(x_low)
        if x_skip is not None:  # skip connection present
            # pad if necessary (odd sizes)
            dy = x_skip.size(2) - x.size(2)
            dx = x_skip.size(3) - x.size(3)
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    """Prompt‑tuning adapter on each ViT block."""
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        dim = blk.attn.qkv.in_features
        self.prompt = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim))

    def forward(self, x):
        return self.blk(x + self.prompt(x))


# --- backbone helpers (unchanged) -----------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, **kw):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, bias=False, **kw),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.b(x)


class RFB_modified(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = BasicConv2d(in_ch, out_ch, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, (1, 3), padding=(0, 1)),
            BasicConv2d(out_ch, out_ch, (3, 1), padding=(1, 0)),
            BasicConv2d(out_ch, out_ch, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, (1, 5), padding=(0, 2)),
            BasicConv2d(out_ch, out_ch, (5, 1), padding=(2, 0)),
            BasicConv2d(out_ch, out_ch, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, out_ch, 1),
            BasicConv2d(out_ch, out_ch, (1, 7), padding=(0, 3)),
            BasicConv2d(out_ch, out_ch, (7, 1), padding=(3, 0)),
            BasicConv2d(out_ch, out_ch, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(out_ch * 4, out_ch, 3, padding=1)
        self.conv_res = BasicConv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x0, x1, x2, x3 = self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))


# ================= SAM2‑UNet ================================
class SAM2UNet(nn.Module):
    """SAM2 backbone + U‑Net decoder with deep supervision (optional)."""

    def __init__(self, checkpoint_path=None, deep_sup=True):
        super().__init__()
        self.deep_sup = deep_sup
        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)

        # strip unused SAM2 heads
        for attr in [
            "sam_mask_decoder", "sam_prompt_encoder", "memory_encoder", "memory_attention",
            "mask_downsample", "obj_ptr_tpos_proj", "obj_ptr_proj", "image_encoder.neck",
        ]:
            mod, _, tail = attr.partition('.')
            module = getattr(sam2, mod)
            if tail:
                delattr(module, tail)
            else:
                delattr(sam2, mod)

        self.encoder = sam2.image_encoder.trunk
        # freeze backbone params
        for p in self.encoder.parameters():
            p.requires_grad = False
        # adapter prompt tuning
        self.encoder.blocks = nn.Sequential(*[Adapter(b) for b in self.encoder.blocks])

        # channels from SAM2 ViT: 144, 288, 576, 1152
        self.rfb1, self.rfb2 = RFB_modified(144, 64), RFB_modified(288, 64)
        self.rfb3, self.rfb4 = RFB_modified(576, 64), RFB_modified(1152, 64)

        # 4‑stage decoder: 32→64→128→256→512 (for 512² input)
        self.up1 = Up(128, 64)  # x4 + x3
        self.up2 = Up(128, 64)  # prev + x2
        self.up3 = Up(128, 64)  # prev + x1
        self.up4 = Up(64, 64)   # final 2× without skip

        self.pred_final = nn.Conv2d(64, 1, 1)
        if self.deep_sup:
            self.pred_side2 = nn.Conv2d(64, 1, 1)
            self.pred_side1 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        # ---------- encoder ----------
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2 = self.rfb1(x1), self.rfb2(x2)
        x3, x4 = self.rfb3(x3), self.rfb4(x4)

        # ---------- decoder ----------
        d3 = self.up1(x4, x3)        # 32→64
        d2 = self.up2(d3, x2)        # 64→128
        d1 = self.up3(d2, x1)        # 128→256
        d0 = self.up4(d1)            # 256→512 (no skip)

        pred = self.pred_final(d0)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)

        if not self.deep_sup:
            return pred  # single output

        side2 = self.pred_side2(d2)
        side1 = self.pred_side1(d3)
        side2 = F.interpolate(side2, size=(H, W), mode="bilinear", align_corners=False)
        side1 = F.interpolate(side1, size=(H, W), mode="bilinear", align_corners=False)
        
        return pred, side2, side1


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet(deep_sup=True).cuda()
        dummy = torch.randn(1, 3, 512, 512).cuda()
        outs = model(dummy)
        if isinstance(outs, tuple):
            print([o.shape for o in outs])
        else:
            print(outs.shape)
