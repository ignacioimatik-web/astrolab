"""AstroLab - models.py
Arquitectura U-Net compartida (inspirada en StarNet de nekitmm/starnet).
Usada por train_starnet.py (eliminación de estrellas) y train_n2n.py (denoising).
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net compacto (encoder-decoder, estilo StarNet)."""

    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        c_in = 1
        for c in ch:
            self.enc.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU()))
            self.pool.append(nn.MaxPool2d(2))
            c_in = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            self.dec.append(nn.Sequential(
                nn.Conv2d(ch[i - 1] * 2, ch[i - 1], 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch[i - 1], ch[i - 1], 3, padding=1), nn.ReLU()))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for idx, (e, p) in enumerate(zip(self.enc, self.pool)):
            x = e(x)
            skips.append(x)
            if idx < len(self.enc) - 1:   # no se agrupa tras el último encoder
                x = p(x)
        for i in range(len(self.up)):
            x = self.up[i](x)
            x = torch.cat([x, skips[len(skips) - 2 - i]], 1)
            x = self.dec[i](x)
        return self.head(x)
