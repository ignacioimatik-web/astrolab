"""AstroLab - train_starnet.py
Nuestro "StarNet" casero: U-Net encoder-decoder (arquitectura inspirada en
nekitmm/starnet) que elimina estrellas. Entrenado con escenas simuladas:
  entrada = imagen apilada CON estrellas → objetivo = escena SIN estrellas.

Uso:  env -u PYTHONPATH .venv/bin/python train_starnet.py
Guarda: demo/starnet_unet.pt  +  demo/resultado_starnet.png (comparativa)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")
os.makedirs(DEMO, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

torch.manual_seed(0)
np.random.seed(0)

PATCH = 160
N_TRAIN_SCENES = 40
N_CROPS = 6
EPOCHS = 30
BATCH = 8
LR = 1e-3


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


def normalize(img):
    """Normalización a [0,1] (las escenas van SIN cielo: fondo ~0)."""
    return np.clip(img / 2000.0, 0, 1)


# 1) dataset: escenas variadas -> parches aleatorios
X, Y = [], []
for s in range(N_TRAIN_SCENES):
    with_stars, without = sim.random_scene(seed=1000 + s)
    # pequeña dosis de ruido en la entrada (como un stack de ~25 subs)
    noisy = with_stars + np.random.default_rng(s).normal(0, 3.0, with_stars.shape)
    for c in range(N_CROPS):
        rng = np.random.default_rng(s * 100 + c)
        y0 = rng.integers(0, sim.H - PATCH)
        x0 = rng.integers(0, sim.W - PATCH)
        X.append(normalize(noisy[y0:y0 + PATCH, x0:x0 + PATCH]))
        Y.append(normalize(without[y0:y0 + PATCH, x0:x0 + PATCH]))
X = torch.tensor(np.stack(X)[:, None]).float()
Y = torch.tensor(np.stack(Y)[:, None]).float()
print(f"dataset: {X.shape[0]} parches de {PATCH}x{PATCH}")

# 2) entrenamiento
model = UNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def weighted_loss(pred, xb, yb):
    """L1 ponderado: castiga no eliminar estrellas y no preservar la señal débil."""
    stars = (xb - yb) > 0.01          # zonas con estrellas
    signal = yb > 0.02                # galaxia/nebulosa/estrellas (señal débil)
    w = 1.0 + 3.0 * stars.float() + 6.0 * signal.float()
    return (w * (pred - yb).abs()).mean()

n = len(X)
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
        opt.zero_grad()
        loss = weighted_loss(model(xb), xb, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:2d}/{EPOCHS}: loss = {tot/n:.4f}")

torch.save(model.state_dict(), os.path.join(DEMO, "starnet_unet.pt"))
print("modelo guardado: demo/starnet_unet.pt")

# 3) evaluación en escena NUEVA (nunca vista)
model.eval()
with_stars, without = sim.random_scene(seed=7777)
stack_in = with_stars + np.random.default_rng(9).normal(0, 3.0, with_stars.shape)
inp = torch.tensor(normalize(stack_in)[None, None]).float().to(DEVICE)
with torch.no_grad():
    pred = model(inp).cpu().numpy()[0, 0]
# sin cielo: los demás paneles van en escala de señal (fondo ~0)
pred_img = np.clip(pred, 0, 1) * 2000.0

# referencia: inpaint clásico (el método actual)
from process import remove_stars
starless_cl, stars = remove_stars(stack_in, fwhm=4.5)

def to_uint8(img):
    sky_p = np.median(img[100:200, 100:200])
    from process import asinh_stretch
    return (asinh_stretch(img, sky_p) * 255).astype(np.uint8)

# métricas: residuo de estrellas (comparar en la MISMA escala: sin cielo)
star_mask = (without < with_stars - 40)
res_inpaint = np.abs(starless_cl - without)[star_mask].mean()
res_unet = np.abs(pred_img - without)[star_mask].mean()
print(f"\nresiduo medio en estrellas | inpaint: {res_inpaint:.1f} | U-Net: {res_unet:.1f}")

# 4) composición
panels = {
    "entrada (con estrellas)": to_uint8(stack_in),
    "inpaint clásico": to_uint8(starless_cl),
    "nuestro StarNet (U-Net)": to_uint8(pred_img),
    "verdad sin estrellas": to_uint8(without),
}
pw, ph = 400, 300
grid = Image.new('RGB', (pw * 2, ph * 2), (5, 5, 12))
d = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 15)
except Exception:
    font = ImageFont.load_default()
for i, (name, arr) in enumerate(panels.items()):
    px, py = (i % 2) * pw, (i // 2) * ph
    img = Image.fromarray(arr, 'L').convert('RGB').resize((pw, ph), Image.LANCZOS)
    grid.paste(img, (px, py))
    d.rectangle([px, py, px + pw - 1, py + ph - 1], outline=(90, 90, 140))
    d.text((px + 10, py + 8), name, fill=(255, 230, 120), font=font)

out_png = os.path.join(DEMO, "resultado_starnet.png")
grid.save(out_png)
print("guardado:", out_png)
