"""AstroLab - train_n2n.py
Nuestro "Noise2Noise" (inspirado en joeylitalien/noise2noise-pytorch y NVlabs/noise2noise):
la red aprende a denoising SIN imagen limpia, usando pares de exposiciones
independientes (sub_i -> sub_j): el ruido es independiente entre ambas, así que
la única solución consistente es predecir la señal limpia.

Entrenado con subs simulados con el ruido realista de la D610 (simulate.make_sub),
por lo que el modelo sirve directamente para los datos reales de la Nikon.

Uso:  env -u PYTHONPATH .venv/bin/python -u train_n2n.py
Guarda: demo/n2n_unet.pt  +  demo/resultado_n2n.png
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim
from models import UNet
from process import denoise_nlm, asinh_stretch

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


def normalize(img):
    """Subs reales: tienen cielo 180. Restamos el cielo y normalizamos."""
    return np.clip((img - sim.SKY) / 2000.0, 0, 1)


def denormalize(x):
    return np.clip(x, 0, 1) * 2000.0 + sim.SKY


# 1) dataset: sub ruidosa -> stack limpio de 25 (misma escena)
#    (N2N puro usa sub->sub; con stacks disponibles el objetivo limpio da mejor
#     resultado y es lo que haremos con los datos reales de la D610)
X, Y = [], []
for s in range(N_TRAIN_SCENES):
    with_stars, _ = sim.random_scene(seed=2000 + s)
    subs_scene = np.stack([sim.make_sub(with_stars, seed=3000 + s * 50 + i) for i in range(25)])
    stack_clean = subs_scene.mean(axis=0)          # objetivo: apilado limpio
    for c in range(N_CROPS):
        rng = np.random.default_rng(s * 100 + c)
        y0 = rng.integers(0, sim.H - PATCH)
        x0 = rng.integers(0, sim.W - PATCH)
        X.append(normalize(subs_scene[c % 25][y0:y0 + PATCH, x0:x0 + PATCH]))
        Y.append(normalize(stack_clean[y0:y0 + PATCH, x0:x0 + PATCH]))
X = torch.tensor(np.stack(X)[:, None]).float()
Y = torch.tensor(np.stack(Y)[:, None]).float()
print(f"dataset: {X.shape[0]} pares (sub ruidosa -> stack limpio) de {PATCH}x{PATCH}")

# 2) entrenamiento (L2: empuja hacia la media condicional = señal limpia)
model = UNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

n = len(X)
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:2d}/{EPOCHS}: loss = {tot/n:.4f}")

torch.save(model.state_dict(), os.path.join(DEMO, "n2n_unet.pt"))
print("modelo guardado: demo/n2n_unet.pt")

# 3) evaluación en escena NUEVA
model.eval()
with_stars, _ = sim.random_scene(seed=9999)
truth = with_stars + sim.SKY

def find_empty_region(scene, size=110):
    for y0 in range(0, sim.H - size, 40):
        for x0 in range(0, sim.W - size, 40):
            if scene[y0:y0 + size, x0:x0 + size].max() < 3:
                return slice(y0, y0 + size), slice(x0, x0 + size)
empty = find_empty_region(with_stars)

def noise(img):
    return img[empty].std()

def err(img):
    return np.abs(img - truth)[empty].mean()

# --- caso A: stack de 25 subs ---
subs = np.stack([sim.make_sub(with_stars, seed=5000 + i) for i in range(25)])
stack25 = subs.mean(axis=0)

def apply_n2n(img):
    inp = torch.tensor(normalize(img)[None, None]).float().to(DEVICE)
    with torch.no_grad():
        out = denormalize(model(inp).cpu().numpy()[0, 0])
    # corregir sesgo de fondo: igualar la mediana del fondo de la entrada
    out -= (np.median(out[empty]) - np.median(img[empty]))
    return out

n2n_stack = apply_n2n(stack25)
nlm_stack = denoise_nlm(stack25, h=1.5 * stack25[10:120, 10:120].std())

# --- caso B: UNA sola sub (donde N2N debe brillar) ---
sub1 = subs[0]
n2n_sub = apply_n2n(sub1)
nlm_sub = denoise_nlm(sub1, h=1.5 * sub1[10:120, 10:120].std())

print(f"\n=== caso A: stack de 25 subs (ruido {noise(stack25):.2f}) ===")
print(f"  NLM        : ruido {noise(nlm_stack):.2f} | error {err(nlm_stack):.2f}")
print(f"  Noise2Noise: ruido {noise(n2n_stack):.2f} | error {err(n2n_stack):.2f}")
print(f"=== caso B: UNA sub (ruido {noise(sub1):.2f}) ===")
print(f"  NLM        : ruido {noise(nlm_sub):.2f} | error {err(nlm_sub):.2f}")
print(f"  Noise2Noise: ruido {noise(n2n_sub):.2f} | error {err(n2n_sub):.2f}")

# 4) composición (caso B: la sub individual es donde se ve la magia)
def to_uint8(img):
    return (asinh_stretch(img, sim.SKY) * 255).astype(np.uint8)

panels = {
    "1 sub cruda (ruido ~15)": to_uint8(sub1),
    "1 sub + NLM (v1)": to_uint8(nlm_sub),
    "1 sub + Noise2Noise": to_uint8(n2n_sub),
    "verdad limpia": to_uint8(truth),
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

out_png = os.path.join(DEMO, "resultado_n2n.png")
grid.save(out_png)
print("\nguardado:", out_png)
