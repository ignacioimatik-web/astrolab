"""AstroLab - demo_planet.py
Integración de Real-ESRGAN (xinntao/Real-ESRGAN): upscale 4x con IA.
Demo planetaria: Júpiter sintético de 512px -> "captura" de 128px (como la
ASI715MC vería el planeta con seeing) -> upscale 4x Real-ESRGAN vs bicúbico.

Uso:  env -u PYTHONPATH .venv/bin/python -u demo_planet.py
Guarda: demo/resultado_planet.png
"""
import os
import sys

# --- shim para basicsr (importa un módulo renombrado de torchvision) ---
import types
import torchvision.transforms.functional as TF
_shim = types.ModuleType("torchvision.transforms.functional_tensor")
for _n in dir(TF):
    if not _n.startswith("_"):
        setattr(_shim, _n, getattr(TF, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _shim

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")
os.makedirs(DEMO, exist_ok=True)
WEIGHTS = os.path.join(HERE, "weights")
os.makedirs(WEIGHTS, exist_ok=True)


# 1) Júpiter sintético (512x512) — bandas, GRS, achatamiento, sombreado de limbo
def make_jupiter(size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    r = size * 0.46
    # disco achatado (Júpiter ~6% más ancho que alto)
    rx, ry = r, r * 0.94
    d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    disc = d2 <= 1.0

    # bandas: color por latitud (borde superior = "norte")
    lat = (yy - cy) / ry
    bands = [
        (0.30, 0.40, (0.80, 0.72, 0.60)),   # zona templada
        (0.16, 0.30, (0.68, 0.55, 0.44)),   # cinturón
        (0.02, 0.16, (0.85, 0.79, 0.68)),   # zona ecuatorial
        (-0.10, 0.02, (0.74, 0.62, 0.50)),  # cinturón ecuatorial
        (-0.28, -0.10, (0.86, 0.80, 0.70)), # zona tropical
        (-0.42, -0.28, (0.66, 0.53, 0.42)), # cinturón
        (-0.52, -0.42, (0.82, 0.75, 0.65)), # zona polar
    ]
    img = np.zeros((size, size, 3))
    for lo, hi, col in bands:
        sel = disc & (lat >= lo) & (lat < hi)
        img[sel] = col
    # variación suave dentro de las bandas (nubes)
    rng = np.random.default_rng(7)
    swirl = 0.06 * np.sin(yy / 9 + 3 * np.sin(xx / 45)) + 0.04 * np.sin(xx / 13)
    img = np.clip(img + swirl[:, :, None] * 0.6, 0, 1)
    # Gran Mancha Roja
    gs = disc & (((xx - cx - rx * 0.42) / (rx * 0.14)) ** 2 + ((yy - cy + ry * 0.10) / (ry * 0.08)) ** 2 <= 1)
    img[gs] = (0.88, 0.42, 0.24)
    # sombreado de limbo (oscurece los bordes)
    img *= ((1.0 - 0.35 * d2) ** 0.5)[..., None]
    return img


# 2) "captura" de 128px: lo que vería la ASI715MC (Júpiter ~145px con seeing)
jup512 = make_jupiter()
small = np.array(Image.fromarray((jup512 * 255).astype(np.uint8)).resize((128, 128), Image.LANCZOS))
rng = np.random.default_rng(3)
small = small + rng.normal(0, 6, small.shape)      # ruido de lectura + seeing
small = np.clip(small, 0, 255).astype(np.uint8)

# 3) upscale con Real-ESRGAN
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
upsampler = RealESRGANer(
    scale=4, model_path=os.path.join(WEIGHTS, "RealESRGAN_x4plus.pth"),
    model=model, half=False, device=torch.device(device))
print("modelo Real-ESRGAN listo (device:", device, ")")

small_bgr = small[:, :, ::-1]  # realesrgan espera BGR
with torch.no_grad():
    out_bgr, _ = upsampler.enhance(small_bgr, outscale=4)
esrgan = np.clip(out_bgr[:, :, ::-1], 0, 255).astype(np.uint8)

# bicúbico (referencia clásica)
bicubic = np.array(Image.fromarray(small).resize((512, 512), Image.BICUBIC))

# 4) métricas vs el Júpiter original
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))

print(f"\n=== métricas (vs Júpiter 512px original) ===")
print(f"  bicúbico   : PSNR {psnr(bicubic, jup512*255):.1f} dB | SSIM {ssim(bicubic, jup512*255, channel_axis=2, data_range=255):.3f}")
print(f"  Real-ESRGAN: PSNR {psnr(esrgan, jup512*255):.1f} dB | SSIM {ssim(esrgan, jup512*255, channel_axis=2, data_range=255):.3f}")

# 5) composición
panels = {
    "captura ASI715MC (128px)": np.array(Image.fromarray(small).resize((400, 400), Image.NEAREST)),
    "upscale bicúbico 4x": np.array(Image.fromarray(bicubic).resize((400, 400), Image.LANCZOS)),
    "Real-ESRGAN 4x (IA)": np.array(Image.fromarray(esrgan).resize((400, 400), Image.LANCZOS)),
    "Júpiter original 512px": np.array(Image.fromarray((jup512 * 255).astype(np.uint8)).resize((400, 400), Image.LANCZOS)),
}
pw = ph = 400
grid = Image.new('RGB', (pw * 2, ph * 2), (5, 5, 12))
d = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 15)
except Exception:
    font = ImageFont.load_default()
for i, (name, arr) in enumerate(panels.items()):
    px, py = (i % 2) * pw, (i // 2) * ph
    grid.paste(Image.fromarray(arr, 'RGB'), (px, py))
    d.rectangle([px, py, px + pw - 1, py + ph - 1], outline=(90, 90, 140))
    d.text((px + 10, py + 8), name, fill=(255, 230, 120), font=font)

out_png = os.path.join(DEMO, "resultado_planet.png")
grid.save(out_png)
print("\nguardado:", out_png)
