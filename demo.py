"""AstroLab - demo.py
Demo end-to-end de nuestro pipeline "RC-Astro casero":
  subs simulados (ruido D610) -> stack -> denoise -> starless -> deconvolución RL
Genera demo/resultado.png con los paneles comparativos y métricas.
"""
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont

import simulate as sim
from process import (denoise_nlm, remove_stars, estimate_psf, deconvolve_rl, asinh_stretch)

t0 = time.time()

# 1) datos
subs, truth = sim.generate_subs(100)
N = subs.shape[0]
print(f"subs generadas: {N} en {time.time()-t0:.1f}s")

def stack(n):
    return subs[:n].mean(axis=0)

s25 = stack(25)
s100 = stack(100)

# 2) métricas de ruido (zona vacía)
def find_empty_region(scene, size=110):
    for y0 in range(0, sim.H - size, 40):
        for x0 in range(0, sim.W - size, 40):
            if scene[y0:y0+size, x0:x0+size].max() < 3:
                return slice(y0, y0+size), slice(x0, x0+size)
empty = find_empty_region(truth)
def noise(img):
    return img[empty].std()

print(f"ruido 25 subs: {noise(s25):.2f} | 100 subs: {noise(s100):.2f}")

# 3) procesado
t1 = time.time()
noise25 = noise(s25)
s25_dn = denoise_nlm(s25, h=1.5 * noise25)   # h adaptativa al ruido real
print(f"denoise NLM: {time.time()-t1:.1f}s | ruido tras denoise: {noise(s25_dn):.2f}")

t1 = time.time()
starless, stars = remove_stars(s25_dn, fwhm=4.5)
n_stars = len(stars) if stars is not None else 0
print(f"star removal: {time.time()-t1:.1f}s | estrellas detectadas: {n_stars}")

t1 = time.time()
psf = estimate_psf(s25_dn, stars)
s25_dc = deconvolve_rl(starless, psf, iterations=25)
print(f"deconvolución RL: {time.time()-t1:.1f}s")

# 4) composición
def to_uint8(img):
    return (asinh_stretch(img, sim.SKY) * 255).astype(np.uint8)

panels = {
    "25 subs (sin procesar)": to_uint8(s25),
    "100 subs": to_uint8(s100),
    "25 + denoise IA": to_uint8(s25_dn),
    "25 + IA + sin estrellas": to_uint8(starless),
    "25 + IA + estrellas + deconv RL": to_uint8(s25_dc),
    "VERDAD (cielo perfecto)": to_uint8(truth + sim.SKY),
}

pw, ph = 400, 300
grid = Image.new('RGB', (pw*3, ph*2), (5, 5, 12))
d = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 16)
except Exception:
    font = ImageFont.load_default()

for i, (name, arr) in enumerate(panels.items()):
    px, py = (i % 3) * pw, (i // 3) * ph
    img = Image.fromarray(arr, 'L').convert('RGB').resize((pw, ph), Image.LANCZOS)
    grid.paste(img, (px, py))
    d.rectangle([px, py, px+pw-1, py+ph-1], outline=(90, 90, 140))
    d.text((px+10, py+8), name, fill=(255, 230, 120), font=font)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "resultado.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
grid.save(out)
print("guardado:", out)

# 5) resumen
print("\n=== RESUMEN ===")
print(f"ruido 25 subs          : {noise(s25):6.2f}")
print(f"ruido 100 subs         : {noise(s100):6.2f}")
print(f"ruido 25 + denoise     : {noise(s25_dn):6.2f}  (mejora {noise(s25)/max(noise(s25_dn),1e-9):.1f}x)")
print(f"estrellas detectadas   : {n_stars}")
print(f"tiempo total           : {time.time()-t0:.1f}s")
