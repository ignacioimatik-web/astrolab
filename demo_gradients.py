"""AstroLab - demo_gradients.py
Integración de GraXpert (repo Steffenhir/GraXpert): eliminación de gradientes.
Demo: apilado con gradiente sintético (cielo sucio) -> GraXpert lo elimina.
"""
import os
import subprocess
import sys
import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim
from process import asinh_stretch

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")
os.makedirs(DEMO, exist_ok=True)

VENV_PY = os.path.join(HERE, ".venv", "bin", "python")
GRAXPERT = os.path.join(HERE, ".venv", "bin", "graxpert")

# 1) datos
subs, truth = sim.generate_subs(50)
stack = subs.mean(axis=0)

# 2) gradiente sintético (Luna baja / contaminación / viñeteado de óptica)
yy, xx = np.mgrid[0:sim.H, 0:sim.W]
grad = 45.0 * (xx / sim.W) + 25.0 * (yy / sim.H) + 20.0 * (xx / sim.W) ** 2
stack_g = stack + grad

print(f"gradiente añadido: +45 ADU a lo ancho, +25 a lo alto, +20 cuadrático")

# 3) guardar FITS y ejecutar GraXpert
in_fits = os.path.join(DEMO, "grad_input.fits")
out_fits = os.path.join(DEMO, "grad_output.fits")
fits.writeto(in_fits, stack_g, overwrite=True)

cmd = [GRAXPERT, in_fits, "-cmd", "background-extraction", "-output", out_fits, "-gpu", "true"]
print("ejecutando:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print("exit:", res.returncode)
if res.returncode != 0:
    print("stderr:", res.stderr[-800:])
    # reintento sin GPU
    print("reintento sin GPU...")
    res = subprocess.run([GRAXPERT, in_fits, "-cmd", "background-extraction",
                          "-output", out_fits, "-gpu", "false"],
                         capture_output=True, text=True, timeout=300)
    print("exit:", res.returncode)
    if res.returncode != 0:
        print("stderr:", res.stderr[-800:])
        sys.exit(1)

stack_gx = fits.getdata(out_fits)
# GraXpert normaliza a [0,1]; devolvemos a la escala de entrada para comparar
stack_gx = stack_gx * float(stack_g.max())
print("salida GraXpert: shape", stack_gx.shape, "| min", stack_gx.min().round(1), "| max", stack_gx.max().round(1))

# 4) métricas de fondo (zona vacía)
def find_empty_region(scene, size=110):
    for y0 in range(0, sim.H - size, 40):
        for x0 in range(0, sim.W - size, 40):
            if scene[y0:y0 + size, x0:x0 + size].max() < 3:
                return slice(y0, y0 + size), slice(x0, x0 + size)
empty = find_empty_region(truth)

def bg_stats(img):
    reg = img[empty]
    # pendiente del fondo: diferencia entre dos mitades de la zona vacía
    half = reg.shape[1] // 2
    tilt = abs(reg[:, :half].mean() - reg[:, half:].mean())
    return reg.std(), tilt

s_clean, t_clean = bg_stats(stack)
s_grad, t_grad = bg_stats(stack_g)
s_gx, t_gx = bg_stats(stack_gx)
print(f"\n=== métricas de fondo ===")
print(f"  sin gradiente : std={s_clean:.2f}  inclinación={t_clean:.2f}")
print(f"  con gradiente : std={s_grad:.2f}  inclinación={t_grad:.2f}")
print(f"  tras GraXpert : std={s_gx:.2f}  inclinación={t_gx:.2f}")

# 5) composición
def to_uint8(img):
    return (asinh_stretch(img, sim.SKY) * 255).astype(np.uint8)

panels = {
    "apilado limpio (referencia)": to_uint8(stack),
    "apilado + gradiente (cielo sucio)": to_uint8(stack_g),
    "tras GraXpert (IA)": to_uint8(stack_gx),
}
pw, ph = 400, 300
grid = Image.new('RGB', (pw * 3, ph), (5, 5, 12))
d = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 16)
except Exception:
    font = ImageFont.load_default()
for i, (name, arr) in enumerate(panels.items()):
    px = i * pw
    img = Image.fromarray(arr, 'L').convert('RGB').resize((pw, ph), Image.LANCZOS)
    grid.paste(img, (px, 0))
    d.rectangle([px, 0, px + pw - 1, ph - 1], outline=(90, 90, 140))
    d.text((px + 10, 8), name, fill=(255, 230, 120), font=font)

out_png = os.path.join(DEMO, "resultado_graxpert.png")
grid.save(out_png)
print("\nguardado:", out_png)
