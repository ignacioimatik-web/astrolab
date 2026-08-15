"""AstroLab - simulate.py
Generador de subs simulados con el ruido realista de la Nikon D610
(12s @ ISO1600, f/10, Bortle 3): Poisson del cielo + ruido de lectura + píxeles calientes.

Escena: galaxia espiral + nebulosa + 55 estrellas (mismo modelo de la demo anterior).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(42)
H, W = 600, 800
SKY = 180.0          # fondo de cielo (ADU)
READ_NOISE = 4.0     # ruido de lectura rms (ADU)
SEEING = 1.9         # σ del seeing/óptica en px (≈2" a 0.82"/px con la D610)


def add_star(arr, cx, cy, flux, sigma=1.1):
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return arr + flux * np.exp(-r2 / (2 * sigma ** 2))


def build_scene(seed=42, seeing=SEEING):
    rng = np.random.default_rng(seed)
    scene = np.zeros((H, W), dtype=np.float64)
    for _ in range(55):
        scene = add_star(scene, rng.uniform(10, W - 10), rng.uniform(10, H - 10),
                         rng.uniform(30, 2500), rng.uniform(0.8, 1.6))
    yy, xx = np.mgrid[0:H, 0:W]
    gx, gy = 300, 300
    r2 = (xx - gx) ** 2 + (yy - gy) ** 2
    scene += 900 * np.exp(-r2 / (2 * 14 ** 2))
    scene += 260 * np.exp(-r2 / (2 * 55 ** 2))
    scene += 70 * np.exp(-r2 / (2 * 110 ** 2))
    for ang, off in [(0.6, 38), (0.6 + np.pi, 38)]:
        r2b = (xx - (gx + off * np.cos(ang))) ** 2 + (yy - (gy + off * np.sin(ang))) ** 2
        scene += 55 * np.exp(-r2b / (2 * 16 ** 2))
    r2c = (xx - (gx + 105)) ** 2 + (yy - (gy - 18)) ** 2
    scene += 130 * np.exp(-r2c / (2 * 9 ** 2))
    nx, ny = 560, 420
    r2n = (xx - nx) ** 2 + (yy - ny) ** 2
    scene += 700 * np.exp(-r2n / (2 * 10 ** 2)) + 150 * np.exp(-r2n / (2 * 45 ** 2))
    scene += 45 * np.exp(-((xx - (nx - 70)) ** 2 + (yy - (ny + 30)) ** 2) / (2 * 30 ** 2))
    scene += 40 * np.exp(-((xx - (nx + 65)) ** 2 + (yy - (ny + 25)) ** 2) / (2 * 32 ** 2))
    # seeing: desenfoque atmosférico + óptico (lo que la deconvolución debe recuperar)
    if seeing:
        scene = gaussian_filter(scene, seeing)
    return scene


def random_scene(seed):
    """Escena aleatoria para entrenar (galaxia, nebulosa y estrellas en posiciones variables).
    Devuelve (con_estrellas, sin_estrellas), ambas con seeing."""
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W), dtype=np.float64)

    # galaxia en posición aleatoria
    gx = rng.uniform(180, W - 180)
    gy = rng.uniform(150, H - 150)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (xx - gx) ** 2 + (yy - gy) ** 2
    f = rng.uniform(0.7, 1.3)
    base += f * 900 * np.exp(-r2 / (2 * 14 ** 2))
    base += f * 260 * np.exp(-r2 / (2 * 55 ** 2))
    base += f * 70 * np.exp(-r2 / (2 * 110 ** 2))
    for ang in [rng.uniform(0, np.pi), 0]:
        off = rng.uniform(30, 50)
        r2b = (xx - (gx + off * np.cos(ang))) ** 2 + (yy - (gy + off * np.sin(ang))) ** 2
        base += f * 55 * np.exp(-r2b / (2 * 16 ** 2))

    # nebulosa en posición aleatoria
    nx = rng.uniform(150, W - 150)
    ny = rng.uniform(120, H - 120)
    r2n = (xx - nx) ** 2 + (yy - ny) ** 2
    fn = rng.uniform(0.6, 1.4)
    base += fn * 700 * np.exp(-r2n / (2 * 10 ** 2))
    base += fn * 150 * np.exp(-r2n / (2 * 45 ** 2))

    # estrellas (solo en la versión "con estrellas")
    n_stars = int(rng.integers(30, 70))
    with_stars = base.copy()
    for _ in range(n_stars):
        with_stars = add_star(with_stars, rng.uniform(15, W - 15), rng.uniform(15, H - 15),
                              rng.uniform(40, 2500), rng.uniform(0.8, 1.8))
    if SEEING:
        with_stars = gaussian_filter(with_stars, SEEING)
        base = gaussian_filter(base, SEEING)
    return with_stars, base


def make_sub(scene, seed=None):
    rng = np.random.default_rng(seed)
    img = rng.poisson(SKY + scene).astype(np.float64)
    img += rng.normal(0, READ_NOISE, img.shape)
    for _ in range(25):
        img[rng.integers(0, H), rng.integers(0, W)] += rng.uniform(120, 450)
    return img


def generate_subs(n, seed=1):
    scene = build_scene()
    subs = np.stack([make_sub(scene, seed=seed + i) for i in range(n)])
    return subs, scene


if __name__ == "__main__":
    import os
    os.makedirs("demo", exist_ok=True)
    subs, scene = generate_subs(100)
    np.save("demo/subs.npy", subs)
    np.save("demo/truth.npy", scene)
    print("guardado: demo/subs.npy (100 subs), demo/truth.npy | forma:", subs.shape)
