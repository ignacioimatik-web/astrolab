"""AstroLab - process.py
Nuestro "RC-Astro casero": módulos de procesado con IA/clásicos.

- denoise_nlm():     denoising no-local-means (baseline, sin entrenar)
- denoise_n2n():    (torch) Noise2Noise entrenado con los PROPIOS subs  [Fase 2]
- remove_stars():   StarXTerminator casero: detección DAOStarFinder + inpainting biharmónico
- estimate_psf():   PSF empírica desde estrellas del propio campo
- deconvolve_rl():  BlurXTerminator casero: deconvolución Richardson-Lucy con PSF medida
"""
import numpy as np
from skimage.restoration import denoise_nl_means, inpaint_biharmonic
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats


# ---------------------------------------------------------------- denoising
def denoise_nlm(img, patch=5, dist=11, h=0.6):
    """Denoising no-local-means. h ~ fracción del sigma del ruido."""
    return denoise_nl_means(img, patch_size=patch, patch_distance=dist, h=h, fast_mode=True)


def denoise_n2n(subs_train, truth, epochs=20, lr=1e-3):
    """Fase 2: red U-Net Noise2Noise entrenada con subs reales -> stack limpio.
    Requiere torch. Se implementa en train_n2n.py."""
    raise NotImplementedError("Fase 2: pip install torch y ejecuta train_n2n.py")


# ---------------------------------------------------------------- star removal
def remove_stars(img, fwhm=2.5, threshold_sigma=6.0, psf_pad=6):
    """StarXTerminator casero: detecta estrellas y las rellena (inpaint)."""
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    stars = finder(img - median)
    mask = np.zeros(img.shape, dtype=bool)
    if stars is not None:
        for x, y in zip(stars['xcentroid'], stars['ycentroid']):
            x, y = int(round(x)), int(round(y))
            x0, x1 = max(0, x - psf_pad), min(img.shape[1], x + psf_pad + 1)
            y0, y1 = max(0, y - psf_pad), min(img.shape[0], y + psf_pad + 1)
            mask[y0:y1, x0:x1] = True
    starless = inpaint_biharmonic(img, mask)
    return starless, stars


# ---------------------------------------------------------------- deconvolution
def estimate_psf(img, stars, box=25, fwhm=2.5):
    """PSF empírica: mediana de recortes normalizados alrededor de estrellas brillantes."""
    if stars is None or len(stars) < 3:
        yy, xx = np.mgrid[0:box, 0:box]
        cx = cy = box // 2
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (fwhm / 2.355) ** 2))
    cuts = []
    half = box // 2
    for x, y, f in zip(stars['xcentroid'], stars['ycentroid'], stars['flux']):
        if f < np.median(stars['flux']):
            continue
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - half), min(img.shape[1], x + half + 1)
        y0, y1 = max(0, y - half), min(img.shape[0], y + half + 1)
        cut = img[y0:y1, x0:x1]
        if cut.shape != (box, box):
            continue
        bkg = np.median(cut)
        c = cut - bkg
        if c.sum() > 0:
            cuts.append(c / c.sum())
    if not cuts:
        yy, xx = np.mgrid[0:box, 0:box]
        cx = cy = box // 2
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (fwhm / 2.355) ** 2))
    psf = np.median(np.stack(cuts), axis=0)
    return psf / psf.sum()


def deconvolve_rl(img, psf, iterations=25, eps=1e-6):
    """BlurXTerminator casero: deconvolución Richardson-Lucy implementada por nosotros
    (la de skimage 0.26 converge mal). Algoritmo clásico, conserva flujo."""
    from scipy.signal import fftconvolve

    def conv(x, k):
        return fftconvolve(x, k, mode='same')

    psf_mirror = psf[::-1, ::-1]
    est = np.full(img.shape, float(np.median(img)))
    for _ in range(iterations):
        ratio = img / (conv(est, psf) + eps)
        est = est * conv(ratio, psf_mirror)
    return np.clip(est, 0, None)


# ---------------------------------------------------------------- utilidades
def asinh_stretch(img, sky, k=30.0, maxval=3000.0):
    """Estirado asinh con sustracción de cielo (mismo que la demo)."""
    return np.clip(np.arcsinh((img - sky) / k) / np.arcsinh(maxval / k), 0, 1)
