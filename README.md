# 🔭 AstroLab

**Pipeline IA de astrofotografía — el "RC-Astro casero"**

Procesado de imágenes astronómicas de cielo profundo con técnicas de IA y procesado clásico: **denoising**, **eliminación de estrellas** y **deconvolución** para recuperar detalle. Pensado para el flujo de un aficionado: cámara réflex (Nikon D610) + telescopio SCT (Celestron NexStar 6 SE) bajo cielos oscuros (Bortle 2-3).

> Resultado clave: **25 subs + nuestro denoise (ruido 0.24) supera a 100 subs crudos (1.42)** — el procesado IA gana a 4× más tiempo de integración.

## 🚀 Quick start

```bash
cd ASTRO
python3 -m venv .venv
.venv/bin/pip install numpy scipy scikit-image astropy photutils pillow torch

# demo end-to-end: 100 subs simulados → stack → denoise → starless → deconvolución
env -u PYTHONPATH .venv/bin/python demo.py
# → genera demo/resultado.png con la comparativa
```

## 🧩 Módulos

| Módulo | Función |
|---|---|
| `simulate.py` | Generador de subs con ruido realista (Poisson del cielo + ruido de lectura + píxeles calientes + seeing) |
| `process.py` | `denoise_nlm()` · `remove_stars()` (StarXTerminator casero) · `estimate_psf()` · `deconvolve_rl()` (BlurXTerminator casero) |
| `demo.py` | Pipeline end-to-end con métricas y comparativa visual |

## 📊 Resultados de la demo (subs simulados con ruido D610)

| Procesado | Ruido de fondo (ADU) | Mejora |
|---|---|---|
| 25 subs crudos | 2.84 | 1× |
| 100 subs crudos | 1.42 | 2× |
| **25 subs + denoise NLM** | **0.24** | **12×** |

## 🗺️ Roadmap (Fase 2)

- [ ] **Noise2Noise U-Net** entrenada con subs reales (PyTorch + MPS)
- [ ] **Real-ESRGAN** upscale 4× para planetas
- [ ] **Night-runner**: captura automática + apilado + informe matutino
- [ ] Integración de **StarNet** para separación de estrellas

## ⚠️ Notas técnicas

- `scikit-image 0.26` trae la Richardson-Lucy **rota** (converge a basura) → `process.py` implementa la suya propia (10 líneas, `fftconvolve`, conserva flujo).
- El estirado **asinh con sustracción del cielo** es el que hace visible el detalle débil (el gamma lo comprime).
- En macOS, el `PYTHONPATH` de otros venvs contamina los imports → ejecutar siempre con `env -u PYTHONPATH`.

## 🛠️ Stack

Python · NumPy · SciPy · scikit-image · astropy · photutils · PyTorch (MPS) · PIL
