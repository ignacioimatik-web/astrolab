#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroLab Platform — servidor web del pipeline de astrofotografía (Mac Studio)
=============================================================================
Plataforma a la que acudes desde el navegador: subes imágenes/vídeo (subs, FITS,
PNG planetario), lanzas una operación del pipeline (gradientes, starless,
denoise, upscale 4x, plate solving, apilado Siril) y ves el resultado.

Arranque (Mac Studio):
  cd ~/AstroLab && env -u PYTHONPATH .venv/bin/python -u platform/server.py

Acceso: http://<tailnet-ip>:8010  (o localhost:8010 en el propio Studio)
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parent.parent   # ~/AstroLab (o ~/Desktop/ASTRO)
JOBS = ROOT / "platform_data" / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

HOST = os.environ.get("ASTROLAB_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASTROLAB_PORT", "8010"))
VENV_PY = os.environ.get("ASTROLAB_PY", str(ROOT / ".venv" / "bin" / "python"))
GRAXPERT = os.environ.get("ASTROLAB_GRAXPERT", str(ROOT / ".venv" / "bin" / "graxpert"))
SIRIL = os.environ.get("ASTROLAB_SIRIL", "/Applications/Siril.app/Contents/MacOS/siril")

# operaciones del pipeline
STEPS = {
    "gradient":  dict(name="Gradientes (GraXpert)",  ext=".fits",  desc="Elimina gradientes de cielo con IA", accept=".fits,.fit,.png"),
    "starless":  dict(name="Starless (StarNet)",     ext=".fits",  desc="Separa estrellas del fondo con nuestro U-Net", accept=".fits,.fit,.png"),
    "denoise":   dict(name="Denoise (N2N)",          ext=".fits",  desc="Reduce ruido con nuestra red Noise2Noise", accept=".fits,.fit,.png"),
    "upscale":   dict(name="Upscale 4x (ESRGAN)",    ext=".png",   desc="Planetas: superresolución 4x con IA", accept=".png,.jpg,.jpeg"),
    "platesolve":dict(name="Plate solve (ASTAP/nova)", ext=".png", desc="Calcula coordenadas RA/Dec de la imagen", accept=".png,.jpg,.jpeg,.fits,.fit"),
    "stack":     dict(name="Apilado (Siril)",        ext=".fits",  desc="Apila subs NEF/SER en una imagen limpia", accept=".nef,.ser,.cr2,.fits"),
}

app = FastAPI(title="AstroLab Platform", version="7.2")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ---------------------------------------------------------------- estado
QUEUE = []          # ids en espera
RUNNING = {}        # id -> info
LOCK = threading.Lock()

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def job_path(jid):
    return JOBS / jid

def read_job(jid):
    try:
        return json.loads((job_path(jid) / "job.json").read_text())
    except Exception:
        return None

def write_job(job):
    (job_path(job["id"]) / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))

def set_progress(job, pct, msg):
    job["progress"] = pct
    job["log"].append(f"[{pct:3d}%] {msg}")
    job["log"] = job["log"][-60:]
    write_job(job)

def fail(job, err):
    job["status"] = "error"
    job["error"] = str(err)[:500]
    job["log"].append(f"❌ {str(err)[:300]}")
    write_job(job)

# ---------------------------------------------------------------- helpers imagen/FITS
def fits_preview(fits_path, out_png, stretch="asinh"):
    """FITS -> PNG con estirado asinh (fondo oscuro), listo para la galería."""
    from astropy.io import fits
    import numpy as np
    from PIL import Image
    d = fits.getdata(fits_path)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    while d.ndim > 2:                       # FITS multicapa -> capa central
        d = d[d.shape[0] // 2]
    d = d.astype(np.float32)
    bg = np.percentile(d, 20)
    lo = np.percentile(d, 0.5)
    hi = np.percentile(d, 99.8)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if stretch == "asinh":
        d = np.arcsinh(d * 8) / np.arcsinh(8)
    img = Image.fromarray((d * 255).astype(np.uint8))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out_png)

# ---------------------------------------------------------------- worker
def worker_loop():
    while True:
        jid = None
        with LOCK:
            if QUEUE:
                jid = QUEUE.pop(0)
        if jid is None:
            time.sleep(1.5)
            continue
        job = read_job(jid)
        if not job:
            continue
        job["status"] = "running"
        job["started"] = now_iso()
        write_job(job)
        try:
            run_step(job)
        except Exception as e:
            fail(job, e)
        job = read_job(jid)
        if job and job["status"] == "running":
            job["status"] = "done"
            job["finished"] = now_iso()
            write_job(job)
        with LOCK:
            RUNNING.pop(jid, None)

def run_step(job):
    step = job["step"]
    files = sorted((job_path(job["id"]) / "input").iterdir())
    if not files:
        raise RuntimeError("No hay ficheros de entrada")
    inp = str(files[0])
    ext = Path(inp).suffix.lower()

    if step == "gradient":
        out = str(job_path(job["id"]) / "output" / "gradient.fits")
        set_progress(job, 20, "GraXpert: cargando modelo IA (CPU en el Studio)")
        r = subprocess.run([GRAXPERT, inp, "-cmd", "background-extraction",
                            "-output", out, "-gpu", "false"],
                           capture_output=True, text=True, timeout=900)
        set_progress(job, 80, "GraXpert: extracción de fondo aplicada")
        if r.returncode != 0 and not os.path.exists(out):
            raise RuntimeError(r.stderr[-400:] or "GraXpert falló")
        fits_preview(out, str(job_path(job["id"]) / "output" / "preview.png"))
        set_progress(job, 100, "Listo: gradientes eliminados")

    elif step == "starless" or step == "denoise":
        import torch
        import models as M
        import numpy as np
        from astropy.io import fits
        name = "starnet_unet.pt" if step == "starless" else "n2n_unet.pt"
        set_progress(job, 15, f"Cargando U-Net ({name}) en MPS...")
        net = M.UNet()
        net.load_state_dict(torch.load(ROOT / "demo" / name, map_location="cpu"))
        net = net.to("mps").eval()
        if ext in (".fits", ".fit"):
            d = fits.getdata(inp).astype(np.float32)
            while d.ndim > 2:
                d = d[d.shape[0] // 2]
            # normalizar al rango de entrenamiento (sin restar cielo)
            lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.9)
            d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            from PIL import Image
            d = np.array(Image.open(inp).convert("L")).astype(np.float32) / 255.0
        set_progress(job, 40, "Inferencia U-Net (MPS)...")
        # inferencia por tiles (la red es convolucional, acepta cualquier tamaño)
        x = torch.from_numpy(d)[None, None].to("mps")
        with torch.no_grad():
            y = net(x)[0, 0].cpu().numpy()
        out = str(job_path(job["id"]) / "output" / f"{step}.fits")
        from astropy.io import fits as f2
        f2.writeto(out, y.astype(np.float32), overwrite=True)
        # PNG de comparación: original | resultado
        import numpy as np2
        comp = np2.concatenate([d, y], axis=1)
        from PIL import Image
        Image.fromarray((np2.clip(comp, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(
            str(job_path(job["id"]) / "output" / "preview.png"))
        set_progress(job, 100, "Listo")

    elif step == "upscale":
        # shim para basicsr: importa `torchvision.transforms.functional_tensor`,
        # renombrado en torchvision moderno -> copiar atributos de `functional`.
        import types
        import torchvision.transforms.functional as TF
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        for _n in dir(TF):
            if not _n.startswith("_"):
                setattr(_shim, _n, getattr(TF, _n))
        sys.modules["torchvision.transforms.functional_tensor"] = _shim
        import numpy as np
        from PIL import Image
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        set_progress(job, 25, "Real-ESRGAN: cargando pesos...")
        w = ROOT / "weights" / "RealESRGAN_x4plus.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path=str(w), model=model,
                          tile=256, tile_pad=10, pre_pad=0, half=False,
                          device="mps" if torch_mps() else "cpu")
        img = np.array(Image.open(inp).convert("RGB"))
        set_progress(job, 60, "Upscale 4x...")
        out, _ = up.enhance(img[:, :, ::-1], outscale=4)
        out = out[:, :, ::-1]
        outp = str(job_path(job["id"]) / "output" / "upscaled_4x.png")
        Image.fromarray(out).save(outp)
        set_progress(job, 100, "Listo: imagen 4x")

    elif step == "platesolve":
        import platesolve as PS
        # Sin pistas RA/Dec: ASTAP a ciegas con d05 se cuelga (busca todo el cielo) →
        # nova primero (fiable online), ASTAP solo como respaldo offline.
        set_progress(job, 20, "nova.astrometry.net online...")
        sol, err = PS.solve_nova(inp)
        if not sol:
            set_progress(job, 70, "nova no resolvió — ASTAP local (intento breve)...")
            try:
                sol, err = PS.solve_astap(inp, timeout=60)
            except Exception as e:
                err = str(e)
        if not sol:
            raise RuntimeError(err or "Sin solución")
        info = {"ra": round(sol["ra"], 3), "dec": round(sol["dec"], 3),
                "scale": round(sol.get("scale", 0), 3),
                "orientation": round(sol.get("orientation", 0), 1),
                "solver": sol.get("solver")}
        outj = str(job_path(job["id"]) / "output" / "solve.json")
        (job_path(job["id"]) / "output").mkdir(exist_ok=True)
        Path(outj).write_text(json.dumps(info, indent=2))
        set_progress(job, 90, f"Resuelto: RA {info['ra']}° Dec {info['dec']}°")
        if ext in (".png", ".jpg", ".jpeg"):
            shutil.copy(inp, job_path(job["id"]) / "output" / "preview.png")
        set_progress(job, 100, "Listo")

    elif step == "stack":
        set_progress(job, 10, "Siril: preparando script de apilado...")
        odir = job_path(job["id"]) / "output"
        odir.mkdir(exist_ok=True)
        ssf = job_path(job["id"]) / "stack.ssf"
        ssf.write_text("requires 1.2.0\nconvert nef\nregister\nstack\n")
        set_progress(job, 30, "Siril: registrando y apilando (lento)...")
        r = subprocess.run([SIRIL, "-d", "-s", str(ssf)], cwd=job_path(job["id"]),
                           capture_output=True, text=True, timeout=3600)
        res = job_path(job["id"]) / "stacked" / "result.fit"
        if not res.exists():
            raise RuntimeError(r.stderr[-300:] or "Siril no produjo resultado")
        shutil.copy(res, odir / "stacked.fits")
        fits_preview(str(odir / "stacked.fits"), str(odir / "preview.png"))
        set_progress(job, 100, "Listo: subs apilados")

    else:
        raise RuntimeError(f"Operación desconocida: {step}")

def torch_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False

# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/api/steps")
def api_steps():
    return {k: v for k, v in STEPS.items()}

@app.get("/api/plan")
def api_plan():
    import nightrunner as nr
    try:
        p = nr.make_plan(verbose=False)
        return JSONResponse({"ok": True, "plan": nr.render_plan(p)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/api/jobs")
async def create_job(step: str = Form(...), files: list[UploadFile] = File(...)):
    if step not in STEPS:
        return JSONResponse({"error": f"Operación no válida: {step}"}, status_code=400)
    jid = uuid.uuid4().hex[:10]
    jp = job_path(jid)
    (jp / "input").mkdir(parents=True)
    (jp / "output").mkdir(parents=True)
    for f in files:
        safe = Path(f.filename or "input").name
        with open(jp / "input" / safe, "wb") as out:
            out.write(await f.read())
    job = dict(id=jid, step=step, step_name=STEPS[step]["name"], status="queued",
               created=now_iso(), progress=0, log=[], error=None,
               files=[f.filename for f in files])
    write_job(job)
    with LOCK:
        QUEUE.append(jid)
        RUNNING[jid] = True
    return JSONResponse({"id": jid, "status": "queued"})

@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for d in sorted(JOBS.iterdir(), reverse=True):
        j = read_job(d.name)
        if j:
            jobs.append({k: j.get(k) for k in ("id", "step", "step_name", "status", "progress", "created", "error", "files")})
    return jobs

@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = read_job(jid)
    if not j:
        return JSONResponse({"error": "no existe"}, status_code=404)
    outs = []
    odir = job_path(jid) / "output"
    if odir.exists():
        for f in sorted(odir.iterdir()):
            outs.append({"name": f.name, "size": f.stat().st_size,
                         "preview": f.suffix.lower() in (".png", ".jpg", ".jpeg")})
    j["outputs"] = outs
    return j

@app.get("/api/jobs/{jid}/file/{name}")
def job_file(jid: str, name: str):
    base = (job_path(jid) / "output").resolve()
    f = (base / name).resolve()
    if not str(f).startswith(str(base)) or not f.exists():
        return JSONResponse({"error": "no existe"}, status_code=404)
    return FileResponse(f)

@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    j = read_job(jid)
    if j and j["status"] in ("queued", "running"):
        return JSONResponse({"error": "no se puede borrar un job activo"}, status_code=400)
    shutil.rmtree(job_path(jid), ignore_errors=True)
    return {"ok": True}

@app.get("/api/gallery")
def gallery():
    """Todos los previews de resultados, ordenados del más reciente al más antiguo."""
    items = []
    for d in sorted(JOBS.iterdir(), reverse=True):
        j = read_job(d.name)
        if not j or j["status"] != "done":
            continue
        odir = job_path(d.name) / "output"
        if not odir.exists():
            continue
        for f in sorted(odir.iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            items.append({
                "job": j["id"], "step": j["step"], "step_name": j["step_name"],
                "name": f.name, "size": f.stat().st_size,
                "created": j.get("created", ""),
                "files": j.get("files", []),
                "url": f"/api/jobs/{j['id']}/file/{f.name}",
            })
    return items

@app.get("/api/health")
def health():
    return {"ok": True, "jobs_queued": len(QUEUE), "time": now_iso()}

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"🌙 AstroLab Platform — {STEPS}  ->  http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
