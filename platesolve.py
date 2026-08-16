"""AstroLab - platesolve.py
Plate solving para el night-runner:
1. ASTAP (local, offline) si hay base de estrellas suficiente
2. nova.astrometry.net (online, anónimo) como respaldo fiable

Uso: env -u PYTHONPATH .venv/bin/python platesolve.py <imagen>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ASTAP = os.path.join(HERE, "astap", "astap_cli")
ASTAP_DB = os.path.join(HERE, "astap", "data")

NOVA = "https://nova.astrometry.net/api"


# ---------------------------------------------------------------- nova online
def _post_json(url, data):
    """nova.astrometry.net espera 'request-json' como campo de formulario."""
    body = urllib.parse.urlencode({"request-json": json.dumps(data)}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _upload_file(url, fields, filepath):
    boundary = "----hermes" + str(os.getpid())
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    with open(filepath, "rb") as f:
        content = f.read()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{os.path.basename(filepath)}\"\r\n"
             "Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def solve_nova(image_path, timeout=300):
    """Resuelve la imagen con nova.astrometry.net (clave desde .nova_key o env)."""
    key_file = os.path.join(HERE, ".nova_key")
    api_key = os.environ.get("NOVA_API_KEY")
    if not api_key and os.path.exists(key_file):
        api_key = open(key_file).read().strip()
    if not api_key:
        return None, "nova: falta API key (.nova_key o NOVA_API_KEY)"
    sess = _post_json(f"{NOVA}/login", {"apikey": api_key})["session"]
    meta = json.dumps({"session": sess, "allow_commercial_usage": "n", "publicly_visible": "n"})
    up = _upload_file(f"{NOVA}/upload", {"request-json": meta}, image_path)
    subid = up.get("subid")
    if not subid:
        return None, f"upload falló: {up}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        sub = _post_json(f"{NOVA}/submissions/{subid}", {})
        jobs = [j for j in sub.get("jobs", []) if j is not None]
        if jobs:
            job = jobs[0]
            info = _post_json(f"{NOVA}/jobs/{job}/info", {})
            if "calibration" in info:
                c = info["calibration"]
                return {"ra": c["ra"], "dec": c["dec"], "scale": c["pixscale"],
                        "orientation": c["orientation"], "solver": "nova.astrometry.net"}, None
            if sub.get("status") == "ERROR":
                return None, "nova: error en el job"
        time.sleep(10)
    return None, "nova: timeout"


# ---------------------------------------------------------------- astap local
def solve_astap(image_path, ra_hint=None, dec_hint=None, fov=None, timeout=180):
    """Resuelve con ASTAP local. ra_hint en horas, dec_hint en grados."""
    if not os.path.exists(ASTAP):
        return None, "ASTAP no instalado"
    cmd = [ASTAP, "-f", image_path, "-d", ASTAP_DB, "-D", "d05",
           "-o", os.path.join(HERE, "demo", "astap_solve"), "-wcs"]
    if ra_hint is not None:
        cmd += ["-ra", str(ra_hint), "-spd", str(90.0 + dec_hint)]  # spd = 90 + Dec (¡no 90 − Dec!)
    if fov:
        cmd += ["-fov", str(fov)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = res.stdout + res.stderr
        for line in out.splitlines():
            if line.startswith("Solution found"):
                parts = line.split()
                # "Solution found: HH MM SS.s -DDd MM SS"
                ra_s, dec_s = parts[2], parts[3]
                return {"ra_str": ra_s, "dec_str": dec_s, "solver": "ASTAP"}, None
        return None, "ASTAP: sin solución"
    except subprocess.TimeoutExpired:
        return None, "ASTAP: timeout"


if __name__ == "__main__":
    img = sys.argv[1]
    ra = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dec = float(sys.argv[3]) if len(sys.argv) > 3 else None
    fov = float(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"=== resolviendo {img} ===")
    sol, err = solve_astap(img, ra / 15.0 if ra else None, dec, fov)
    if sol:
        print("ASTAP:", sol)
    else:
        print("ASTAP falló:", err)
        sol, err = solve_nova(img)
        print("nova:", sol if sol else f"falló: {err}")
