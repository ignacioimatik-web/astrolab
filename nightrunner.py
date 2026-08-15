#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroLab Night Runner (v7)
==========================
Orquestador de la noche astronómica para NexStar 6 SE + ASI715MC + Nikon D610
(La Mata de Morella — 40.6165N, -0.27983W, 830 m, Bortle 2-3)

Flujo de una noche:
  1. Meteo (Open-Meteo)      -> ventana de cielo despejado en la noche astronómica
  2. Efemérides (Meeus + JPL Horizons) -> Sol, Luna, planetas
  3. Catálogo DSO            -> ranking de objetivos (altitud maxima × interferencia lunar)
  4. Fases operativas: --capture (gphoto2 D610) / --process (Siril + pipeline IA)
  5. Informe (--plan / --plan-cron para cron de Hermes: silencioso si la noche es mala)

Uso (SIEMPRE con el venv del proyecto):
  env -u PYTHONPATH .venv/bin/python -u nightrunner.py --plan
  env -u PYTHONPATH .venv/bin/python -u nightrunner.py --plan-cron
  env -u PYTHONPATH .venv/bin/python -u nightrunner.py --validate
  env -u PYTHONPATH .venv/bin/python -u nightrunner.py --capture --dry-run
  env -u PYTHONPATH .venv/bin/python -u nightrunner.py --process <dir_subs>
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Madrid")
LAT, LON, ELEV = 40.6165, -0.27983, 830.0
FOCAL_MM = 1500.0                      # NexStar 6 SE f/10
ROOT = os.path.dirname(os.path.abspath(__file__))
NOVA_KEY = os.path.join(ROOT, ".nova_key")

# Escala por cámara a 1500 mm: ASI715 = 0.28"/px (FOV 17.6'x9.9'),
# D610 = 0.81"/px (FOV 81'x54'). px_um ASI715 = 2.0 aprox (verificar con ASIStudio).
CAMERAS = {
    "D610":   {"px_um": 5.9, "w": 6016, "h": 4016, "scale": 0.81, "fov_w": 81.0, "fov_h": 54.0},
    "ASI715": {"px_um": 2.0, "w": 3840, "h": 2160, "scale": 0.28, "fov_w": 17.6, "fov_h": 9.9},
}

# ---------------------------------------------------------------- astronomía (Meeus)
def _rad(d): return math.radians(d)
def _deg(r): return math.degrees(r)

def jd_ut(dt_ut):
    y, m = dt_ut.year, dt_ut.month
    d = dt_ut.day + (dt_ut.hour + dt_ut.minute / 60 + dt_ut.second / 3600) / 24
    if m <= 2: y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5

def sun_eq(jd):
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(_rad(M))
         + (0.019993 - 0.000101 * T) * math.sin(_rad(2 * M)) + 0.000289 * math.sin(_rad(3 * M)))
    lam = L0 + C - 0.00569 - 0.00478 * math.sin(_rad(125.04 - 1934.136 * T))
    eps = 23.43929111 - 0.0130042 * T
    ra = _deg(math.atan2(math.cos(_rad(eps)) * math.sin(_rad(lam)), math.cos(_rad(lam)))) % 360
    dec = _deg(math.asin(math.sin(_rad(eps)) * math.sin(_rad(lam))))
    return ra, dec

def moon_eq(jd):
    T = (jd - 2451545.0) / 36525.0
    Lp = 218.316 + 481267.881 * T
    D = 297.154 + 445267.111 * T
    M = 357.529 + 35999.050 * T
    Mp = 134.963 + 477198.867 * T
    F = 93.272 + 483202.018 * T
    lam = (Lp + 6.289 * math.sin(_rad(Mp)) + 1.274 * math.sin(_rad(2 * D - Mp))
           + 0.658 * math.sin(_rad(2 * D)) + 0.214 * math.sin(_rad(2 * Mp))
           - 0.186 * math.sin(_rad(M)) - 0.114 * math.sin(_rad(2 * F)))
    bet = (5.128 * math.sin(_rad(F)) + 0.280 * math.sin(_rad(Mp + F)) + 0.277 * math.sin(_rad(Mp - F))
           + 0.173 * math.sin(_rad(2 * D - F)) + 0.055 * math.sin(_rad(2 * D + F - Mp))
           + 0.046 * math.sin(_rad(2 * D - F - Mp)) + 0.033 * math.sin(_rad(Mp + F))
           + 0.017 * math.sin(_rad(2 * Mp + F)))
    eps = 23.43929111 - 0.0130042 * T
    ra = _deg(math.atan2(math.cos(_rad(eps)) * math.sin(_rad(lam)) - math.sin(_rad(eps)) * math.tan(_rad(bet)),
                         math.cos(_rad(lam)))) % 360
    dec = _deg(math.asin(math.sin(_rad(eps)) * math.sin(_rad(lam)) * math.cos(_rad(bet))
                         + math.cos(_rad(eps)) * math.sin(_rad(bet))))
    return ra, dec

def gmst_deg(jd):
    T = (jd - 2451545.0) / 36525.0
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T * T - T * T * T / 38710000.0) % 360

def lst_deg(dt_local):
    return (gmst_deg(jd_ut(dt_local.astimezone(dt.timezone.utc))) + LON) % 360

def alt_of(ra, dec, dt_local):
    lst = lst_deg(dt_local)
    ha = (lst - ra) % 360
    if ha > 180: ha -= 360
    return _deg(math.asin(math.sin(_rad(LAT)) * math.sin(_rad(dec))
                          + math.cos(_rad(LAT)) * math.cos(_rad(dec)) * math.cos(_rad(ha))))

def separation(ra1, dec1, ra2, dec2):
    dra, ddec = _rad(ra1 - ra2), _rad(dec1 - dec2)
    a = math.sin(ddec / 2) ** 2 + math.cos(_rad(dec1)) * math.cos(_rad(dec2)) * math.sin(dra / 2) ** 2
    return _deg(2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def moon_illum(jd):
    sra, sdec = sun_eq(jd)
    mra, mdec = moon_eq(jd)
    cos_elong = (math.sin(_rad(sdec)) * math.sin(_rad(mdec))
                 + math.cos(_rad(sdec)) * math.cos(_rad(mdec)) * math.cos(_rad(sra - mra)))
    return (1 - cos_elong) / 2  # elongación 0 -> nueva (0), 180 -> llena (1)

def moon_phase(illum, jd):
    sra, _ = sun_eq(jd)
    mra, _ = moon_eq(jd)
    waxing = ((mra - sra) % 360) < 180  # luna al este del sol: creciente
    if illum < 0.04: return "nueva"
    if illum < 0.46: return "creciente" if waxing else "menguante"
    if illum < 0.54: return "cuarto creciente" if waxing else "cuarto menguante"
    if illum < 0.96: return "gibosa creciente" if waxing else "gibosa menguante"
    return "llena"

def moon_rise_set(night_start):
    """Salida/puesta real de la Luna (cruces de altitud por el horizonte) en la noche."""
    rise = set_ = None
    t0 = night_start - dt.timedelta(hours=16)  # barrido desde las ~04:00 (cubre salidas de mañana)
    prev = None
    for i in range(24 * 6):  # 24 h en pasos de 10 min
        t = t0 + dt.timedelta(minutes=10 * i)
        mra, mdec = moon_eq(jd_ut(t.astimezone(dt.timezone.utc)))
        a = alt_of(mra, mdec, t)
        if prev is not None and prev[0] < 0 <= a and rise is None:
            rise = prev[1] + dt.timedelta(minutes=10)
        if prev is not None and prev[0] > 0 >= a and set_ is None:
            set_ = prev[1] + dt.timedelta(minutes=10)
        prev = (a, t)
    return rise, set_

def ra_hms(ra):
    h = ra / 15
    return f"{int(h)}h{int((h % 1) * 60):02d}m"

def _urlopen(url, timeout=30):
    """urlopen con certifi (el almacén de certificados del CPython de uv está roto
    contra algunos hosts — p.ej. ssd.jpl.nasa.gov: CERTIFICATE_VERIFY_FAILED)."""
    import ssl
    import certifi
    return urllib.request.urlopen(url, timeout=timeout,
                                  context=ssl.create_default_context(cafile=certifi.where()))

# ---------------------------------------------------------------- meteorología (Open-Meteo)
def fetch_weather():
    q = urllib.parse.urlencode({
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,precipitation,cloudcover,wind_speed_10m",
        "timezone": "Europe/Madrid", "forecast_days": 2})
    with _urlopen(f"https://api.open-meteo.com/v1/forecast?{q}") as r:
        j = json.load(r)
    h = j["hourly"]
    return {t: dict(temp=h["temperature_2m"][i], hum=h["relative_humidity_2m"][i],
                    dew=h["dewpoint_2m"][i], precip=h["precipitation"][i],
                    cloud=h["cloudcover"][i], wind=h["wind_speed_10m"][i])
            for i, t in enumerate(h["time"])}

# ---------------------------------------------------------------- planetas (JPL Horizons)
PLANETS = {"Saturno": "699", "Júpiter": "599", "Marte": "499", "Venus": "299"}

def planet_positions(dt_local):
    jd = jd_ut(dt_local.astimezone(dt.timezone.utc))
    out = {}
    for name, cmd in PLANETS.items():
        q = urllib.parse.urlencode({"format": "text", "COMMAND": cmd, "OBJ_DATA": "'NO'",
                                    "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'",
                                    "CENTER": "'500@399'", "TLIST": f"'{jd:.6f}'", "QUANTITIES": "'1'"})
        try:
            with _urlopen(f"https://ssd.jpl.nasa.gov/api/horizons.api?{q}") as r:
                txt = r.read().decode()
            m = re.search(r"\$\$SOE\n(.*?)\n\$\$EOE", txt, re.S)
            if not m: out[name] = None; continue
            line = m.group(1).strip()
            if "," in line:
                fields = [f.strip() for f in line.split(",")]
                ra_s, dec_s = fields[2], fields[3]
            else:  # formato texto: "2026-Aug-17 00:00:00.000 00 55 33.02 +03 07 30.3"
                parts = line.split()
                ra_s, dec_s = " ".join(parts[2:5]), " ".join(parts[5:8])
            ra_parts = [float(x) for x in ra_s.split()]
            dec_parts = [float(x) for x in dec_s.split()]
            ra = 15 * (ra_parts[0] + ra_parts[1] / 60 + ra_parts[2] / 3600)
            sign = -1 if dec_s.lstrip().startswith("-") else 1
            dec = sign * (abs(dec_parts[0]) + dec_parts[1] / 60 + dec_parts[2] / 3600)
            out[name] = (ra % 360, dec)
        except Exception as e:
            out[name] = None
    return out

# ---------------------------------------------------------------- catálogo DSO
# ra/dec J2000 (grados), size en arcmin. cam: recomendación; subs: sugerencia realista.
TARGETS = [
    dict(id="M57", name="Nebulosa del Anillo", tipo="planetaria", ra=283.40, dec=33.03, mag=8.8, size=1.4,
         cam="ASI715 (campo pequeño ideal)", subs="ASI715: 200-300 × 5-10 s gain 300 · D610: 60 × 10 s ISO 3200"),
    dict(id="M27", name="Dumbbell (Mancuerna)", tipo="planetaria", ra=299.90, dec=22.72, mag=7.5, size=8.0,
         cam="D610 · ASI715 con mosaico", subs="D610: 80 × 15 s ISO 3200 · ASI715: 300 × 10 s"),
    dict(id="M13", name="Cúmulo de Hércules", tipo="globular", ra=250.42, dec=36.46, mag=5.8, size=20.0,
         cam="D610 (cabe justo en ASI715)", subs="D610: 60-80 × 15 s ISO 3200"),
    dict(id="M92", name="M92 (Hércules)", tipo="globular", ra=259.28, dec=43.14, mag=6.5, size=14.0,
         cam="D610", subs="D610: 60 × 15 s ISO 3200"),
    dict(id="M51", name="Whirlpool (Remolino)", tipo="galaxia", ra=202.48, dec=47.20, mag=8.4, size=11.0,
         cam="D610", subs="D610: 100-150 × 20 s ISO 3200 (f/10: acumula tiempo)"),
    dict(id="M63", name="Girasol", tipo="galaxia", ra=198.96, dec=42.03, mag=8.6, size=12.0,
         cam="D610", subs="D610: 100 × 20 s ISO 3200"),
    dict(id="M81", name="Bode", tipo="galaxia", ra=148.89, dec=69.07, mag=6.9, size=27.0,
         cam="D610 (borde del FOV)", subs="D610: 100 × 20 s ISO 3200"),
    dict(id="M82", name="Cigarro", tipo="galaxia", ra=148.97, dec=69.68, mag=8.4, size=11.0,
         cam="D610", subs="D610: 100 × 20 s ISO 3200"),
    dict(id="M64", name="Ojo Negro", tipo="galaxia", ra=194.18, dec=21.68, mag=8.5, size=10.0,
         cam="D610", subs="D610: 100 × 20 s ISO 3200"),
    dict(id="M101", name="Molinete", tipo="galaxia", ra=210.80, dec=54.35, mag=7.9, size=29.0,
         cam="D610", subs="D610: 150 × 20 s ISO 3200 (baja brillo superficial)"),
    dict(id="M16", name="Nebulosa del Águila", tipo="nebulosa", ra=274.70, dec=-13.78, mag=6.0, size=35.0,
         cam="D610 + filtro UHC", subs="D610: 60-100 × 20 s ISO 6400 + UHC"),
    dict(id="M17", name="Omega (Cisne)", tipo="nebulosa", ra=275.20, dec=-16.17, mag=6.0, size=46.0,
         cam="D610 + filtro UHC", subs="D610: 60 × 20 s ISO 6400 + UHC"),
    dict(id="M8", name="Laguna", tipo="nebulosa", ra=270.96, dec=-24.38, mag=6.0, size=90.0,
         cam="D610 + UHC (baja en Morella: máx ~25°)", subs="D610: 60 × 20 s ISO 6400 + UHC"),
    dict(id="M20", name="Trífida", tipo="nebulosa", ra=270.65, dec=-23.03, mag=6.3, size=28.0,
         cam="D610 + UHC (baja)", subs="D610: 60 × 20 s ISO 6400 + UHC"),
    dict(id="M1", name="Cangrejo (Crab)", tipo="resto sup.", ra=83.63, dec=22.01, mag=8.4, size=6.0,
         cam="ASI715", subs="ASI715: 300 × 10 s gain 300 (invierno)"),
    dict(id="NGC7789", name="Rosa de Carolina", tipo="cúmulo abierto", ra=359.93, dec=56.72, mag=6.7, size=25.0,
         cam="D610", subs="D610: 40 × 10 s ISO 3200 (otoño)"),
    dict(id="NGC869", name="Doble Cúmulo de Perseo", tipo="cúmulo abierto", ra=35.68, dec=57.13, mag=3.7, size=60.0,
         cam="D610", subs="D610: 30 × 10 s ISO 3200 (otoño)"),
]

# ---------------------------------------------------------------- plan de noche
def make_plan(now=None, weather=None, planets=None, verbose=True):
    now = now or dt.datetime.now(TZ)
    if now.hour < 12:
        night_date = now.date()
    else:
        night_date = now.date()
    start = dt.datetime(night_date.year, night_date.month, night_date.day, 20, 0, tzinfo=TZ)
    hours = [start + dt.timedelta(hours=i) for i in range(13)]  # 20:00 -> 08:00

    weather = weather or fetch_weather()
    if planets is None:
        planets = planet_positions(hours[len(hours) // 2])

    # --- ventana de cielo ---
    rows = []
    for h in hours:
        jd = jd_ut(h.astimezone(dt.timezone.utc))
        sun_alt = alt_of(*sun_eq(jd), h)
        mra, mdec = moon_eq(jd)
        moon_alt = alt_of(mra, mdec, h)
        wkey = h.strftime("%Y-%m-%dT%H:%M")
        w = weather.get(wkey, dict(temp=15, hum=60, dew=8, precip=0, cloud=0, wind=10))
        rows.append(dict(hour=h, sun_alt=sun_alt, moon_alt=moon_alt, moon_illum=moon_illum(jd),
                         **w))
    astro = [r for r in rows if r["sun_alt"] < -18 and r["hour"] >= now]
    good = [r for r in astro if r["precip"] == 0 and r["cloud"] <= 50 and r["wind"] <= 25]
    # ventanas = rachas de >=2 h buenas
    windows, cur = [], []
    for r in astro:
        if r in good:
            cur.append(r)
        else:
            if len(cur) >= 2: windows.append(cur)
            cur = []
    if len(cur) >= 2: windows.append(cur)
    window = max(windows, key=len) if windows else []
    night_ok = len(window) >= 2

    # --- ranking de objetivos ---
    ranked = []
    for t in TARGETS:
        best, best_h = None, None
        for r in window:
            a = alt_of(t["ra"], t["dec"], r["hour"])
            if a > 15 and (best is None or a > best):
                best, best_h = a, r
        if best is None or best < 25:
            continue
        msep = separation(t["ra"], t["dec"], *moon_eq(jd_ut(best_h["hour"].astimezone(dt.timezone.utc))))
        if best_h["moon_alt"] < 0 or best_h["moon_illum"] < 0.30:
            mfactor = 1.0
        elif msep < 25: mfactor = 0.2
        elif msep < 50: mfactor = 0.5
        else: mfactor = 0.85
        score = 100 * (best / 90) ** 1.5 * mfactor
        ranked.append(dict(t=t, alt=best, hora=best_h["hour"], sep=msep, score=score))

    ranked.sort(key=lambda x: -x["score"])
    planets_vis = []
    for name, (pra, pdec) in (planets or {}).items():
        if pra is None: continue
        best = max((alt_of(pra, pdec, r["hour"]) for r in window), default=-90)
        if best > 20:
            bh = max(window, key=lambda r: alt_of(pra, pdec, r["hour"]))
            planets_vis.append(dict(name=name, alt=best, hora=bh["hour"]))

    plan = dict(date=night_date, window=window, night_ok=night_ok, ranked=ranked,
                planets=planets_vis, rows=rows, now=now)
    if verbose:
        print(render_plan(plan))
    return plan

def render_plan(p):
    d = p["date"].strftime("%A %d/%m/%Y")
    w = p["window"]
    moon = p["rows"][0]
    jd0 = jd_ut(moon["hour"].astimezone(dt.timezone.utc))
    illum = moon["moon_illum"]
    fase = moon_phase(illum, jd0)
    rise, set_ = moon_rise_set(p["rows"][0]["hour"].replace(hour=20))
    L = []
    L.append(f"🌙 PLAN DE NOCHE — {d} · La Mata de Morella (40.62N, -0.28W, 830 m)")
    if w:
        h0, h1 = w[0]["hour"].strftime("%H:%M"), w[-1]["hour"].strftime("%H:%M")
        n = len(w)
        cloud = int(sum(r["cloud"] for r in w) / n)
        wind = int(sum(r["wind"] for r in w) / n)
        L.append(f"☁️ Cielo: ventana {h0}–{h1} ({n} h) · nubes ~{cloud}% · viento {wind} km/h")
        margin = min((r["temp"] - r["dew"]) for r in w)
        L.append(f"💧 Rocío: margen {margin:.1f} °C — {'⚠️ riesgo alto' if margin < 2.5 else 'bajo riesgo'}")
    else:
        L.append("☁️ Cielo: ❌ SIN VENTANA útil esta noche (nubes/precipitación o noche corta)")
    rstr = rise.strftime("%H:%M") if rise else "—"
    sstr = set_.strftime("%H:%M") if set_ else "—"
    L.append(f"🌑 Luna: {fase} {illum * 100:.0f}% · sale {rstr} · se pone {sstr}")
    L.append(f"📷 Escala: ASI715 0.28\"/px (FOV 17.6'×9.9') · D610 0.81\"/px (FOV 81'×54') a f/10")
    L.append("")
    if p["ranked"]:
        L.append("⭐ OBJETIVOS DSO (top 5):")
        for i, r in enumerate(p["ranked"][:5], 1):
            t = r["t"]
            L.append(f"  {i}. {t['id']} {t['name']} — {t['tipo']} · máx {r['alt']:.0f}° a las {r['hora'].strftime('%H:%M')} · score {r['score']:.0f}")
            L.append(f"     📷 {t['cam']} · {t['subs']}")
    else:
        L.append("⭐ Ningún DSO del catálogo por encima de 25° esta noche.")
    if p["planets"]:
        L.append("🪐 Planetas (efemérides JPL):")
        for pl in sorted(p["planets"], key=lambda x: -x["alt"]):
            L.append(f"  · {pl['name']}: máx {pl['alt']:.0f}° a las {pl['hora'].strftime('%H:%M')}")
    L.append("")
    L.append("✅ Checklist: Eneloop ×2 cargadas · GB40 + cable pinzas→barril · anti-rocío + parasol · "
             "tarjeta SD formateada · cables USB · abrigo (2-8 °C)")
    return "\n".join(L)

# ---------------------------------------------------------------- fases operativas
def validate(quiet=False):
    checks = []
    def add(name, ok, extra=""):
        checks.append((name, ok, extra))
    def which(cmd):
        r = subprocess.run(["which", cmd], capture_output=True, text=True)
        return r.stdout.strip() or None

    add("gphoto2 (D610)", bool(which("gphoto2")), which("gphoto2") or "brew install gphoto2")
    add("siril (apilado)", bool(which("siril")), which("siril") or "brew install siril (fórmula, NO --cask)")
    astap = os.path.join(ROOT, "astap", "astap_cli")
    add("astap_cli", os.path.exists(astap), astap)
    db = os.path.join(ROOT, "astap", "data")
    ndb = len(os.listdir(db)) if os.path.isdir(db) else 0
    add("base ASTAP d05", os.path.isdir(db) and ndb > 1000, f"{ndb} archivos (H17 para campos de telescopio)")
    add("clave nova (.nova_key)", os.path.exists(NOVA_KEY), "gitignored, chmod 600")
    w = os.path.join(ROOT, "weights", "RealESRGAN_x4plus.pth")
    add("pesos Real-ESRGAN", os.path.exists(w), f"{os.path.getsize(w)//10**6} MB" if os.path.exists(w) else "pesos fuera del repo")
    try:
        import torch
        add("torch + MPS", torch.backends.mps.is_available(), f"torch {torch.__version__}")
    except Exception as e:
        add("torch + MPS", False, str(e))
    add("modelos U-Net", os.path.exists(os.path.join(ROOT, "demo", "starnet_unet.pt"))
        and os.path.exists(os.path.join(ROOT, "demo", "n2n_unet.pt")), "train_starnet.py / train_n2n.py")
    if not quiet:
        print("🔧 VALIDACIÓN DEL SISTEMA:")
        for name, ok, extra in checks:
            print(f"  {'✅' if ok else '❌'} {name}: {extra if not ok else ''}")
        nok = sum(1 for _, ok, _ in checks if not ok)
        print(f"\n  {'¡TODO LISTO PARA LA NOCHE! 🎉' if nok == 0 else f'{nok} pendientes — ver arriba'}")
    return all(ok for _, ok, _ in checks)

def capture(dry_run=False):
    """Fase de captura con la Nikon D610 vía gphoto2 (patrón verificado en el eclipse).
    La ASI715MC (planetary) se captura con ASICap/ASIStudio en SER — fuera de CLI por ahora."""
    print("📷 FASE DE CAPTURA (D610) —", "ENSAYO (sin cámara)" if dry_run else "REAL")
    if dry_run:
        print("  · killall -9 ptpcamerad (liberar puerto PTP)")
        print("  · gphoto2 --auto-detect -> Nikon D610")
        print("  · dial en M, ISO 3200, disparo manual")
        print("  · bucle: gphoto2 --capture-image-and-download --filename cap_%Y%m%d_%H%M%S.nef")
        print("  · caffeinate -dimsu durante la sesión")
        return
    subprocess.run(["killall", "-9", "ptpcamerad"], capture_output=True)
    r = subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True)
    if "Nikon" not in r.stdout:
        print("❌ D610 no detectada. ¿Cable USB y dial en M?")
        return 1
    print("✅ D610 detectada — capturando (Ctrl-C para parar)...")
    try:
        subprocess.run(["caffeinate", "-dimsu", "gphoto2",
                        "--capture-image-and-download", "--filename", "cap_%Y%m%d_%H%M%S.nef"])
    except KeyboardInterrupt:
        print("\n⏹ Captura interrumpida.")
    return 0

def process(data_dir):
    """Apilado Siril + pipeline IA (GraXpert -> StarNet -> N2N)."""
    if not os.path.isdir(data_dir):
        print(f"❌ No existe {data_dir}")
        return 1
    nefs = sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".nef", ".ser")))
    if not nefs:
        print(f"❌ Sin NEF/SER en {data_dir}")
        return 1
    print(f"🔬 APILADO: {len(nefs)} subs de {data_dir}")
    script = os.path.join(data_dir, "stack.ssf")
    with open(script, "w") as f:
        f.write("requires 1.2.0\nconvert nef\nregister\nstack\n")
    r = subprocess.run(["siril", "-d", "-s", script], cwd=data_dir)
    if r.returncode != 0:
        print("⚠️ Siril falló — ¿instalado? (brew install --cask siril)")
        return 1
    stack = os.path.join(data_dir, "stacked", "result.fit")
    print(f"✅ Stack: {stack}")
    print("➡️ Siguiente: graxpert -cmd background-extraction · starnet · n2n (pipeline v2-v4)")
    return 0

# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="AstroLab Night Runner v7")
    ap.add_argument("--plan", action="store_true", help="plan de esta noche (siempre imprime)")
    ap.add_argument("--plan-cron", action="store_true", help="para cron: NADA si la noche es mala")
    ap.add_argument("--validate", action="store_true", help="comprueba herramientas")
    ap.add_argument("--capture", action="store_true", help="fase de captura D610")
    ap.add_argument("--process", metavar="DIR", help="apilado Siril + pipeline")
    ap.add_argument("--dry-run", action="store_true", help="ensayo sin hardware")
    args = ap.parse_args()

    if args.plan_cron:
        p = make_plan(verbose=False)
        if p["night_ok"] and p["ranked"]:
            print(render_plan(p))
        return 0
    if args.plan:
        make_plan()
        return 0
    if args.validate:
        return 0 if validate() else 1
    if args.capture:
        return capture(dry_run=args.dry_run)
    if args.process:
        return process(args.process)
    ap.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
