#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nas_export.py — exportación de resultados al NAS Synology via SMB directo
=========================================================================
Cliente SMB puro (smbprotocol): SIN montajes de macOS -> inmune al TCC
que bloquea el acceso a volúmenes desde procesos SSH/launchd.

Config (env vars o ~/AstroLab/platform/.nas_key, chmod 600, gitignored):
  NAS_HOST    (default 100.110.148.69 = synology-920 tailnet)
  NAS_USER    (default jistev)
  NAS_SHARE   (default photo — share verificado accesible)
  NAS_PASS    (en .nas_key si no en env)

Uso:
  .venv/bin/python platform/nas_export.py --test           # conexión + shares
  .venv/bin/python platform/nas_export.py --upload <file>  # sube a AstroLab/<fecha>/
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import (
    Open, FilePipePrinterAccessMask, FileAttributes, FileInformationClass,
    ImpersonationLevel, CreateDisposition, CreateOptions,
)

HERE = Path(__file__).resolve().parent
NAS_HOST = os.environ.get("NAS_HOST", "100.110.148.69")
NAS_PORT = int(os.environ.get("NAS_PORT", "445"))
NAS_USER = os.environ.get("NAS_USER", "jistev")
NAS_SHARE = os.environ.get("NAS_SHARE", "photo")
REMOTE_SUBDIR = os.environ.get("NAS_SUBDIR", "AstroLab")

# Sync/CreateOptions combinables
FILE_SYNC_IO = 0x00000001
FILE_DIRECTORY_FILE = 0x00000001
FILE_NON_DIRECTORY_FILE = 0x00000040


def _load_pass():
    env = os.environ.get("NAS_PASS")
    if env:
        return env
    kf = HERE / ".nas_key"
    if kf.exists():
        return kf.read_text().strip()
    raise RuntimeError("Falta contraseña NAS: NAS_PASS env o .nas_key (chmod 600)")


def _connect(share=None):
    conn = Connection(uuid.uuid4(), NAS_HOST, NAS_PORT)
    conn.connect()
    sess = Session(conn, NAS_USER, _load_pass(), require_encryption=True)
    sess.connect()
    tree = TreeConnect(sess, share or NAS_SHARE)
    tree.connect()
    return conn, tree


def test_shares(candidates=("home", "homes", "photo", "video", "data",
                            "backup", "files", "media", "music", "docker",
                            "public", "AstroLab")):
    ok = []
    for sh in candidates:
        try:
            conn, _ = _connect(sh)
            conn.disconnect()
            ok.append(sh)
        except Exception:
            pass
    return ok


def mkdir_recursive(tree, path):
    parts = [p for p in path.split("/") if p]
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}" if cur else p
        try:
            o = Open(tree, cur)
            o.create(ImpersonationLevel.Impersonation,
                     FilePipePrinterAccessMask.GENERIC_READ,
                     FileAttributes.FILE_ATTRIBUTE_DIRECTORY, 0,
                     CreateDisposition.FILE_OPEN, 0)
            o.close()
        except Exception:
            o = Open(tree, cur)
            o.create(ImpersonationLevel.Impersonation,
                     FilePipePrinterAccessMask.GENERIC_WRITE,
                     FileAttributes.FILE_ATTRIBUTE_DIRECTORY, 0,
                     CreateDisposition.FILE_CREATE, CreateOptions.FILE_DIRECTORY_FILE)
            o.close()


def upload(local_path, remote_subdir=REMOTE_SUBDIR):
    local = Path(local_path)
    if not local.exists():
        raise FileNotFoundError(local)
    conn, tree = _connect()
    try:
        base = f"{remote_subdir}/{datetime.now():%Y-%m-%d}"
        mkdir_recursive(tree, base)
        remote = f"{base}/{local.name}"
        print(f"  subiendo -> {NAS_SHARE}/{remote}")
        o = Open(tree, remote)
        o.create(ImpersonationLevel.Impersonation,
                 FilePipePrinterAccessMask.GENERIC_WRITE,
                 FileAttributes.FILE_ATTRIBUTE_NORMAL, 0,
                 CreateDisposition.FILE_CREATE, FILE_NON_DIRECTORY_FILE)
        try:
            with open(local, "rb") as f:
                while True:
                    chunk = f.read(1024 * 512)
                    if not chunk:
                        break
                    o.write(chunk, 0)
        finally:
            o.close()
        return f"smb://{NAS_HOST}/{NAS_SHARE}/{remote}"
    finally:
        conn.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Exportación NAS (smbprotocol)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--upload", metavar="FILE")
    args = ap.parse_args()
    if args.test:
        print(f"Conectando a {NAS_USER}@{NAS_HOST}:{NAS_PORT} (share {NAS_SHARE})...")
        ok = test_shares()
        print(f"Shares accesibles: {', '.join(ok) if ok else 'NINGUNO'}")
        return 0
    if args.upload:
        print(upload(args.upload))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())