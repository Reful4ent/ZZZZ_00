#!/usr/bin/env python3
"""Download the datasets used by p2geom (TZ 02) into the layout the README expects.

    data/hpatches-sequences-release/{i_*,v_*}/...   (HPatches sequences)
    data/mvtec_ad/<category>/{train,test,ground_truth}/...   (MVTec AD)

Sources (the original hosts are gone: icvl.ee.ic.ac.uk does not resolve and the
MVTec mydrive.ch links return 404):
  * HPatches: Hugging Face mirror by the dataset author (vbalnt/hpatches), MIT.
  * MVTec AD: Hugging Face mirror foersben/mvtec-ad with the original directory
    layout. License CC BY-NC-SA 4.0: non-commercial use only; cite Bergmann et al.

Every file is checked against the SHA-256 (or size, for small non-LFS files)
published by the Hub. Existing verified files are skipped, so reruns resume.
Standard library only.

Usage:
    python scripts/download_datasets.py                      # both, carpet/leather/tile/wood
    python scripts/download_datasets.py --only mvtec --categories carpet
    python scripts/download_datasets.py --data-dir /big/disk/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HF = "https://huggingface.co"

HPATCHES_REPO = "vbalnt/hpatches"
HPATCHES_FILE = "hpatches-sequences-release.zip"
HPATCHES_SHA256 = "99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670"
HPATCHES_DIR = "hpatches-sequences-release"

MVTEC_REPO = "foersben/mvtec-ad"
MVTEC_DIR = "mvtec_ad"
MVTEC_MAIN = ("carpet", "leather", "tile", "wood")   # p2geom.datasets.mvtec.MAIN_CATEGORIES
MVTEC_ALL = MVTEC_MAIN + ("bottle", "cable", "capsule", "grid", "hazelnut", "metal_nut",
                          "pill", "screw", "toothbrush", "transistor", "zipper")


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "p2geom-download/1"})
    return urllib.request.urlopen(req, timeout=60)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified(path: Path, size: int, sha256: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return sha256 is None or _sha256(path) == sha256


def _download(url: str, dest: Path, size: int, sha256: str | None, progress: bool = False) -> None:
    """Download to dest via a .part file; raise if size or hash do not match."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    done = 0
    with _get(url) as r, part.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                print(f"\r  {done / 2**20:8.1f} / {size / 2**20:.1f} MiB", end="", flush=True)
    if progress:
        print()
    if done != size or (sha256 and h.hexdigest() != sha256):
        part.unlink()
        raise RuntimeError(f"verification failed for {url}: got {done} bytes, sha256 {h.hexdigest()}")
    part.replace(dest)


def _hf_tree(repo: str, path: str) -> list[dict]:
    """All files under `path` in a Hub dataset repo (follows pagination)."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}?recursive=true"
    files: list[dict] = []
    while url:
        with _get(url) as r:
            files += [e for e in json.load(r) if e["type"] == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
    return files


def _hf_file_url(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def fetch_hpatches(data_dir: Path, keep_zip: bool) -> None:
    out = data_dir / HPATCHES_DIR
    if out.is_dir() and any(out.glob("[iv]_*")):
        print(f"HPatches: {out} exists, skipped")
        return
    zpath = data_dir / HPATCHES_FILE
    if _verified(zpath, 1279545572, HPATCHES_SHA256):
        print(f"HPatches: {zpath} already downloaded")
    else:
        print(f"HPatches: downloading {HPATCHES_FILE} (1.2 GiB)")
        _download(_hf_file_url(HPATCHES_REPO, HPATCHES_FILE), zpath, 1279545572, HPATCHES_SHA256,
                  progress=True)
    print(f"HPatches: extracting to {data_dir}")
    tmp = data_dir / (HPATCHES_DIR + ".extract")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)            # zipfile drops absolute paths and '..'
    src = tmp / HPATCHES_DIR
    if not src.is_dir():
        raise RuntimeError(f"{HPATCHES_FILE} has no top-level {HPATCHES_DIR}/")
    src.replace(out)
    shutil.rmtree(tmp)
    if not keep_zip:
        zpath.unlink()
    print(f"HPatches: {len(list(out.glob('[iv]_*')))} sequences in {out}")


def fetch_mvtec(data_dir: Path, categories: list[str]) -> None:
    root = data_dir / MVTEC_DIR
    for cat in categories:
        files = _hf_tree(MVTEC_REPO, cat)
        if not files:
            raise RuntimeError(f"MVTec: no files for category {cat!r} in {MVTEC_REPO}")
        fetched = 0
        for i, e in enumerate(files, 1):
            sha = (e.get("lfs") or {}).get("oid")
            dest = root / e["path"]
            if not _verified(dest, e["size"], sha):
                _download(_hf_file_url(MVTEC_REPO, e["path"]), dest, e["size"], sha)
                fetched += 1
            print(f"\rMVTec {cat}: {i}/{len(files)} files", end="", flush=True)
        print(f" ({fetched} downloaded)")
    print(f"MVTec: {root}  (CC BY-NC-SA 4.0, non-commercial use only)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="default: ./data")
    ap.add_argument("--only", choices=("hpatches", "mvtec"), help="download one dataset only")
    ap.add_argument("--categories", nargs="+", default=list(MVTEC_MAIN), choices=MVTEC_ALL,
                    metavar="CAT", help=f"MVTec categories (default: {' '.join(MVTEC_MAIN)})")
    ap.add_argument("--keep-zip", action="store_true", help="keep the HPatches zip after extraction")
    a = ap.parse_args(argv)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "hpatches"):
        fetch_hpatches(a.data_dir, a.keep_zip)
    if a.only in (None, "mvtec"):
        fetch_mvtec(a.data_dir, a.categories)
    return 0


if __name__ == "__main__":
    sys.exit(main())
