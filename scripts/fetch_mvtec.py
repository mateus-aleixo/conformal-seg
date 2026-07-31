"""Download MVTec AD (once, ~5.3 GB) and extract the two categories used here.

Source: a full mirror of the original archive on Hugging Face. Officially the
dataset lives at mvtec.com (form-gated); licence is CC BY-NC-SA 4.0 either way —
non-commercial use, attribution, no redistribution. This script downloads to
data/, extracts metal_nut/ and grid/, and leaves nothing in git.

    python scripts/fetch_mvtec.py            # download + extract
    python scripts/fetch_mvtec.py --keep-zip # keep the archive afterwards
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = ("https://huggingface.co/datasets/hdtech/mvtech_anomaly_detection/"
       "resolve/main/mvtech_anomaly_detection.zip")
CATEGORIES = ("metal_nut", "grid")

DATA = Path(__file__).parent.parent / "data"
ZIP = DATA / "mvtec_ad.zip"
ROOT = DATA / "mvtec"


def download() -> None:
    if ZIP.exists() and ZIP.stat().st_size > 5_000_000_000:
        print(f"archive already present: {ZIP}")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    print("downloading MVTec AD (~5.3 GB, one time) ...")
    req = Request(URL, headers={"User-Agent": "conformal-seg fetcher"})
    with urlopen(req, timeout=120) as resp, ZIP.open("wb") as f:
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done % (1 << 28) < (1 << 20):  # every ~256 MB
                print(f"  {done / 1e9:.1f} GB ...")
    print(f"  done: {ZIP.stat().st_size / 1e9:.2f} GB")


def extract() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        prefix = ""  # archives sometimes nest under a single top folder
        top = {n.split("/", 1)[0] for n in names if "/" in n}
        if len(top) == 1 and not any(t in CATEGORIES for t in top):
            prefix = f"{next(iter(top))}/"
        wanted = [n for n in names if any(n.startswith(f"{prefix}{c}/") for c in CATEGORIES)]
        if not wanted:
            sys.exit(f"no {CATEGORIES} entries found in the archive; inspect {ZIP}")
        print(f"extracting {len(wanted)} files for {CATEGORIES} ...")
        for n in wanted:
            target = ROOT / n.removeprefix(prefix)
            if n.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, target.open("wb") as dst:
                dst.write(src.read())
    print(f"extracted under {ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    download()
    extract()
    if not args.keep_zip:
        ZIP.unlink(missing_ok=True)
        print("archive removed (use --keep-zip to keep it)")
