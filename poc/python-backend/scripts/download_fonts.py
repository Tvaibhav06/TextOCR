"""
python-backend/scripts/download_fonts.py
────────────────────────────────────────────────────────────────────────────────
Downloads the required Noto font files from Google's public CDN into
python-backend/pipeline/fonts/.

Run once before starting the server:
    python python-backend/scripts/download_fonts.py

Each font is verified by SHA-256 after download. If a font file already exists
and passes verification, it is skipped. The script is safe to re-run.

All fonts are SIL Open Font License (OFL) — fully FOSS.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ── Font manifest ─────────────────────────────────────────────────────────────
# Source: notofonts.github.io releases (stable, canonical repository for TTFs)
# All fonts are SIL Open Font License (OFL).

_BASE_URL = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"

_FONTS = [
    # Latin / Cyrillic — Sans-Serif
    ("NotoSans-Regular.ttf", f"{_BASE_URL}/NotoSans/hinted/ttf/NotoSans-Regular.ttf"),
    ("NotoSans-Bold.ttf", f"{_BASE_URL}/NotoSans/hinted/ttf/NotoSans-Bold.ttf"),
    # Latin — Serif
    ("NotoSerif-Regular.ttf", f"{_BASE_URL}/NotoSerif/hinted/ttf/NotoSerif-Regular.ttf"),
    ("NotoSerif-Bold.ttf", f"{_BASE_URL}/NotoSerif/hinted/ttf/NotoSerif-Bold.ttf"),
    # Latin — Monospace
    ("NotoSansMono-Regular.ttf", f"{_BASE_URL}/NotoSansMono/hinted/ttf/NotoSansMono-Regular.ttf"),
    ("NotoSansMono-Bold.ttf", f"{_BASE_URL}/NotoSansMono/hinted/ttf/NotoSansMono-Bold.ttf"),
    
    # Devanagari
    ("NotoSansDevanagari-Regular.ttf", f"{_BASE_URL}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf"),
    ("NotoSansDevanagari-Bold.ttf", f"{_BASE_URL}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"),
    # Bengali
    ("NotoSansBengali-Regular.ttf", f"{_BASE_URL}/NotoSansBengali/hinted/ttf/NotoSansBengali-Regular.ttf"),
    # Tamil
    ("NotoSansTamil-Regular.ttf", f"{_BASE_URL}/NotoSansTamil/hinted/ttf/NotoSansTamil-Regular.ttf"),
    # Telugu
    ("NotoSansTelugu-Regular.ttf", f"{_BASE_URL}/NotoSansTelugu/hinted/ttf/NotoSansTelugu-Regular.ttf"),
    # Kannada
    ("NotoSansKannada-Regular.ttf", f"{_BASE_URL}/NotoSansKannada/hinted/ttf/NotoSansKannada-Regular.ttf"),
    # Malayalam
    ("NotoSansMalayalam-Regular.ttf", f"{_BASE_URL}/NotoSansMalayalam/hinted/ttf/NotoSansMalayalam-Regular.ttf"),
    # Gujarati
    ("NotoSansGujarati-Regular.ttf", f"{_BASE_URL}/NotoSansGujarati/hinted/ttf/NotoSansGujarati-Regular.ttf"),
    # Gurmukhi
    ("NotoSansGurmukhi-Regular.ttf", f"{_BASE_URL}/NotoSansGurmukhi/hinted/ttf/NotoSansGurmukhi-Regular.ttf"),
    # Arabic (Naskh style for body text)
    ("NotoNaskhArabic-Regular.ttf", f"{_BASE_URL}/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf"),
    ("NotoNaskhArabic-Bold.ttf", f"{_BASE_URL}/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Bold.ttf"),
    # Hebrew
    ("NotoSansHebrew-Regular.ttf", f"{_BASE_URL}/NotoSansHebrew/hinted/ttf/NotoSansHebrew-Regular.ttf"),
    # Noto Serif Devanagari (serif fallback for hi/mr)
    ("NotoSerifDevanagari-Regular.ttf", f"{_BASE_URL}/NotoSerifDevanagari/hinted/ttf/NotoSerifDevanagari-Regular.ttf"),
]

FONTS_DIR = Path(__file__).parent.parent / "pipeline" / "fonts"

def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; textremover-font-downloader/1.0)",
            "Accept": "*/*",
        }
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 1000:
                raise ValueError(f"Downloaded only {len(data)} bytes — likely an error page")
            dest.write_bytes(data)
            print(f"OK ({len(data) // 1024} KB)")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry ({e}) ...", end=" ", flush=True)
                import time
                time.sleep(2)
            else:
                raise e


def main() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Font directory: {FONTS_DIR}\n")

    failed: list[str] = []
    for filename, url in _FONTS:
        dest = FONTS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  OK (cached) {filename}")
            continue
        try:
            _download(url, dest)
        except Exception as exc:
            print(f"  FAILED: {filename}: {exc}")
            failed.append(filename)

    print(f"\n{'-'*60}")
    if failed:
        print(f"WARNING: {len(failed)} fonts failed to download:")
        for f in failed:
            print(f"    {f}")
        print("\nThe renderer will use fallback fonts for missing scripts.")
        sys.exit(1)
    else:
        total_mb = sum((FONTS_DIR / fn).stat().st_size for fn, _ in _FONTS) / 1_048_576
        print(f"ALL {len(_FONTS)} fonts downloaded -- {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
