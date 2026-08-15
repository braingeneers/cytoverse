"""Check every manuscript figure against Cell Press requirements.

Reports, per figure: physical width vs its assigned column width, embedded font
types, the smallest type size actually on the page, and (for raster figures)
effective dpi at the assigned width.

Everything is measured from the file -- page box, font dictionary, and the `Tf`
operators in the content streams -- rather than inferred from the code that
generated it. That distinction matters: the first pass of Phase 2 looked correct
by construction but had 3.5-4.9 pt mathtext glyphs on the page.

    uv run scripts/figures/verify.py

Exits non-zero if any figure fails. See paper/FIGURES.md for the assignments.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zlib
from pathlib import Path

MM_PER_IN = 25.4
MIN_PT = 6.0
MIN_DPI = 300

# Assigned width in mm; see the Phase 0 table in paper/FIGURES.md.
FIGURES = [
    ("paper/figures/captured/fig1_ui_guide.png", 174, "raster"),
    ("paper/figures/fig2_architecture.pdf", 174, "vector"),
    # Figs 3 and 4 are blocked on missing query datasets.
    ("paper/figures/fig5_recall_vs_probes.pdf", 114, "vector"),
    ("paper/figures/fig6_distortion_grid.pdf", 174, "vector"),
    ("paper/figures/fig7_performance_scaling.pdf", 174, "vector"),
]


def pdf_page_mm(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    parts = next(x for x in out.splitlines() if x.startswith("Page size:")).split()
    return (
        float(parts[2]) / 72.0 * MM_PER_IN,
        float(parts[4]) / 72.0 * MM_PER_IN,
    )


def pdf_fonts(path: Path) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["pdffonts", str(path)], capture_output=True, text=True, check=True
    ).stdout
    rows = [r for r in out.splitlines()[2:] if r.strip()]
    return [(r.split()[0], " ".join(r.split()[1:3])) for r in rows]


def pdf_min_type_pt(path: Path) -> float | None:
    """Smallest font size in any content stream, via the `Tf` operators."""
    data = path.read_bytes()
    sizes: set[float] = set()
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        chunk = data[start:end]
        try:
            chunk = zlib.decompress(chunk)
        except zlib.error:
            pass
        for t in re.finditer(rb"/[A-Za-z0-9]+\s+([\d.]+)\s+Tf", chunk):
            try:
                value = float(t.group(1))
            except ValueError:
                continue
            if value > 0:
                sizes.add(round(value, 2))
    return min(sizes) if sizes else None


def png_px(path: Path) -> tuple[int, int]:
    buf = path.read_bytes()
    if buf[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    return int.from_bytes(buf[16:20], "big"), int.from_bytes(buf[20:24], "big")


def main() -> int:
    failures: list[str] = []
    print(f"{'figure':38s} {'width':>9s} {'target':>7s} {'metric':>9s}  status")
    print("-" * 80)

    for rel, target_mm, kind in FIGURES:
        path = Path(rel)
        if not path.exists():
            print(f"{path.name:38s} {'-':>9s} {target_mm:7d} {'-':>9s}  MISSING")
            failures.append(f"{rel}: missing")
            continue

        problems: list[str] = []
        if kind == "vector":
            w_mm, _ = pdf_page_mm(path)
            if abs(w_mm - target_mm) > 0.5:
                problems.append(f"width {w_mm:.1f} mm != {target_mm} mm")

            type3 = [n for n, t in pdf_fonts(path) if "Type 3" in t]
            if type3:
                problems.append(f"Type 3 font(s): {', '.join(type3)}")

            non_arial = [
                n
                for n, _ in pdf_fonts(path)
                if not re.search(r"arial|helvetica", n, re.I)
            ]
            if non_arial:
                problems.append(f"non-Arial font(s): {', '.join(non_arial)}")

            min_pt = pdf_min_type_pt(path)
            metric = f"{min_pt:.2f}pt" if min_pt else "no text"
            if min_pt is not None and min_pt < MIN_PT - 0.05:
                problems.append(f"type {min_pt:.2f} pt below {MIN_PT} pt")
        else:
            px_w, _ = png_px(path)
            dpi = px_w / (target_mm / MM_PER_IN)
            w_mm = target_mm
            metric = f"{dpi:.0f}dpi"
            if dpi < MIN_DPI:
                problems.append(f"{dpi:.0f} dpi below {MIN_DPI}")

        status = "OK" if not problems else "FAIL"
        print(f"{path.name:38s} {w_mm:8.1f}m {target_mm:7d} {metric:>9s}  {status}")
        for p in problems:
            print(f"{'':38s} -> {p}")
            failures.append(f"{rel}: {p}")

    print()
    if failures:
        print(f"{len(failures)} problem(s) found.")
        return 1
    print("All figures meet Cell Press requirements.")
    print("NOTE: Figs 3 and 4 are not in this set -- blocked on missing query")
    print("datasets. See paper/FIGURES.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
