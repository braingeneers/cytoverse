"""Shared figure styling for the Cell Patterns submission.

Two rules make the rest of the pipeline work:

1. Figures are authored at their FINAL print width. A figure drawn 15 in wide and
   shrunk to 174 mm by LaTeX takes its fonts down with it -- that is how the
   original set ended up with 2-4 pt labels. Author at 6.85 in and 7 pt text
   stays 7 pt on the page.

2. We do NOT use ``bbox_inches="tight"``. Tight bbox trims the canvas to the ink,
   so the saved PDF is narrower than the figsize you asked for; ``\\includegraphics``
   then scales it back up and the type size drifts again. ``layout="constrained"``
   fits the content inside the exact figsize instead, so scale factor is 1.0 and
   what you specify is what prints.

See paper/FIGURES.md for the per-figure width assignment.
"""

from __future__ import annotations

import subprocess

import matplotlib as mpl

MM_PER_IN = 25.4

# Cell Press permitted widths, in inches.
WIDTH_1COL = 85 / MM_PER_IN  # 3.346 in
WIDTH_1_5COL = 114 / MM_PER_IN  # 4.488 in
WIDTH_2COL = 174 / MM_PER_IN  # 6.850 in

# Cell Press asks for Arial or Helvetica with a ~6 pt floor at final size.
BASE_PT = 7
SMALL_PT = 6


def use_style() -> None:
    """Apply Cell Press-compatible defaults to matplotlib's global rcParams."""
    mpl.rcParams.update(
        {
            # Type 42 = embedded TrueType. The default Type 3 is neither Arial
            # nor Helvetica, is not text-extractable, and is a common production
            # rejection.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # Mathtext ignores font.sans-serif and defaults to DejaVu, which
            # silently embeds DejaVuSans-Oblique the moment you write $n_{sub}$.
            # Point it at Arial so a subscript does not drag in a second family.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "font.size": BASE_PT,
            "axes.titlesize": BASE_PT,
            "axes.labelsize": BASE_PT,
            "xtick.labelsize": SMALL_PT,
            "ytick.labelsize": SMALL_PT,
            "legend.fontsize": SMALL_PT,
            "figure.titlesize": BASE_PT,
            # Hairlines scaled for a figure that is not going to be reduced.
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            # Deliberately NOT "tight" -- see module docstring.
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "savefig.transparent": False,
        }
    )


def save(fig, path, target_width_in: float) -> None:
    """Save, then verify the PDF page really is the intended physical width.

    Checks the written file rather than ``fig.get_size_inches()`` -- matplotlib
    snaps the canvas to whole pixels at figure dpi, so the requested and actual
    sizes differ by a fraction of a millimetre. The PDF page box is what the
    journal measures, so that is what we assert on.
    """
    # Omit CreationDate so a rebuild is byte-identical and does not dirty the
    # working tree with a figure whose content did not change.
    fig.savefig(path, metadata={"CreationDate": None})

    out = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith("Page size:"))
    parts = line.split()
    got_mm = float(parts[2]) / 72.0 * MM_PER_IN
    got_h_mm = float(parts[4]) / 72.0 * MM_PER_IN
    want_mm = target_width_in * MM_PER_IN

    if abs(got_mm - want_mm) > 0.3:
        raise ValueError(
            f"{path}: page is {got_mm:.2f} mm wide, expected {want_mm:.2f} mm"
        )
    print(f"  {path}  {got_mm:.1f} x {got_h_mm:.1f} mm")
