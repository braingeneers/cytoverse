"""Figure 2 -- compute and data flow diagram, sized for 174 mm.

The original was authored with dpi="150", which scales the PDF by 150/72 and
blew it up to 29.8 in wide. LaTeX then squeezed that into 174 mm (0.23x),
dragging the 14 pt labels down to ~2.3 pt. It also left the cluster label in
Times New Roman because graph-level labels do not inherit the node fontname.

Here we render at graphviz's natural scale, measure it, and iterate the font
size so that after fitting to 174 mm the text lands at the target point size.

    uv run scripts/figures/fig2_architecture.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from graphviz import Digraph

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

OUT = Path("paper/figures/fig2_architecture")
TARGET_WIDTH_IN = 174 / 25.4
TARGET_PT = 7.0
FONT = "Arial"


def build(node_pt: float, edge_pt: float, cluster_pt: float, fit: bool) -> Digraph:
    g = Digraph(comment="Cytoverse Data Flow - Horizontal Full", engine="dot")
    # No dpi attribute: leave graphviz at its native 72 units per inch so the
    # PDF's physical size is predictable.
    g.attr(rankdir="LR", bgcolor="white", nodesep="0.2", ranksep="0.2", fontname=FONT)
    if fit:
        # Fit the drawing to the final print width. Fonts scale with it, which is
        # why node_pt is pre-compensated by the caller.
        g.attr(size=f"{TARGET_WIDTH_IN},100")
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname=FONT,
        fontsize=str(node_pt),
    )
    g.attr("edge", fontname=FONT, fontsize=str(edge_pt), color="#333333")

    g.attr("node", fillcolor="#e8f4fd", color="#2196F3")
    g.node("h5ad", "H5AD\nFile", shape="cylinder")

    g.attr("node", fillcolor="#fff3e0", color="#ff9800")
    g.node("align", "Gene\nAlign")

    g.attr("node", fillcolor="#e8f5e9", color="#4caf50")
    g.node("embed_model", "scFM")
    g.node("umap_model", "PUMAP")

    g.attr("node", fillcolor="#f3e5f5", color="#9c27b0", shape="ellipse")
    g.node("embeddings", "Embeddings")
    g.node("coords", "2D\nCoordinates")

    g.attr("node", fillcolor="#fce4ec", color="#e91e63", shape="box")
    g.node("ivf", "IVF")
    g.node("pq", "PQ")
    g.node("ann", "ANN\nSearch")

    g.attr("node", fillcolor="#e0f2f1", color="#009688", shape="parallelogram")
    g.node("neighbors", "Nearest\nNeighbor\nLabels")
    g.node("viz", "Visualization")

    g.edge("h5ad", "align", label="stream")
    g.edge("align", "embed_model")
    g.edge("embed_model", "embeddings")
    g.edge("embeddings", "umap_model")
    g.edge("embeddings", "ivf")
    g.edge("umap_model", "coords")
    g.edge("coords", "viz")
    g.edge("ivf", "ann")
    g.edge("pq", "ann")
    g.edge("ann", "neighbors")
    g.edge("neighbors", "viz")

    with g.subgraph(name="cluster_ref") as c:
        c.attr(
            label="Static\nReference Data",
            style="filled,dashed",
            fillcolor="#f5f5f5",
            color="#757575",
            # fontname here too: cluster labels do NOT inherit the node font,
            # which is how Times New Roman got into the original.
            fontname=FONT,
            fontsize=str(cluster_pt),
        )
        c.attr(
            "node",
            fillcolor="#fafafa",
            color="#757575",
            shape="cylinder",
            fontsize=str(node_pt),
        )
        c.node("ref_embeddings", "20M+\nEmbeddings")
        c.node("ref_labels", "Annotations")
        c.node("ref_centroids", "Centroids")
        c.node("ref_codebooks", "Codebooks")

    g.edge("ref_centroids", "ivf", style="dashed", color="gray")
    g.edge("ref_codebooks", "pq", style="dashed", color="gray")
    g.edge("ref_embeddings", "pq", style="dashed", color="gray")
    g.edge("ref_labels", "neighbors", style="dashed", color="gray")

    for pair in (("umap_model", "ivf"), ("coords", "ann")):
        with g.subgraph() as s:
            s.attr(rank="same")
            for n in pair:
                s.node(n)

    return g


def render_pdf(g: Digraph, stem: Path) -> Path:
    """Render via the macOS quartz backend.

    graphviz's default cairo PDF backend emits a stray unnamed Type 3 font
    alongside the embedded ArialMT. quartz produces a single ArialMT TrueType
    and no Type 3, which is what Cell Press wants. (Tradeoff: quartz omits the
    ToUnicode map, so text is not extractable -- font type is the requirement,
    extractability is not.)
    """
    src = stem.with_suffix(".gv")
    src.write_text(g.source)
    out = stem.with_suffix(".pdf")
    subprocess.run(
        ["dot", "-Tpdf:quartz:quartz", str(src), "-o", str(out)], check=True
    )
    src.unlink()
    return out


def pdf_size_in(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith("Page size:"))
    parts = line.split()
    return float(parts[2]) / 72.0, float(parts[4]) / 72.0


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Iterate: node font size changes node boxes, which changes the natural
    # width, which changes the fit scale. Converges in a few passes.
    node_pt = 14.0
    for i in range(6):
        render_pdf(build(node_pt, node_pt * 0.9, node_pt * 0.9, fit=False), OUT)
        natural_w, _ = pdf_size_in(OUT.with_suffix(".pdf"))
        scale = TARGET_WIDTH_IN / natural_w
        effective = node_pt * scale
        print(f"  pass {i}: font {node_pt:.1f}pt, natural {natural_w:.2f}in, "
              f"scale {scale:.3f} -> {effective:.2f}pt final")
        if abs(effective - TARGET_PT) < 0.05:
            break
        node_pt *= TARGET_PT / effective

    g = build(node_pt, node_pt * 0.9, node_pt * 0.9, fit=False)
    render_pdf(g, OUT)
    g.render(str(OUT), format="svg", cleanup=True)  # editable source

    # graphviz's size= attribute does not reliably scale the PDF page (it left
    # us at 199 mm when asked for 174), so rescale the page with Ghostscript.
    # Text scales with it, landing at TARGET_PT by construction of the loop.
    natural_w, natural_h = pdf_size_in(OUT.with_suffix(".pdf"))
    scale = TARGET_WIDTH_IN / natural_w
    paper_w = round(TARGET_WIDTH_IN * 72.0)
    paper_h = round(natural_h * scale * 72.0)

    tmp = OUT.with_suffix(".natural.pdf")
    OUT.with_suffix(".pdf").rename(tmp)
    subprocess.run(
        [
            "gs", "-q", "-o", str(OUT.with_suffix(".pdf")),
            "-sDEVICE=pdfwrite",
            f"-dDEVICEWIDTHPOINTS={paper_w}",
            f"-dDEVICEHEIGHTPOINTS={paper_h}",
            "-dFIXEDMEDIA",
            "-dAutoRotatePages=/None",
            "-c", f"<</BeginPage{{{scale} {scale} scale}}>> setpagedevice",
            "-f", str(tmp),
        ],
        check=True,
    )
    tmp.unlink()

    w, h = pdf_size_in(OUT.with_suffix(".pdf"))
    print(f"\n  {OUT}.pdf  {w * 25.4:.1f} x {h * 25.4:.1f} mm")
    print(f"  node text {node_pt * scale:.2f}pt at final size")


if __name__ == "__main__":
    main()
