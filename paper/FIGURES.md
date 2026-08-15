# Figure preparation for Cell Patterns submission

Working document for bringing the seven cited figures to Cell Press production
standard. Phase 0 (width assignment) is settled; phases 1-4 are pending.

## Journal requirements

Per the [Cell Press digital image guidelines](https://www.cell.com/figureguidelines?mobileUi=0):

- **Resolution at final print size:** >=300 dpi color/grayscale, 500 dpi bitmap
  B&W, 1000 dpi line art.
- **Permitted widths** (2-column format, used by research articles): **85 mm**
  (1 column), **114 mm** (1.5 columns), **174 mm** (full page width).
  The journal does not assign a width per figure -- authors choose one of the
  three and must hit 300 dpi *at that chosen size*.
- **Height:** each figure must fit on a single page with its legend.
- **Fonts:** Arial or Helvetica, ~6 pt minimum at final size.
- **Formats:** TIFF for raster, EPS/PDF for vector. RGB acceptable.

> Not independently verified against the live page: cell.com returns 403 to
> automated fetches and a Cloudflare challenge in-browser. The 85/114/174 mm set
> is consistent across multiple retrievals, but one secondhand summary lists only
> 85 and 174 mm. **Confirm the 1.5-column option before finalizing Fig 5.**

## Phase 0 decisions (settled)

| Fig | File | Width | Height | Current status | Action needed |
|-----|------|-------|--------|----------------|---------------|
| 1 | `cytoverse_tutorial.pdf` | **174 mm** | 159 mm | 129 dpi FAIL | Recapture; move callout legend to caption; flatten CropBox |
| 2 | `architecture_horizontal_full.pdf` | **174 mm** | 67 mm | vector OK, text ~2.3 pt FAIL | Re-render with fonts sized for 174 mm |
| 3 | `Cytoverse_sspsygene.png` | **174 mm** | 100 mm | 225 dpi FAIL | Recapture at 4x |
| 4 | `importance.pdf` | **174 mm** | 72 mm | 265 dpi FAIL | Recapture at 4x + rebuild composite reproducibly |
| 5 | `recall_vs_probes.pdf` | **114 mm** | 68 mm | vector OK, text 8.2 pt OK | Regenerate at 114 mm target |
| 6 | `distortion_vs_recall_grid.pdf` | **174 mm** | 103 mm | vector OK, text ~4.5 pt FAIL | Restructure: shared axes + 3 ticks/panel |
| 7 | `performance_scaling.pdf` | **174 mm** | 72 mm | vector OK, text ~4.4 pt FAIL | Regenerate at 174 mm target |

Heights are artwork only, excluding caption. All fit within a single page.

### Pixel targets for raster figures

| Width | px @ 300 dpi | px from 4x recapture | headroom |
|-------|--------------|----------------------|----------|
| 174 mm | 2055 | 6400 | 3.1x (934 dpi) |
| 114 mm | 1347 | 6400 | 4.8x (1426 dpi) |

A 1600x900 viewport at `deviceScaleFactor: 4` clears the bar comfortably;
2x would also pass at 174 mm (467 dpi) if 4x proves unwieldy.

### Rationale for the three judgment calls

**Fig 6 -- restructure rather than resize.** At 174 mm the four columns give
~38 mm per panel, each carrying its own y-axis and five x-tick labels of the
form `0.0000 / 0.0005 / 0.0010`. Twelve-character tick labels in a 38 mm panel
are unreadable at 6 pt at any dpi. Resolution: keep all 12 panels, use one
shared x-label per column and y-label per row, reduce to 3 ticks per panel with
a shared scientific-notation exponent. Recovers ~30% of panel area with no data
loss.

**Fig 5 -- 114 mm.** A single-panel, three-series line chart does not warrant
full width; at 174 mm it would run 104 mm tall. Falls back to 174 mm if the
1.5-column option turns out to be unavailable.

**Fig 1 -- legend moves to the caption.** At full width the artwork is 185 mm
tall, of which the bottom 14% (55.1 pt of the 393.3 pt crop) is the numbered
callout list. Moving that text into the caption drops the artwork to 159 mm,
sets the callout text in typeset body font, and removes it from figure-font
rules entirely. The circled numerals stay overlaid on the screenshot.

## Known hazards

**Fig 1 CropBox/MediaBox mismatch.** MediaBox is 720x405 pt (the full 16:9
Google slide); CropBox is 369.5x393.3 pt. `pdftex.def` defaults to `cropbox`,
so the compiled paper is correct today -- verified: 100 pt wide renders 106.4 pt
tall under the default and cropbox, but 56.2 pt under mediabox. Illustrator and
some RIPs honor MediaBox and would render the uncropped slide with half the
canvas empty. Flatten the crop before submission.

**Type 3 fonts throughout.** Every matplotlib PDF embeds DejaVu Sans as Type 3
(`pdffonts` shows `uni=no` -- text is not even extractable). Type 3 is neither
Arial nor Helvetica and is a common production rejection. Fix via
`pdf.fonttype=42`, `ps.fonttype=42`, `font.family='Helvetica'`.
`architecture_horizontal_full.pdf` additionally mixes Times New Roman with Arial
plus an unnamed Type 3; `cytoverse_tutorial.pdf` sources its circled numerals
from MS-PGothic.

**Fonts are fine.** Arial resolves to a real TTF at
`/System/Library/Fonts/Supplemental/Arial.ttf` and Helvetica to
`/System/Library/Fonts/Helvetica.ttc`; matplotlib finds both. Figures use Arial,
Cell Press's stated preference.

## Source-of-truth gaps

- **Figs 5, 6** -- regenerable now from `notebooks/analysis.ipynb` cells 18/24
  against `data/ivfpq{8,16,32}x256`. Note cell 21 is a *markdown* duplicate of
  the distortion-grid code saving at `dpi=300` while live cell 24 saves at
  `dpi=150`; dedupe before rerunning.
- **Fig 2** -- source is `notebooks/figures/architecture_horizontal_full.svg`.
- **Fig 7** -- byte-identical to
  `.trees/benchmark/benchmark/m4/results/m4_benchmark_figure.pdf`, generated by
  `benchmark/m4/analyze.py:133` on the unmerged `benchmark` branch **with
  uncommitted modifications to analyze.py itself**. Nothing on `main` can
  rebuild it. Cherry-pick before proceeding.
- **Fig 1** -- fully reproducible: `public/sample.h5ad` + the `scimilarity`
  reference, both local. Captured, see Phase 1 status below.
- **Fig 3** -- query file `ExcitatoryNeurons_5000perLab.h5ad` is absent.
  Clearly NGN2-derived (the `-Brennand` suffix and `MG6` batch both appear in
  `data/ngn2_sspsygene_data_control.h5ad`, whose own barcodes lack the lab
  suffix, so it was built by concatenating per-lab subsets with
  `index_unique='-'`). But the name does not reproduce the figure: excitatory
  neurons capped at 5,000/lab gives Brennand 1,708 + Duan 5,000 + Wells 1,921 =
  **8,629 cells, not the 10,000 shown**. An equivalent 10,000-cell subsample can
  be built and documented; the original file cannot be recovered from disk.
- **Fig 4** -- query file unidentified and not on this machine. Note that
  `data/adata_metaatlas_final_raw.h5ad` indexes rows by stringified row number,
  so it "contains" cells `50530`/`73829` trivially by having >=73,830 rows --
  that is not evidence of provenance. Its `Type.v1` for those rows reads
  *Newborn Neuron* and *IPC*, contradicting the figure's *Inhibitory Neuron* and
  *Excitatory Neuron*.

  **This is the highest-risk item in the set.** `article.tex:178` quotes these
  two cells' importance scores verbatim -- ROBO2 0.0108, GRIK1 0.0103, LRP1B
  0.0064, RALYL 0.0057. Recapturing with different cells silently invalidates
  that paragraph. Perturbation importance is deterministic, so if the original
  file resurfaces, reproducing those four values is an exact pass/fail check.

## Phase 1 status: capture harness working, Figs 3/4 blocked on data

Run with:

    npx playwright test --config playwright.figures.config.ts

- `playwright.figures.config.ts` -- viewport 1600x900 at `deviceScaleFactor: 4`,
  headed so the scatterplot gets the real GPU (headless swiftshader renders
  points differently enough to matter).
- `tests/figures/helpers.ts` -- `capture()` reads the PNG IHDR back and throws
  unless dimensions are exactly viewport x 4, so an upscaled capture fails loudly
  rather than shipping a file that only looks like 300 dpi.
- `tests/figures/capture.spec.ts` -- Fig 1 working; Figs 3/4 stubbed.
- `src/ScatterPlotWebGL.vue` exposes the scatterplot on `window` under
  `import.meta.env.DEV` so cells can be selected by index instead of hunting
  canvas pixels. Stripped from production builds.

**Fig 1 captured: 6400x3600 = 934 dpi at 174 mm.** Verified genuinely rendered
at 4x (1:1 crops show clean text antialiasing and hard-edged WebGL points, not
resampling). State matches the published draft: scimilarity / prediction /
23,381,150 reference cells / 1,000,000 plotted / 74 labels / Total 1,000.

Two gotchas found the hard way, both now fixed in `helpers.ts`:

- **Vuetify overlay swallows the next click.** Clicking Start immediately after
  a dropdown selection lands on the closing scrim. It presents as "the run never
  started" with no error anywhere -- the first attempt sat for 15 minutes waiting
  on a status element that never rendered. `selectExact` now waits for
  `.v-overlay__content` to detach.
- **`hasText` substring-matches.** The existing e2e tests use
  `hasText: 'scimilarity'`, which selects **`scimilarity-subset`** -- the wrong
  239 MB reference instead of the 1.2 GB one. `selectExact` compares trimmed
  textContent exactly.

### Open items for Fig 1

- Viewport aspect is now a free parameter. At 1600x900 the figure is only
  **98 mm tall** at 174 mm wide, well under the 159 mm Phase 0 assumed from the
  old artwork. Pick the aspect deliberately before finalizing.
- Label counts differ slightly from the published draft (macrophage 440 vs 466,
  alveolar macrophage 245 vs 230). Labeling is not bit-reproducible -- expected
  with approximate IVFPQ search. No text quotes these numbers, so cosmetic, but
  confirm before swapping the figure.
- Callout overlay (circled numerals) still needs a reproducible compositing step.

## Remaining phases

1. ~~Recapture Figs 1, 3, 4~~ -- Fig 1 done. Figs 3 and 4 blocked, see
   "Source-of-truth gaps" above; neither query dataset is on this machine.
2. ~~Fix vector fonts and physical sizes~~ -- **done**, see Phase 2 status below.
3. **Reproducibility** -- cherry-pick the m4 benchmark work onto `main`; add a
   `make figures` target; remove `empty.eps`, `fig.eps` and the six uncited PDFs.
4. **Export and verify** -- vector to EPS via `pdftocairo -eps`, raster to TIFF
   via `pdftocairo -tiff -r 300` at the exact final width. Add a check script
   reporting px width, mm width, and effective dpi per deliverable.

Available tooling: `gs`, `dot`, `pdftocairo`, `tiffcp`, `exiftool`, `pdflatex`,
matplotlib 3.10.5 / PIL 11.3 in `.venv`. ImageMagick and Inkscape are absent.

## Phase 2 status: Figs 2, 5, 6, 7 regenerated and compliant

    uv run scripts/figures/compute_ivfpq_metrics.py   # once, caches to paper/figures/data/
    uv run scripts/figures/fig2_architecture.py
    uv run scripts/figures/fig5_recall_vs_probes.py
    uv run scripts/figures/fig6_distortion_grid.py
    uv run scripts/figures/fig7_performance_scaling.py

| Fig | Width | Height | Min type | Fonts | Was |
|-----|-------|--------|----------|-------|-----|
| 2 | 173.9 mm | 62.8 mm | 6.27 pt | ArialMT only | 2.3 pt, Times + Type 3 |
| 5 | 114.0 mm | 70.6 mm | 6.00 pt | ArialMT only | 8.2 pt, Type 3 DejaVu |
| 6 | 174.0 mm | 90.4 mm | 6.00 pt | ArialMT only | 4.5 pt, Type 3 DejaVu |
| 7 | 174.0 mm | 73.1 mm | 6.00 pt | ArialMT only | 4.4 pt, Type 3 DejaVu |

All four sit at their assigned width, carry a single embedded ArialMT, contain
no Type 3, and have no text below 6 pt. Verified by reading the PDF page box and
extracting every `Tf` operator from the content streams -- not by construction.

`scripts/figures/style.py` holds the shared rcParams and the width constants.

### Two correctness checks worth keeping

- **Fig 5 reproduces published Table 1.** The regenerated n_sub=32 recall curve
  matches `article.tex:203` (0.57 / 0.71 / 0.77 / 0.82 / 0.86 / 0.89 / 0.91 /
  0.91) to within 0.005. The compute path is sound.
- **Fig 7 re-derives the manuscript's fitted constants** and fails the build if
  any drifts >5% from `article.tex:255-257`. All six match: CPU 13 s + 14.6 s/1k,
  GPU 27 s + 12.1 s/1k, memory 2.7 GB + 4.0 KB/cell.

### Traps found while doing this

- **`dpi="150"` inflates graphviz PDFs by 150/72.** That alone is why Fig 2 was
  29.8 in wide and got crushed to 2.3 pt.
- **graphviz's cairo PDF backend emits a stray unnamed Type 3 font.** The macOS
  `-Tpdf:quartz:quartz` renderer produces a single ArialMT and no Type 3.
  Tradeoff: quartz omits the ToUnicode map, so text is not extractable.
- **graphviz's `size=` attribute did not scale the page** (asked 174 mm, got
  199 mm). Fig 2 is rescaled with Ghostscript instead.
- **Mathtext ignores `font.sans-serif`.** A single `$n_{sub}$` silently embedded
  DejaVuSans-Oblique. Fixed via `mathtext.fontset: custom` in style.py.
- **Mathtext sub/superscripts render at 0.7x**, so 6 pt base text put 4.2 pt
  glyphs on the page. All subscripts are now plain text; Fig 7 uses a literal
  U+00B2 for R-squared.
- **Arial lacks U+207B (superscript minus)**, so "x10^-3" box-glyphs. Fig 6
  scales distortion by 1e3 and puts "x 10^3" in the axis label instead.
- **Do not use `bbox_inches="tight"`** -- it trims the canvas so the PDF is
  narrower than the figsize requested, and LaTeX scales it back up, drifting the
  type size. Use `layout="constrained"`.

### Notes

- `scripts/ivfpq.py`'s `search()` signature is `(query_vector, model_path,
  n_probe, k)`. `analysis.ipynb` cell 11 still calls it with
  `n_partitions_search=` and **would fail against the current API** -- worth
  fixing in the notebook or retiring those cells in favour of these scripts.
- Fig 7's CPU runtime R-squared fits at 0.9996; `article.tex:255` says 0.9997.
  Rounding-level, affects no claim, but the text says 0.9997.
- Fig 7 still reads its inputs from `.trees/benchmark/...` pending the Phase 3
  cherry-pick.
