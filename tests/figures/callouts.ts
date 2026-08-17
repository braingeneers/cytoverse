import { Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { OUT_DIR } from './helpers';

/**
 * Figure 1 callout anchors.
 *
 * The numbered markers are drawn in LaTeX over the screenshot rather than baked
 * into it (see paper/FIGURES.md). Positions are read from the live DOM and
 * written as viewport fractions, so re-running the capture at a different size
 * regenerates correct coordinates instead of silently shifting the markers off
 * their targets.
 *
 * `place` says which edge of the element the marker sits on, so a marker never
 * covers the control it is pointing at.
 */
export interface Callout {
  n: number;
  selector: string;
  /**
   * Where the marker sits relative to the element. `inside-left` keeps a marker
   * on controls that are flush against the sidebar edge, where an outside-left
   * marker would fall off the figure.
   */
  place:
    | 'left'
    | 'inside-left'
    | 'inside-right'
    | 'right'
    | 'above'
    | 'below'
    | 'center';
  /** Vertical anchor. Use 'top' for elements taller than the viewport. */
  anchorY?: 'top' | 'center';
  /** Extra nudges in CSS px, to separate markers or aim them at content. */
  dx?: number;
  dy?: number;
  legend: string;
}

export const CALLOUTS: Callout[] = [
  { n: 1, selector: '[title="Select .h5ad Query File"]', place: 'right',
    legend: 'Select \\texttt{.h5ad} file from local disk' },
  { n: 2, selector: '[data-testid="run-stop-button"]', place: 'right',
    legend: 'Click to begin embedding and labeling' },
  // 3 appears twice in the original: the plot and the label tally update together.
  { n: 3, selector: 'canvas.scatterplot-canvas', place: 'center', dx: -20, dy: 110,
    legend: 'Labels and UMAP points appear in real time' },
  { n: 3, selector: '.labels-section', place: 'right', anchorY: 'top', dy: 80,
    legend: '' },
  { n: 4, selector: '[data-testid="download-button"]', place: 'right',
    legend: 'Export CSV with cell id, label and confidence' },
  { n: 5, selector: '[data-testid="create-index-button"]', place: 'right',
    legend: 'Export embeddings JSON to share' },
  { n: 6, selector: '[data-testid="model-select-dropdown"]', place: 'inside-right',
    legend: 'Change the base reference embeddings' },
  { n: 7, selector: '[data-testid="category-select-dropdown"]', place: 'inside-right',
    legend: 'Change the label category' },
  { n: 8, selector: '[data-testid="import-index-button"]', place: 'right',
    legend: 'Import user reference JSON' },
  { n: 9, selector: '.legend', place: 'right',
    legend: 'UMAP legend: base, user and query' },
];

/** Read each anchor's position from the DOM as a fraction of the viewport. */
export async function measureCallouts(page: Page, name: string) {
  const viewport = page.viewportSize();
  if (!viewport) throw new Error('Callout measurement needs an explicit viewport.');

  const anchors: Array<{ n: number; x: number; y: number; legend: string }> = [];
  for (const c of CALLOUTS) {
    const el = page.locator(c.selector).first();
    if ((await el.count()) === 0) {
      throw new Error(`Callout ${c.n}: no element matches ${c.selector}`);
    }
    const box = await el.boundingBox();
    if (!box) throw new Error(`Callout ${c.n}: ${c.selector} is not visible`);

    // Offset so the marker sits beside its target rather than on top of it.
    const pad = 18;
    let x = box.x + box.width / 2;
    let y = c.anchorY === 'top' ? box.y : box.y + box.height / 2;
    if (c.place === 'left') x = box.x - pad;
    if (c.place === 'inside-left') x = box.x + pad;
    if (c.place === 'inside-right') x = box.x + box.width - pad;
    if (c.place === 'right') x = box.x + box.width + pad;
    if (c.place === 'above') y = box.y - pad;
    if (c.place === 'below') y = box.y + box.height + pad;
    x += c.dx ?? 0;
    y += c.dy ?? 0;

    // A marker outside the frame is invisible in the figure but produces no
    // error anywhere downstream, so fail loudly here instead.
    const fx = x / viewport.width;
    const fy = y / viewport.height;
    if (fx < 0.01 || fx > 0.99 || fy < 0.01 || fy > 0.99) {
      throw new Error(
        `Callout ${c.n} (${c.selector}) lands at (${fx.toFixed(3)}, ` +
          `${fy.toFixed(3)}), outside the captured frame`
      );
    }

    anchors.push({
      n: c.n,
      x: +(x / viewport.width).toFixed(4),
      y: +(y / viewport.height).toFixed(4),
      legend: c.legend,
    });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const file = path.join(OUT_DIR, `${name}.callouts.json`);
  fs.writeFileSync(file, JSON.stringify({ viewport, anchors }, null, 2));
  console.log(`  ${name}.callouts.json  ${anchors.length} anchors`);
  return anchors;
}
