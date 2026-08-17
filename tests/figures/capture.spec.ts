import { test } from '@playwright/test';
import { CAPTURE_SCALE } from '../../playwright.figures.config';
import {
  selectExact,
  runAndWait,
  waitForPlot,
  selectPoint,
  waitForGenes,
  readGenes,
  capture,
} from './helpers';
import { measureCallouts } from './callouts';

/**
 * Base captures for the Cell Patterns figures. These produce clean, high-DPI
 * screenshots; callouts and multi-panel composition are applied afterwards by
 * the compositing step so that annotation stays reproducible.
 *
 * See paper/FIGURES.md for width assignments and the dpi budget.
 */

// Figure 1 -- UI guide. Reference 'scimilarity', category 'prediction',
// query is public/sample.h5ad which the app auto-loads on startup.
test('fig1: UI guide base capture', async ({ page }) => {
  await page.goto('/');

  await selectExact(page, 'model-select-dropdown', 'scimilarity');
  await selectExact(page, 'category-select-dropdown', 'prediction');

  await runAndWait(page);
  await waitForPlot(page);

  await capture(page, 'fig1_ui_guide', CAPTURE_SCALE);
  // Callout positions are measured from the live DOM and overlaid in LaTeX, so
  // the numbers stay vector text and track the UI if it is ever recaptured.
  await measureCallouts(page, 'fig1_ui_guide');
});

// Figure 3 -- NGN2 neurons against the SSPsyGene reference.
//
// BLOCKED: the query file shown in the published draft,
// 'ExcitatoryNeurons_5000perLab.h5ad', is not in the repo or on disk. Its
// barcodes carry a lab suffix ('MG6_ACGCAGCCAAGCTGTT-Brennand') absent from
// data/ngn2_sspsygene_data_control.h5ad ('MG1_AAAGCAAGTCTAGTGT'), so it was
// built by concatenating per-lab excitatory-neuron subsets with
// index_unique='-'. Reconstructable, but the subsample seed is unknown.
test.skip('fig3: sspsygene labeling base capture', async ({ page }) => {
  await page.goto('/');

  await page.locator('[type="file"]').setInputFiles('data/figures/ExcitatoryNeurons_5000perLab.h5ad');
  await selectExact(page, 'model-select-dropdown', 'sspsygene');
  await selectExact(page, 'category-select-dropdown', 'Type.v1');

  await runAndWait(page);
  await waitForPlot(page);

  await capture(page, 'fig3_sspsygene', CAPTURE_SCALE);
});

// Figure 4 -- feature importance for one inhibitory and one excitatory cell.
//
// The Results text quotes these cells' importance scores verbatim (ROBO2
// 0.0108, GRIK1 0.0103, LRP1B 0.0064, RALYL 0.0057), so the recapture MUST use
// the same two cells -- IDs 50530 and 73829 -- or the text becomes wrong.
// Perturbation importance is deterministic (zero out each expressed gene,
// measure L2 drift), so reproducing those exact values is the pass/fail test
// for a faithful recapture.
//
// BLOCKED: the source query file is unidentified. Note that
// data/adata_metaatlas_final_raw.h5ad uses stringified row numbers as its
// index, so it "contains" 50530/73829 trivially -- that is NOT evidence of
// provenance, only that the file has >=73830 rows. Its Type.v1 annotations for
// those rows (Newborn Neuron, IPC) also disagree with the labels in the
// published figure (Inhibitory Neuron, Excitatory Neuron), so the query most
// likely came from a different dataset that is not on this machine.
test.skip('fig4: feature importance panels', async ({ page }) => {
  await page.goto('/');

  await page.locator('[type="file"]').setInputFiles('data/figures/fig4_importance_query.h5ad');
  await selectExact(page, 'model-select-dropdown', 'sspsygene');
  await selectExact(page, 'category-select-dropdown', 'Type.v1');

  await runAndWait(page);
  await waitForPlot(page);

  // Indices are resolved at capture time from the query cell names; see the
  // compositing script. Each panel is captured separately and composed after.
  for (const [name, index] of [
    ['fig4_inhibitory_50530', -1],
    ['fig4_excitatory_73829', -1],
  ] as Array<[string, number]>) {
    await selectPoint(page, index);
    await waitForGenes(page);
    console.log(name, await readGenes(page));
    await capture(page, name, CAPTURE_SCALE);
  }
});
