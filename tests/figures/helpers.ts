import { Page, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

export const OUT_DIR = path.resolve('paper/figures/captured');

/**
 * Pick an item from a Vuetify select by EXACT label.
 *
 * hasText does substring matching, which silently picks 'scimilarity-subset'
 * when you ask for 'scimilarity'. Match on trimmed textContent instead.
 */
export async function selectExact(page: Page, testId: string, label: string) {
  await page.getByTestId(testId).click();
  await page.waitForSelector('.v-list-item');
  const items = page.locator('.v-list-item');
  const n = await items.count();
  for (let i = 0; i < n; i++) {
    const item = items.nth(i);
    const text = ((await item.textContent()) ?? '').trim();
    if (text === label) {
      await item.click();
      // The Vuetify menu overlay animates out. Clicking Start while it is still
      // detaching silently lands on the scrim instead of the button, which
      // presents as "the run never started" with no error anywhere.
      await page
        .waitForSelector('.v-overlay__content', { state: 'detached', timeout: 30_000 })
        .catch(() => {});
      await page.waitForTimeout(500);
      return;
    }
  }
  const seen: string[] = [];
  for (let i = 0; i < n; i++) seen.push(((await items.nth(i).textContent()) ?? '').trim());
  throw new Error(`No option exactly matching "${label}" in ${testId}. Saw: ${seen.join(', ')}`);
}

/** Run the labeling pipeline and wait for it to report Finished. */
export async function runAndWait(page: Page, timeout = 15 * 60 * 1000) {
  await page.getByTestId('run-stop-button').click();
  await expect(page.getByTestId('status')).toHaveText(/Finished/, { timeout });
}

/** Wait until the WebGL scatterplot has been created and has drawn points. */
export async function waitForPlot(page: Page, timeout = 60_000) {
  await page.waitForFunction(
    () => Boolean((window as unknown as Record<string, unknown>).__cytoverseScatterplot),
    undefined,
    { timeout }
  );
  // regl draws asynchronously; give the first frame a beat to land.
  await page.waitForTimeout(1500);
}

/**
 * Select a point by its GLOBAL index in the scatterplot's column data
 * (baseRef points first, then userRef, then query). Fires the same 'select'
 * event a real click would, so the Cell Information popup opens identically.
 *
 * Relies on the DEV-only window hook in src/ScatterPlotWebGL.vue.
 */
export async function selectPoint(page: Page, globalIndex: number) {
  await page.evaluate((idx) => {
    const sp = (window as unknown as Record<string, unknown>).__cytoverseScatterplot as {
      select: (points: number[]) => void;
    };
    sp.select([idx]);
  }, globalIndex);
  await expect(page.locator('.cell-info-popup')).toBeVisible({ timeout: 30_000 });
}

/** Wait for the feature-importance gene list to finish computing. */
export async function waitForGenes(page: Page, timeout = 5 * 60 * 1000) {
  await expect(page.locator('.cell-info-popup .gene-item').first()).toBeVisible({ timeout });
  await expect(page.locator('.cell-info-popup .loading-state')).toHaveCount(0);
}

/** Read back the gene/importance pairs currently shown, for verification. */
export async function readGenes(page: Page): Promise<Array<{ gene: string; score: string }>> {
  return page.locator('.cell-info-popup .gene-item').evaluateAll((els) =>
    els.map((el) => ({
      gene: el.querySelector('.gene-name')?.textContent?.trim() ?? '',
      score: el.querySelector('.gene-score')?.textContent?.trim() ?? '',
    }))
  );
}

/** Save a full-viewport capture and assert it really came out at the expected scale. */
export async function capture(page: Page, name: string, expectedScale: number) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const file = path.join(OUT_DIR, `${name}.png`);
  await page.screenshot({ path: file, scale: 'device' });

  const viewport = page.viewportSize();
  if (!viewport) throw new Error('No viewport set; figure captures need an explicit one.');

  // Verify the PNG is genuinely at device scale rather than upscaled CSS pixels.
  const buf = fs.readFileSync(file);
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  const wantW = viewport.width * expectedScale;
  const wantH = viewport.height * expectedScale;
  if (width !== wantW || height !== wantH) {
    throw new Error(
      `${name}.png is ${width}x${height}, expected ${wantW}x${wantH} ` +
        `(viewport ${viewport.width}x${viewport.height} @ ${expectedScale}x)`
    );
  }
  // 174 mm = 6.8504 in
  const dpi = width / (174 / 25.4);
  console.log(`  ${name}.png  ${width}x${height}  =>  ${dpi.toFixed(0)} dpi at 174 mm`);
  return { file, width, height, dpi };
}
