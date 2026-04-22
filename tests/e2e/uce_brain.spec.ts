import { test, expect } from '@playwright/test';

test('UCE-brain model appears in dropdown', async ({ page }) => {
  await page.goto('/');
  const modelSelector = await page.getByTestId('model-select-dropdown');
  await modelSelector.click();
  await page.waitForSelector('.v-list-item');
  await expect(
    page.locator('.v-list-item', { hasText: 'SSPsyGene - UCE Brain' })
  ).toBeVisible();
});

test('UCE-brain labels 10 cells end-to-end', async ({ page }) => {
  test.setTimeout(300_000); // 5 min — first-run 400 MB download + per-cell inference
  const consoleMsgs: string[] = [];
  page.on('console', (msg) => {
    consoleMsgs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on('pageerror', (err) => consoleMsgs.push(`[pageerror] ${err.message}\n${err.stack}`));

  await page.goto('/');

  const modelSelector = await page.getByTestId('model-select-dropdown');
  await modelSelector.click();
  await page.waitForSelector('.v-list-item');
  await page
    .locator('.v-list-item', { hasText: 'SSPsyGene - UCE Brain' })
    .click();

  const fileInputElement = page.locator('[type="file"]');
  await fileInputElement.setInputFiles(
    'tests/fixtures/GSE136831_subsample_10.h5ad'
  );

  await page.getByTestId('run-stop-button').click();

  try {
    await expect(page.getByTestId('status')).toHaveText(/Finished/, {
      timeout: 180000, // 3 min — protein embeddings are 400 MB first time
    });
  } finally {
    console.log('--- Captured console output ---');
    for (const m of consoleMsgs) console.log(m);
    console.log('--- End console output ---');
  }
});
