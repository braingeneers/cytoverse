import { test, expect } from '@playwright/test';

test('Calculate feature importance for a query cell', async ({ page }) => {
  await page.goto('/');

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]');
  await fileInputElement.setInputFiles('fixtures/GSE136831_subsample_100.h5ad');

  await page.getByTestId('run-stop-button').click();

  await expect(page.getByTestId('status')).toHaveText(/Finished/, {
    timeout: 60000, // Wait up to 60 seconds for the status to change
  });

  // Click on a query cell in the scatter plot
  // Use the exposed selectPoint method from the ScatterPlotWebGL component
  // Query cells start at index 1,000,000 (after the 1M reference cells)
  const wrapper = page.locator('.scatterplot-container');

  await wrapper.evaluate((el) => {
    // Vue 3 stores exposed properties on __vueParentComponent or __vue__
    const vueEl = el as Element & {
      __vueParentComponent?: {
        exposed?: { selectPoint?: (index: number) => void };
      };
      __vue__?: { exposed?: { selectPoint?: (index: number) => void } };
    };
    const component = vueEl.__vueParentComponent || vueEl.__vue__;

    if (!component?.exposed?.selectPoint) {
      throw new Error('selectPoint method not exposed!');
    }

    // Select the first query cell (index 1,000,000 = first cell after 1M reference cells)
    component.exposed.selectPoint(1000000);
  });

  // Wait for the cell info popup to appear (with longer timeout for initial click detection)
  await expect(page.locator('.cell-info-popup')).toBeVisible({ timeout: 10000 });

  // Verify the popup shows cell information
  await expect(page.locator('.popup-title')).toHaveText('Cell Information');

  // Wait for feature importance calculation to complete (up to 30 seconds)
  await expect(page.locator('.gene-list')).toBeVisible({ timeout: 30000 });

  // Verify that gene names are displayed in the list
  const geneItems = page.locator('.gene-item');
  await expect(geneItems.first()).toBeVisible();

  // Verify we have 10 genes (default topN)
  const geneCount = await geneItems.count();
  expect(geneCount).toBe(10);

  // Verify the top 3 genes and their importance scores
  const gene1 = geneItems.nth(0);
  await expect(gene1.locator('.gene-rank')).toHaveText('1');
  await expect(gene1.locator('.gene-name')).toHaveText('IL1R2');
  await expect(gene1.locator('.gene-score')).toHaveText('0.0147');

  const gene2 = geneItems.nth(1);
  await expect(gene2.locator('.gene-rank')).toHaveText('2');
  await expect(gene2.locator('.gene-name')).toHaveText('VCAN');
  await expect(gene2.locator('.gene-score')).toHaveText('0.0141');

  const gene3 = geneItems.nth(2);
  await expect(gene3.locator('.gene-rank')).toHaveText('3');
  await expect(gene3.locator('.gene-name')).toHaveText('GPX1');
  await expect(gene3.locator('.gene-score')).toHaveText('0.0135');
});
