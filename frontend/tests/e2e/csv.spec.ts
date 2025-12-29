import { test, expect } from '@playwright/test';
import { parse } from 'csv-parse/sync';
import { readFileSync } from 'fs';

test('Navigate to root, run analysis, download and validate CSV', async ({ page }) => {
  test.setTimeout(4 * 60000); // Set timeout to 5 minutes for this test
  await page.goto('/');
  await page.getByTestId('radio-cpu-option').click();

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]');
  await fileInputElement.setInputFiles('fixtures/GSE136831_subsample_100.h5ad');

  await page.getByTestId('run-stop-button').click();
  await expect(page.getByTestId('status')).toHaveText(/Finished/, {
    timeout: 1 * 60000, // Wait for all cells to be labeled
  });

  // Setup download promise before clicking
  const downloadPromise = page.waitForEvent('download');

  // Click download button
  await page.getByTestId('download-button').click();

  // Wait for download to complete
  const download = await downloadPromise;

  // Save the downloaded file to a buffer
  const buffer = await download.createReadStream();
  let csvContent = '';

  for await (const chunk of buffer) {
    csvContent += chunk.toString();
  }

  // Parse CSV content
  const records: Array<{
    cell_id: string;
    category_label: string;
    confidence?: string;
  }> = parse(csvContent, {
    columns: true,
    skip_empty_lines: true,
  });

  // Load reference CSV file
  const referenceContent = readFileSync(
    'fixtures/GSE136831_subsample_100.labels.csv',
    'utf-8'
  );
  const referenceRecords: Array<{ cell_id: string; category_label: string }> = parse(
    referenceContent,
    {
      columns: true,
      skip_empty_lines: true,
    }
  );

  // Create a map of cell_id to category_label from reference file
  const referenceMap = new Map<string, string>();
  referenceRecords.forEach((record) => {
    referenceMap.set(record.cell_id, record.category_label);
  });

  // Validate that we have records
  expect(records.length).toBeGreaterThan(0);
  expect(records.length).toBe(referenceRecords.length);

  // Calculate percentage of matching labels
  let matchingCount = 0;
  let matchingWithConfidence1 = 0;
  let matchingWithConfidence09 = 0;
  let totalWithConfidence1 = 0;
  let totalWithConfidence09 = 0;
  const totalCount = records.length;

  records.forEach((record) => {
    expect(record).toHaveProperty('cell_id');
    expect(record).toHaveProperty('category_label');

    const cellId = record.cell_id;
    const actualLabel = record.category_label;
    const expectedLabel = referenceMap.get(cellId);
    const confidence = record.confidence ? parseFloat(record.confidence) : 1.0;

    // Verify the cell exists in reference
    expect(expectedLabel).toBeDefined();

    // Count totals at each confidence level
    if (confidence === 1.0) {
      totalWithConfidence1++;
    }
    if (confidence >= 0.9) {
      totalWithConfidence09++;
    }

    // Count matches
    if (actualLabel === expectedLabel) {
      matchingCount++;

      // Count matches with confidence = 1.0
      if (confidence === 1.0) {
        matchingWithConfidence1++;
      }

      // Count matches with confidence >= 0.9
      if (confidence >= 0.9) {
        matchingWithConfidence09++;
      }
    }
  });

  // Calculate match percentages
  const matchPercentage = (matchingCount / totalCount) * 100;
  const matchPercentageConf1 =
    totalWithConfidence1 > 0
      ? (matchingWithConfidence1 / totalWithConfidence1) * 100
      : 0;
  const matchPercentageConf09 =
    totalWithConfidence09 > 0
      ? (matchingWithConfidence09 / totalWithConfidence09) * 100
      : 0;

  console.log(
    `Total matching labels: ${matchingCount}/${totalCount} (${matchPercentage.toFixed(
      2
    )}%)`
  );
  console.log(
    `Matching with confidence = 1.0: ${matchingWithConfidence1}/${totalWithConfidence1} (${matchPercentageConf1.toFixed(
      2
    )}%)`
  );
  console.log(
    `Matching with confidence >= 0.9: ${matchingWithConfidence09}/${totalWithConfidence09} (${matchPercentageConf09.toFixed(
      2
    )}%)`
  );

  expect(matchPercentage).toBeGreaterThan(70);
  expect(matchPercentageConf09).toBeGreaterThan(80);
  expect(matchPercentageConf1).toBeGreaterThan(90);
});
