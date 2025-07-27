import { test, expect } from '@playwright/test'
import { parse } from 'csv-parse/sync'

test('Navigate to root, run analysis, download and validate CSV', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]')
  await fileInputElement.setInputFiles('tests/GSE136831_subsample_10.h5ad')

  await page.getByTestId('run-stop-button').click()
  await expect(page.locator('[data-cy="status"]')).toHaveText(/Complete/, {
    timeout: 60000, // Wait up to 60 seconds for the status to change
  })

  // Setup download promise before clicking
  const downloadPromise = page.waitForEvent('download')
  
  // Click download button
  await page.getByTestId('download-button').click()
  
  // Wait for download to complete
  const download = await downloadPromise
  
  // Save the downloaded file to a buffer
  const buffer = await download.createReadStream()
  let csvContent = ''
  
  for await (const chunk of buffer) {
    csvContent += chunk.toString()
  }
  
  // Parse CSV content
  const records = parse(csvContent, {
    columns: true,
    skip_empty_lines: true
  })
  
  // Validate CSV structure and content
  expect(records.length).toBeGreaterThan(0)
  
  // Check that each record has expected fields
  records.forEach((record: any) => {
    expect(record).toHaveProperty('barcode')
    expect(record).toHaveProperty('prediction')
    expect(record).toHaveProperty('probability')
    
    // Validate data types
    expect(typeof record.barcode).toBe('string')
    expect(typeof record.prediction).toBe('string')
    expect(parseFloat(record.probability)).not.toBeNaN()
    
    // Validate probability is between 0 and 1
    const probability = parseFloat(record.probability)
    expect(probability).toBeGreaterThanOrEqual(0)
    expect(probability).toBeLessThanOrEqual(1)
  })
  
  // Validate that predictions are valid cell types
  const predictions = records.map((r: any) => r.prediction)
  const uniquePredictions = [...new Set(predictions)]
  expect(uniquePredictions.length).toBeGreaterThan(0)
})