import { test, expect } from '@playwright/test'
import { parse } from 'csv-parse/sync'
import { readFileSync } from 'fs'

test('Navigate to root, run analysis, download and validate CSV', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  // const fileInputElement = page.locator('[type="file"]')
  // await fileInputElement.setInputFiles('tests/GSE136831_subsample_100.h5ad')

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
  const records: Array<{ cell_id: string; category_label: string }> = parse(csvContent, {
    columns: true,
    skip_empty_lines: true,
  })

  // Load reference CSV file
  const referenceContent = readFileSync('web/tests/sample.labels.csv', 'utf-8')
  const referenceRecords: Array<{ cell_id: string; category_label: string }> = parse(
    referenceContent,
    {
      columns: true,
      skip_empty_lines: true,
    }
  )

  // Create a map of cell_id to category_label from reference file
  const referenceMap = new Map<string, string>()
  referenceRecords.forEach((record) => {
    referenceMap.set(record.cell_id, record.category_label)
  })

  // Validate that we have records
  expect(records.length).toBeGreaterThan(0)
  expect(records.length).toBe(referenceRecords.length)

  // Calculate percentage of matching labels
  let matchingCount = 0
  const totalCount = records.length

  records.forEach((record) => {
    expect(record).toHaveProperty('cell_id')
    expect(record).toHaveProperty('category_label')

    const cellId = record.cell_id
    const actualLabel = record.category_label
    const expectedLabel = referenceMap.get(cellId)

    // Verify the cell exists in reference
    expect(expectedLabel).toBeDefined()

    // Count matches
    if (actualLabel === expectedLabel) {
      matchingCount++
    }
  })

  // Calculate match percentage
  const matchPercentage = (matchingCount / totalCount) * 100

  console.log(`Matching labels: ${matchingCount}/${totalCount} (${matchPercentage.toFixed(2)}%)`)

  // Test passes if match percentage is > 70%
  expect(matchPercentage).toBeGreaterThan(70)
})
