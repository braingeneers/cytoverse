import { test, expect } from '@playwright/test'

test('Navigate to root, do NOT wait for results', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]')
  await fileInputElement.setInputFiles('tests/GSE136831_subsample_10.h5ad')
  await page.getByTestId('run-stop-button').click()
})

test('Navigate to root, click run, wait for results', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]')
  await fileInputElement.setInputFiles('tests/GSE136831_subsample_100.h5ad')

  await page.getByTestId('run-stop-button').click()
  await expect(page.getByTestId('status')).toHaveText(/Complete/, {
    timeout: 60000, // Wait up to 60 seconds for the status to change
  })
})

test('Open non-h5ad file and show error', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]')
  await fileInputElement.setInputFiles('tests/not-anndata.h5ad')
  await page.getByTestId('run-stop-button').click()
  await expect(page.getByTestId('error-title')).toHaveText(/Failed to extract/, {
    timeout: 60000, // Wait up to 60 seconds for the status to change
  })
})
