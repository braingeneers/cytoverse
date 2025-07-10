import { test } from '@playwright/test'

test('Navigate to root, click run, wait for results', async ({ page }) => {
  await page.goto('/')

  await page.getByTestId('radio-cpu-option').check()

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  const fileInputElement = page.locator('[type="file"]')
  await fileInputElement.setInputFiles('tests/GSE136831_subsample_10.h5ad')

  await page.getByTestId('run-stop-button').click()
  // await page.waitForTimeout(1000)
  // await page.getByTestId('run-stop-button').click()
})
