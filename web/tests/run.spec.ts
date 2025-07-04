import { test } from '@playwright/test'

test('Navigate to root, click run, wait for results', async ({ page }) => {
  await page.goto('/')

  await page.getByTestId('radio-cpu-option').check()

  await page.getByTestId('run-stop-button').click()
  // await page.waitForTimeout(1000)
  // await page.getByTestId('run-stop-button').click()
})
