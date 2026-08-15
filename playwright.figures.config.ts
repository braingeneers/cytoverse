import { defineConfig, devices } from '@playwright/test';

/**
 * Figure capture harness for the Cell Patterns submission.
 *
 * Separate from playwright.config.ts so the regular e2e suite keeps its
 * maximized-window / native-DPR settings. Here we pin an exact viewport and
 * device scale factor so captures are reproducible and high enough resolution
 * to clear 300 dpi at Cell Press column widths.
 *
 * See paper/FIGURES.md for the width assignment and dpi budget.
 *
 *   npx playwright test --config playwright.figures.config.ts
 */

/** 4x the CSS viewport. 1600 x 4 = 6400 px -> 934 dpi at 174 mm. */
export const CAPTURE_SCALE = 4;

/** CSS-pixel viewport. Chosen to give the scatterplot a 16:9-ish stage. */
export const VIEWPORT = { width: 1600, height: 900 };

export default defineConfig({
  testDir: './tests/figures',
  testMatch: 'tests/figures/*.spec.ts',
  fullyParallel: false,
  workers: 1,
  // Labeling 10k+ cells against a 23M-cell reference is minutes, not seconds.
  timeout: 20 * 60 * 1000,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'off',
    video: 'off',
    // Headed: we want the real GPU for the WebGL scatterplot, and headless
    // swiftshader renders points differently enough to matter for a figure.
    headless: false,
    viewport: VIEWPORT,
    deviceScaleFactor: CAPTURE_SCALE,
  },
  projects: [
    {
      name: 'figures',
      use: {
        ...devices['Desktop Chrome'],
        // Must come after the spread: Desktop Chrome pins 1280x720 @ 1x.
        viewport: VIEWPORT,
        deviceScaleFactor: CAPTURE_SCALE,
        launchOptions: {
          args: [
            `--window-size=${VIEWPORT.width + 20},${VIEWPORT.height + 120}`,
            '--hide-scrollbars',
            '--force-color-profile=srgb',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
