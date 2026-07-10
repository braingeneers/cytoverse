// macOS (M4) driver for the CytoVerse app: one run of MODEL × DEVICE over an h5ad, capturing
// wall-clock (in-browser MutationObserver) and renderer-process RSS (macOS `ps`, sampled async).
// Drives the production build via `vite preview`. Env: BASE, MODEL, DEVICE(cpu|gpu), H5AD, HEADED.
import { chromium } from 'playwright';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const execFileP = promisify(execFile);

const BASE = process.env.BASE || 'http://localhost:4173/';
const MODEL = process.env.MODEL || 'scimilarity';
const DEVICE = process.env.DEVICE || 'cpu';
const H5AD = process.env.H5AD;
const HEADED = process.env.HEADED === '1';
const DEBUG_PORT = 9333;
if (!H5AD) { console.error('set H5AD'); process.exit(1); }

// macOS `ps`: rss in KB. Sum the renderer processes descending from the launched Chromium root
// (identified by our unique --remote-debugging-port). Renderer = tab footprint (main+worker+WASM).
async function psRows() {
  const { stdout } = await execFileP('ps', ['-Ao', 'pid=,ppid=,rss=,command='],
    { maxBuffer: 64 * 1024 * 1024, timeout: 5000 });
  return stdout.split('\n').map((l) => {
    const m = l.match(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
    return m ? { pid: +m[1], ppid: +m[2], rssKb: +m[3], cmd: m[4] } : null;
  }).filter(Boolean);
}
function rootPid(rows) {
  const r = rows.find((x) => x.cmd.includes(`--remote-debugging-port=${DEBUG_PORT}`) && !x.cmd.includes('--type='));
  return r ? r.pid : null;
}
async function rendererRss(root) {
  const rows = await psRows();
  const kids = new Map();
  for (const r of rows) { (kids.get(r.ppid) || kids.set(r.ppid, []).get(r.ppid)).push(r.pid); }
  const desc = new Set(); const st = [root];
  while (st.length) { const p = st.pop(); for (const c of kids.get(p) || []) if (!desc.has(c)) { desc.add(c); st.push(c); } }
  let renderer = 0, total = 0;
  for (const r of rows) if (desc.has(r.pid)) { total += r.rssKb * 1024; if (r.cmd.includes('--type=renderer')) renderer += r.rssKb * 1024; }
  return { renderer, total };
}
const MB = (b) => +(b / 1e6).toFixed(0);

const browser = await chromium.launch({
  headless: !HEADED,
  args: [`--remote-debugging-port=${DEBUG_PORT}`, '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
const eps = [];
page.on('console', (m) => { const t = m.text(); if (/embedding session|mapping session|WebGPU|WebAssembly|falling back/i.test(t)) eps.push(t); });

await page.goto(BASE);
await page.getByTestId('model-select-dropdown').click();
await page.waitForSelector('.v-list-item');
await page.getByText(MODEL, { exact: true }).first().click(); // exact so 'scimilarity' != 'scimilarity-subset'
await page.getByTestId(DEVICE === 'gpu' ? 'radio-gpu-option' : 'radio-cpu-option').click();

const rows0 = await psRows();
const root = rootPid(rows0);
if (!root) { console.error('could not find chromium root pid'); process.exit(2); }

await page.locator('[type="file"]').setInputFiles(H5AD);
await page.getByTestId('run-stop-button').waitFor({ state: 'visible' });
await page.evaluate(() => {
  const w = window; const el = document.querySelector('[data-testid="status"]');
  w.__s = performance.now(); w.__f = null;
  const obs = new MutationObserver(() => { if (w.__f === null && /Finished/.test(el?.textContent || '')) w.__f = performance.now(); });
  if (el) obs.observe(el, { childList: true, characterData: true, subtree: true });
});

const base = await rendererRss(root);
let peakR = 0, peakT = 0;
const t0 = Date.now();
await page.getByTestId('run-stop-button').click({ timeout: 180000 });
let finished = false, status = '';
const MAX_MS = parseInt(process.env.MAX_MS || String(20 * 60 * 1000), 10);
while (Date.now() - t0 < MAX_MS) {
  const r = await rendererRss(root);
  if (r.renderer > peakR) peakR = r.renderer;
  if (r.total > peakT) peakT = r.total;
  status = (await page.getByTestId('status').textContent().catch(() => ''))?.trim() || '';
  if (/Finished/.test(status)) { finished = true; break; }
  await page.waitForTimeout(500);
}
const wallMs = await page.evaluate(() => { const w = window; return (w.__f ?? performance.now()) - w.__s; });
let atRest = await rendererRss(root);
const settle = Date.now() + 5000;
while (Date.now() < settle) { atRest = await rendererRss(root); if (atRest.renderer > peakR) peakR = atRest.renderer; await page.waitForTimeout(500); }

const result = {
  model: MODEL, device: DEVICE, finished, status,
  wall_s: +(wallMs / 1000).toFixed(1),
  baseline_renderer_rss_mb: MB(base.renderer),
  peak_renderer_rss_mb: MB(peakR),
  at_rest_renderer_rss_mb: MB(atRest.renderer),
  peak_total_chrome_rss_mb: MB(peakT),
  embedding_ep_log: eps.find((l) => /embedding session/i.test(l)) || '(none)',
  fell_back_to_wasm: eps.some((l) => /falling back/i.test(l)),
};
console.log('M4_RESULT', JSON.stringify(result, null, 2));
await browser.close();
