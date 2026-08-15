// Reset-button validation: run a labeling job, record at-rest renderer RSS, click the Reset button,
// let it settle, and report how much memory the reset releases. Optionally re-run (RERUN=1) to
// confirm the next run re-downloads the model + index into a fresh worker.
// Env: BASE, MODEL, DEVICE(cpu|gpu), H5AD (required), HEADED (always use 1 on macOS), MAX_MS, RERUN.
import { chromium } from 'playwright';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const execFileP = promisify(execFile);

const BASE = process.env.BASE || 'http://localhost:4173/';
const MODEL = process.env.MODEL || 'scimilarity';
const DEVICE = process.env.DEVICE || 'cpu';
const H5AD = process.env.H5AD;
const HEADED = process.env.HEADED !== '0';
const RERUN = process.env.RERUN === '1';
const DEBUG_PORT = 9334;
if (!H5AD) { console.error('set H5AD'); process.exit(1); }

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

async function runOnce(page, root, label) {
  await page.evaluate(() => {
    const w = window; const el = document.querySelector('[data-testid="status"]');
    w.__s = performance.now(); w.__f = null;
    const obs = new MutationObserver(() => { if (w.__f === null && /Finished/.test(el?.textContent || '')) w.__f = performance.now(); });
    if (el) obs.observe(el, { childList: true, characterData: true, subtree: true });
  });
  let peakR = 0;
  const t0 = Date.now();
  await page.getByTestId('run-stop-button').click({ timeout: 180000 });
  const MAX_MS = parseInt(process.env.MAX_MS || String(40 * 60 * 1000), 10);
  let finished = false, status = '';
  while (Date.now() - t0 < MAX_MS) {
    const r = await rendererRss(root); if (r.renderer > peakR) peakR = r.renderer;
    status = (await page.getByTestId('status').textContent().catch(() => ''))?.trim() || '';
    if (/Finished/.test(status)) { finished = true; break; }
    await page.waitForTimeout(500);
  }
  // settle at-rest
  let atRest = await rendererRss(root);
  const settle = Date.now() + 6000;
  while (Date.now() < settle) { atRest = await rendererRss(root); if (atRest.renderer > peakR) peakR = atRest.renderer; await page.waitForTimeout(500); }
  const wallMs = await page.evaluate(() => { const w = window; return (w.__f ?? performance.now()) - w.__s; });
  console.log(`  [${label}] finished=${finished} wall=${(wallMs/1000).toFixed(1)}s peak=${MB(peakR)}MB at_rest=${MB(atRest.renderer)}MB status="${status}"`);
  return { finished, wall_s: +(wallMs/1000).toFixed(1), peak: MB(peakR), at_rest: MB(atRest.renderer) };
}

const browser = await chromium.launch({
  headless: !HEADED,
  args: [`--remote-debugging-port=${DEBUG_PORT}`, '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
const eps = [];
page.on('console', (m) => { const t = m.text(); if (/embedding session|Loading|index|WebGPU|WebAssembly|falling back|reset/i.test(t)) eps.push(t); });

await page.goto(BASE);
await page.getByTestId('model-select-dropdown').click();
await page.waitForSelector('.v-list-item');
await page.getByText(MODEL, { exact: true }).first().click();
await page.getByTestId(DEVICE === 'gpu' ? 'radio-gpu-option' : 'radio-cpu-option').click();

const root = rootPid(await psRows());
if (!root) { console.error('no chromium root pid'); process.exit(2); }
await page.locator('[type="file"]').setInputFiles(H5AD);
await page.getByTestId('run-stop-button').waitFor({ state: 'visible' });

const baseline = MB((await rendererRss(root)).renderer);
console.log(`baseline_renderer=${baseline}MB`);

const run1 = await runOnce(page, root, 'run#1');

// ---- click Reset, settle, measure release ----
await page.getByTestId('reset-button').waitFor({ state: 'visible', timeout: 10000 });
await page.getByTestId('reset-button').click();
let post = await rendererRss(root), postMin = post.renderer;
const settle = Date.now() + 10000; // give the OS time to reclaim the terminated worker's pages
while (Date.now() < settle) { post = await rendererRss(root); if (post.renderer < postMin) postMin = post.renderer; await page.waitForTimeout(500); }
const resetStatus = (await page.getByTestId('status').textContent().catch(() => ''))?.trim() || '';
console.log(`  [reset] post_reset=${MB(post.renderer)}MB (min ${MB(postMin)}MB) status="${resetStatus}"`);

let run2 = null;
if (RERUN) {
  await page.getByTestId('run-stop-button').waitFor({ state: 'visible' });
  run2 = await runOnce(page, root, 'run#2 (after reset)');
}

const result = {
  model: MODEL, device: DEVICE, cells_file: H5AD.split('/').pop(),
  baseline_mb: baseline,
  run1_at_rest_mb: run1.at_rest, run1_peak_mb: run1.peak, run1_wall_s: run1.wall_s,
  post_reset_mb: MB(post.renderer), post_reset_min_mb: MB(postMin),
  released_mb: run1.at_rest - MB(post.renderer),
  released_vs_min_mb: run1.at_rest - MB(postMin),
  reset_status: resetStatus,
  run2_reloaded: run2 ? (eps.filter((l) => /Loading|index/i.test(l)).length > 0) : null,
  run2_wall_s: run2 ? run2.wall_s : null,
};
console.log('RESET_RESULT', JSON.stringify(result, null, 2));
await browser.close();
