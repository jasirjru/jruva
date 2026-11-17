// Regression for competing global shortcut handlers. APIs are synthetic.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, '../../docs/full-spectrum-audit-20261010');
const base = process.env.BASE_URL || 'http://127.0.0.1:33619';
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.origin !== base) return route.abort();
  if (!url.pathname.startsWith('/api/')) return route.continue();
  if (url.pathname.endsWith('/auth/me')) return route.fulfill({ status: 401, json: { detail: 'Synthetic guest' } });
  return route.fulfill({ status: 200, json: { models: [], sessions: [], files: [] } });
});
const page = await context.newPage(), results = [];
try {
  for (const key of ['Escape', 'Control+k']) {
    await page.goto(base);
    await page.getByRole('textbox', { name: 'Message input', exact: true }).waitFor();
    await page.keyboard.press('Control+k');
    await page.waitForFunction(() => document.activeElement?.classList.contains('cmd-palette-input'));
    await page.keyboard.press(key);
    try {
      await page.getByRole('dialog', { name: 'Global Workspace Command Palette' }).waitFor({ state: 'hidden', timeout: 2000 });
      results.push({ key, passed: true });
    } catch (error) { results.push({ key, passed: false, error: error.message }); }
  }
} finally {
  fs.writeFileSync(path.join(outDir, 'keyboard-results.json'), JSON.stringify(results, null, 2));
  await browser.close();
}
console.log(JSON.stringify(results));
assert.ok(results.every(result => result.passed));
