// Actual DOM and keyboard behavior; synthetic API responses, no DOM removal.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, '../../docs/full-spectrum-audit-20261010');
const base = process.env.BASE_URL || 'http://127.0.0.1:33619';
const browser = await chromium.launch({ headless: true }), results = [];
try {
  for (const [width, height] of [[375, 667], [414, 896], [768, 1024], [1280, 800], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height } });
    await context.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.origin !== base) return route.abort();
      if (!url.pathname.startsWith('/api/')) return route.continue();
      if (url.pathname.endsWith('/auth/me')) return route.fulfill({ status: 401, json: { detail: 'Synthetic guest' } });
      return route.fulfill({ status: 200, json: { models: [], sessions: [], files: [] } });
    });
    const page = await context.newPage(), errors = [], focus = [];
    await page.goto(base);
    await page.locator('.jru-initial-splash-overlay').waitFor({ state: 'hidden' });
    await page.getByRole('textbox', { name: 'Message input', exact: true }).waitFor();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    if (overflow) errors.push('Horizontal overflow');
    for (let n = 0; n < 8; n++) {
      await page.keyboard.press('Tab');
      const item = await page.evaluate(() => {
        const e = document.activeElement;
        return { tag: e.tagName, name: e.getAttribute('aria-label') || e.getAttribute('title') || e.textContent?.trim(), hiddenAncestor: !!e.closest('[aria-hidden="true"]') };
      });
      focus.push(item);
      if (item.hiddenAncestor) errors.push('Keyboard focus entered hidden content: ' + item.name);
    }
    await page.screenshot({ path: path.join(outDir, `viewports/independent-${width}.png`), fullPage: true });
    results.push({ width, height, overflow, focus, errors, passed: errors.length === 0 });
    await context.close();
  }
} finally {
  fs.writeFileSync(path.join(outDir, 'accessibility-independent.json'), JSON.stringify(results, null, 2));
  await browser.close();
}
console.log(JSON.stringify(results));
assert.ok(results.every(row => row.passed));
