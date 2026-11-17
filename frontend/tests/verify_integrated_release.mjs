/** Static production-build preview + real isolated backend; no external requests.
 * Start the synthetic backend on 127.0.0.1:48991 before running this script.
 * This does not verify Next.js runtime rewrites or hosted model generation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.dirname(frontend);
const identity = JSON.parse(fs.readFileSync(path.join(root, 'scratch/browser-identity.json'), 'utf8'));
const results = { mode: 'static-build-preview-real-local-api', checks: [], externalRequestsBlocked: 0 };
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  const oldFiles = await context.request.get('http://127.0.0.1:48991/api/v1/library/files', { headers: { Authorization: `Bearer ${identity.token}` } });
  for (const file of (await oldFiles.json()).files) {
    if (['browser_release.txt', 'browser_corrupt.pdf'].includes(file.filename)) {
      await context.request.delete(`http://127.0.0.1:48991/api/v1/library/files/${file.id}`, { headers: { Authorization: `Bearer ${identity.token}` } });
    }
  }
  await context.addCookies([{ name: 'access_token', value: identity.token, url: 'http://127.0.0.1:48992', httpOnly: true, sameSite: 'Lax' }]);
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.host !== '127.0.0.1:48992') {
      results.externalRequestsBlocked++;
      return route.abort();
    }
    if (url.pathname.startsWith('/api/v1/')) {
      if (url.pathname === '/api/v1/auth/logout') {
        await context.clearCookies();
      }
      try {
        const response = await route.fetch({ url: `http://127.0.0.1:48991${url.pathname}${url.search}` });
        return route.fulfill({ response });
      } catch {
        return route.abort();
      }
    }
    let file;
    if (url.pathname === '/') file = path.join(frontend, '.next/server/app/index.html');
    else if (url.pathname.startsWith('/_next/static/')) {
      file = path.resolve(frontend, '.next/static', url.pathname.slice('/_next/static/'.length));
      assert(file.startsWith(path.resolve(frontend, '.next/static') + path.sep));
    }
    if (!file || !fs.existsSync(file)) return route.fulfill({ status: 404, body: '' });
    const type = file.endsWith('.js') ? 'application/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.html') ? 'text/html' : 'application/octet-stream';
    return route.fulfill({ status: 200, contentType: type, body: fs.readFileSync(file) });
  });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:48992');
  await page.getByRole('button', { name: 'Library', exact: true }).click();
  await page.getByText('Your Knowledge Library is empty.', { exact: false }).waitFor();
  results.checks.push('library navigation and empty state');
  await page.locator('input[type=file]').setInputFiles({ name: 'browser_release.txt', mimeType: 'text/plain', buffer: Buffer.from('Synthetic browser beacon BV621 opens on 21 May 2034.') });
  const row = page.locator('tr').filter({ hasText: 'browser_release.txt' });
  await row.getByText('✓ INDEXED', { exact: true }).waitFor();
  results.checks.push('real text upload and indexed status');
  const retrieval = await context.request.get('http://127.0.0.1:48991/api/v1/library/search?q=BV621', { headers: { Authorization: `Bearer ${identity.token}` } });
  assert.equal(retrieval.status(), 200);
  const retrieved = await retrieval.json();
  assert(retrieved.results.some(item => item.text.includes('21 May 2034')));
  results.retrieval = retrieved.results.map(({ text, doc_name, page, file_id }) => ({ text, doc_name, page, file_id }));
  results.checks.push('real dense/BM25 retrieval through local API');
  await page.locator('input[type=file]').setInputFiles({ name: 'browser_corrupt.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\ncorrupt synthetic binary\x00\xff') });
  const failed = page.locator('tr').filter({ hasText: 'browser_corrupt.pdf' });
  await failed.getByText('⚠️ FAILED', { exact: true }).waitFor();
  await failed.getByRole('button', { name: '🔄 Retry', exact: true }).click();
  await page.getByText('⚠️ Reindex error:', { exact: false }).waitFor();
  results.checks.push('corrupted PDF failed state and retry failure');
  await page.screenshot({ path: path.join(root, 'scratch/browser-library.png'), fullPage: true });
  await row.getByTitle('Delete document', { exact: true }).click();
  await row.waitFor({ state: 'detached' });
  results.checks.push('document deletion updates library');
  await page.getByRole('button', { name: 'User profile menu', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Log Out', exact: true }).click();
  await failed.waitFor({ state: 'detached', timeout: 5000 });
  results.checks.push('logout clears private library state');
  await page.unrouteAll({ behavior: 'ignoreErrors' });
  await context.close();
  results.status = 'passed';
} catch (error) {
  results.status = 'failed';
  results.error = String(error);
  process.exitCode = 1;
} finally {
  await browser.close();
  fs.writeFileSync(path.join(root, 'scratch/browser-results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
}
