import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_DIR = path.resolve(__dirname, '..');

async function runReproduction() {
  console.log('='.repeat(70));
  console.log('REPRODUCING PRODUCTION SESSION REQUEST STORM (2 SAME-ORIGIN TABS)');
  console.log('='.repeat(70));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });

  let sessionRequestCount = 0;
  const sessionRequestsPerTab = { tab1: 0, tab2: 0 };
  let testStartTime = Date.now();

  await context.route('**/*', async (route) => {
    const url = new URL(route.request().url());

    if (url.pathname.startsWith('/api/v1/')) {
      const p = url.pathname;
      if (p === '/api/v1/sessions') {
        sessionRequestCount++;
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ sessions: [] }),
        });
      }
      if (p === '/api/v1/auth/me') {
        return route.fulfill({ status: 401, contentType: 'application/json', body: '{"detail":"Unauthenticated"}' });
      }
      if (p === '/api/v1/models') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            models: [{ id: 'auto', name: 'Auto', provider: 'auto', recommended: true, available: true }],
          }),
        });
      }
      if (p === '/api/v1/library/files') {
        return route.fulfill({ status: 200, contentType: 'application/json', body: '{"files":[]}' });
      }
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }

    // Static assets
    let filePath;
    if (url.pathname === '/') {
      filePath = path.join(FRONTEND_DIR, '.next', 'server', 'app', 'index.html');
    } else if (url.pathname.startsWith('/_next/static/')) {
      const rel = url.pathname.slice('/_next/static/'.length);
      filePath = path.resolve(FRONTEND_DIR, '.next', 'static', rel);
    }

    if (filePath && fs.existsSync(filePath)) {
      const type = filePath.endsWith('.js')
        ? 'application/javascript'
        : filePath.endsWith('.css')
        ? 'text/css'
        : filePath.endsWith('.html')
        ? 'text/html'
        : 'application/octet-stream';
      return route.fulfill({ status: 200, contentType: type, body: fs.readFileSync(filePath) });
    }

    return route.fulfill({ status: 404, body: '' });
  });

  console.log('Launching Tab 1...');
  const page1 = await context.newPage();
  await page1.goto('http://127.0.0.1:48992', { waitUntil: 'domcontentloaded' });
  const splash1 = page1.locator('.jru-initial-splash-overlay');
  if (await splash1.isVisible().catch(() => false)) {
    await splash1.click();
  }
  await page1.waitForTimeout(1000);
  const countAfterTab1 = sessionRequestCount;
  console.log(`Requests after Tab 1 mount: ${countAfterTab1} /api/v1/sessions calls`);

  console.log('Launching Tab 2 in the same browser context (same origin)...');
  const page2 = await context.newPage();
  await page2.goto('http://127.0.0.1:48992', { waitUntil: 'domcontentloaded' });
  const splash2 = page2.locator('.jru-initial-splash-overlay');
  if (await splash2.isVisible().catch(() => false)) {
    await splash2.click();
  }

  console.log('Triggering New Chat in Tab 1 (writing new chatbot_session_id)...');
  await page1.evaluate(() => {
    localStorage.setItem('chatbot_session_id', crypto.randomUUID());
  });

  console.log('Observing 2-tab interaction for 5 seconds...');
  const startObsCount = sessionRequestCount;
  const startObsTime = Date.now();
  await page1.waitForTimeout(5000);
  const endObsCount = sessionRequestCount;
  const elapsedSec = ((Date.now() - startObsTime) / 1000).toFixed(1);

  const deltaCalls = endObsCount - startObsCount;
  const ratePerSec = (deltaCalls / (elapsedSec || 1)).toFixed(1);

  console.log('\n--- REPRODUCTION RESULTS ---');
  console.log(`Initial Tab 1 requests: ${countAfterTab1}`);
  console.log(`Total session requests observed: ${endObsCount}`);
  console.log(`Delta requests during 5s concurrent window: ${deltaCalls} calls (~${ratePerSec} req/sec)`);

  if (deltaCalls > 10) {
    console.log(`\n[CRITICAL CONFIRMED] Session request storm reproduced! ${deltaCalls} requests fired in ${elapsedSec}s!`);
  } else {
    console.log('\nNo storm detected during 5s window.');
  }

  await browser.close();
}

runReproduction().catch((err) => {
  console.error('Reproduction failed with error:', err);
  process.exit(1);
});
