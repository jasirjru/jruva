// Integrated local UI; all API calls are synthetic, external traffic blocked.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { chromium } from 'playwright';

const base = process.env.BASE_URL || 'http://127.0.0.1:3129';
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
const requests = [];
const chatRequests = [];
let eofCount = 0;
let reloadCount = 0;
const checks = [];
const check = (name, condition) => { assert.ok(condition, name); checks.push(name); };
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.origin !== base) return route.abort();
  if (!url.pathname.startsWith('/api/')) return route.continue();
  const json = (status, data) => route.fulfill({ status, json: data });
  if (url.pathname === '/api/v1/image/generate') {
    requests.push({ key: route.request().headers()['idempotency-key'], body: route.request().postDataJSON() });
    return requests.length === 1
      ? json(503, { error_type: 'operation_indeterminate', error_message: 'Outcome unknown. Check this request without generating again.' })
      : json(200, { success: true, image_id: 'synthetic', model: 'synthetic', width: 1024, height: 1024 });
  }
  if (url.pathname === '/api/v1/chat') {
    const request = route.request().postDataJSON();
    chatRequests.push(request);
    if (request.message.includes('EOF simulation') && ++eofCount === 1) {
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body:
        `event: message\ndata: ${JSON.stringify({ type: 'token', content: 'Synthetic incomplete' })}\n\n` });
    }
    const unknown = chatRequests.length === 1 || (request.message.includes('reload simulation') && ++reloadCount === 1);
    return route.fulfill({ status: 200, contentType: 'text/event-stream', body:
      `event: message\ndata: ${JSON.stringify({ type: 'token', content: unknown ? 'Synthetic outcome unknown' : 'Synthetic resolved' })}\n\n` +
      `event: done\ndata: ${JSON.stringify({ type: 'finish', finish_reason: unknown ? 'error' : 'stop' })}\n\n` });
  }
  if (url.pathname.includes('/sessions/') && url.pathname.endsWith('/messages') && reloadCount) {
    return json(200, { messages: [
      { role: 'user', content: 'Generate reload simulation image' },
      { role: 'assistant', content: 'Synthetic outcome unknown' },
    ] });
  }
  if (url.pathname.endsWith('/auth/me')) return json(401, { detail: 'Synthetic guest' });
  return json(200, { images: [], sessions: [], files: [], models: [] });
});
try {
  const page = await context.newPage();
  await page.exposeFunction('_assemblyRecordAbort', request => chatRequests.push(request));
  await page.addInitScript(() => {
    const originalFetch = window.fetch;
    let interrupted = false;
    window.fetch = async (url, options) => {
      if (String(url).endsWith('/api/v1/chat') && options?.body) {
        const request = JSON.parse(options.body);
        if (request.message.includes('cancel simulation') && !interrupted) {
          interrupted = true;
          await window._assemblyRecordAbort(request);
          return new Response(new ReadableStream({ start(controller) {
            controller.enqueue(new TextEncoder().encode('event: message\ndata: {"type":"token","content":"Synthetic interrupted"}\n\n'));
            options.signal.addEventListener('abort', () => controller.error(new DOMException('Synthetic abort', 'AbortError')));
          } }), { headers: { 'Content-Type': 'text/event-stream' } });
        }
      }
      return originalFetch(url, options);
    };
  });
  await page.goto(base);
  await page.getByTitle('Images Studio & Gallery', { exact: true }).click();
  await page.locator('#image-prompt-input').fill('A synthetic blue house');
  await page.getByRole('button', { name: /Generate Image/ }).click();
  await page.waitForFunction(() => document.body.innerText.includes('Outcome unknown'));
  check('new image intent has explicit identity', !!requests[0].key);
  await page.reload();
  await page.getByTitle('Images Studio & Gallery', { exact: true }).click();
  await page.locator('#image-prompt-input').fill('A synthetic blue house');
  await page.getByRole('button', { name: /Retry|Generate Image/ }).click();
  await page.waitForFunction(() => document.querySelector('#image-prompt-input')?.value === '');
  check('retry retains identical key', requests[0].key === requests[1].key);
  check('retry retains immutable payload', JSON.stringify(requests[0].body) === JSON.stringify(requests[1].body));
  await page.locator('#image-prompt-input').fill('A synthetic blue house');
  await page.getByRole('button', { name: /Generate Image/ }).click();
  await page.waitForFunction(() => document.querySelector('#image-prompt-input')?.value === '');
  check('new identical intent gets fresh identity', requests[2].key !== requests[1].key);
  for (const width of [375, 768, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    check(`studio no horizontal overflow at ${width}`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  }
  await page.getByRole('button', { name: /Back to Chat/ }).click();
  await page.locator('textarea').first().fill('Generate a synthetic image');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByText('Synthetic outcome unknown', { exact: true }).waitFor();
  check('unknown chat outcome does not automatically repeat', chatRequests.length === 1);
  await page.getByRole('button', { name: 'Regenerate response', exact: true }).click();
  await page.getByText('Synthetic resolved', { exact: true }).waitFor();
  check('failed chat retry retains turn identity', chatRequests[0].conversation_id === chatRequests[1].conversation_id);
  await page.getByRole('button', { name: 'Regenerate response', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.btn-spinner'));
  await page.getByText('Synthetic resolved', { exact: true }).waitFor();
  check('completed chat regeneration creates fresh turn identity', chatRequests[2].conversation_id !== chatRequests[1].conversation_id);
  await page.locator('textarea').first().fill('Generate EOF simulation image');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByText('Synthetic incomplete', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Stop generation', exact: true }).waitFor({ state: 'hidden' });
  const eofRetryResponse = page.waitForResponse(response => response.url().endsWith('/api/v1/chat'));
  await page.getByRole('button', { name: 'Regenerate response', exact: true }).last().click();
  await eofRetryResponse;
  await page.getByText('Synthetic resolved', { exact: true }).last().waitFor();
  check('premature EOF retry retains turn identity', chatRequests[3].conversation_id === chatRequests[4].conversation_id);
  await page.locator('textarea').first().fill('Generate cancel simulation image');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByText('Synthetic interrupted', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Stop generation', exact: true }).click();
  await page.getByRole('button', { name: 'Stop generation', exact: true }).waitFor({ state: 'hidden' });
  const cancelledRetryResponse = page.waitForResponse(response => response.url().endsWith('/api/v1/chat'));
  await page.getByRole('button', { name: 'Regenerate response', exact: true }).last().click();
  await cancelledRetryResponse;
  await page.getByText('Synthetic resolved', { exact: true }).last().waitFor();
  check('aborted dispatch retry retains turn identity', chatRequests[5].conversation_id === chatRequests[6].conversation_id);
  await page.locator('textarea').first().fill('Generate reload simulation image');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByText('Synthetic outcome unknown', { exact: true }).last().waitFor();
  await page.getByRole('button', { name: 'Stop generation', exact: true }).waitFor({ state: 'hidden' });
  await page.reload();
  await page.getByText('Synthetic outcome unknown', { exact: true }).waitFor();
  const restoredRetryResponse = page.waitForResponse(response => response.url().endsWith('/api/v1/chat'));
  await page.getByRole('button', { name: 'Regenerate response', exact: true }).last().click();
  await restoredRetryResponse;
  check('restored unknown chat retry retains turn identity', chatRequests[7].image_intent_id === chatRequests[8].image_intent_id);
  console.log(JSON.stringify({ passed: checks.length, failed: 0, checks }));
  fs.mkdirSync('../scratch/assembly-browser', { recursive: true });
  fs.writeFileSync('../scratch/assembly-browser/results.json', JSON.stringify({ passed: checks.length, failed: 0, checks }, null, 2));
} finally { await browser.close(); }

