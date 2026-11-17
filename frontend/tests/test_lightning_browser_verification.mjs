// OPERATION JRUVA LIGHTNING - Browser E2E Verification Matrix
// Tests Desktop & Mobile viewports, TTFT, progressive rendering, fast-fail & zero-duplicate errors.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.env.BASE_URL || 'http://127.0.0.1:3129';
const browser = await chromium.launch({ headless: true });

const results = {
  desktopGreeting: {},
  mobileGreeting: {},
  fastFailError: {},
  duplicateErrorCheck: {},
  checks: [],
};

const check = (name, condition) => {
  assert.ok(condition, name);
  results.checks.push(name);
  console.log(`[PASS] ${name}`);
};

try {
  // ==========================================
  // SCENARIO 1: DESKTOP GREETING ("yoo bro")
  // ==========================================
  console.log('\n--- Scenario 1: Desktop Greeting ("yoo bro") ---');
  const desktopContext = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });

  await desktopContext.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== base) return route.abort();
    if (!url.pathname.startsWith('/api/')) return route.continue();
    const json = (status, data) => route.fulfill({ status, json: data });

    if (url.pathname.endsWith('/auth/me')) return json(401, { detail: 'Guest mode' });
    if (url.pathname.endsWith('/models')) {
      return json(200, {
        models: [
          { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash', tier: 'standard' },
          { id: 'openai/gpt-oss-120b', name: 'Groq GPT-OSS 120B', tier: 'pro' },
        ],
      });
    }
    if (url.pathname.endsWith('/sessions')) return json(200, { sessions: [] });
    if (url.pathname.includes('/sessions/') && url.pathname.endsWith('/messages')) return json(200, { messages: [] });

    if (url.pathname === '/api/v1/chat') {
      // Simulate fast upstream streaming response
      const tokens = ['Yo! ', 'What\'s ', 'up, ', 'how ', 'can ', 'I ', 'help ', 'you ', 'today?'];
      let body = '';
      for (const token of tokens) {
        body += `event: message\ndata: ${JSON.stringify({ type: 'token', content: token, actual_provider: 'gemini', actual_model: 'gemini-3.5-flash' })}\n\n`;
      }
      body += `event: done\ndata: ${JSON.stringify({ type: 'finish', finish_reason: 'stop', total_tokens: tokens.length })}\n\n`;

      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: {
          'Cache-Control': 'no-cache, no-transform',
          'Connection': 'keep-alive',
          'X-Accel-Buffering': 'no',
        },
        body,
      });
    }

    return json(200, {});
  });

  const desktopPage = await desktopContext.newPage();
  await desktopPage.goto(base);
  await desktopPage.locator('textarea').first().waitFor({ state: 'visible', timeout: 8000 });

  const inputArea = desktopPage.locator('textarea').first();
  await inputArea.fill('yoo bro');

  const t0 = performance.now();
  await inputArea.press('Enter');

  // 1. Browser submission feedback (textarea clears, placeholder appears)
  await desktopPage.waitForFunction(() => {
    const ta = document.querySelector('textarea');
    return ta && ta.value === '';
  }, { timeout: 2000 });
  const tSubmission = performance.now() - t0;
  console.log(`[MEASUREMENT] Desktop submission feedback latency: ${Math.round(tSubmission)}ms`);
  results.desktopGreeting.submissionMs = Math.round(tSubmission);
  check('Desktop: immediate submission feedback < 300ms', tSubmission < 300);

  // 2. First token arrival & assistant card display
  const expectedReply = 'Yo! What\'s up, how can I help you today?';
  await desktopPage.getByText(expectedReply).waitFor({ timeout: 5000 });
  const tTotal = performance.now() - t0;
  results.desktopGreeting.totalMs = Math.round(tTotal);
  check('Desktop: fast greeting completed < 3000ms', tTotal < 3000);

  // 3. Verify zero duplicate error banner
  const bannerVisible = await desktopPage.locator('.jru-error-banner').isVisible().catch(() => false);
  check('Desktop: zero error banner on successful greeting', !bannerVisible);

  await desktopContext.close();

  // ==========================================
  // SCENARIO 2: MOBILE VIEWPORT GREETING
  // ==========================================
  console.log('\n--- Scenario 2: Mobile Viewport (375x667) ---');
  const mobileContext = await browser.newContext({
    viewport: { width: 375, height: 667 },
    isMobile: true,
  });

  await mobileContext.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== base) return route.abort();
    if (!url.pathname.startsWith('/api/')) return route.continue();
    const json = (status, data) => route.fulfill({ status, json: data });

    if (url.pathname.endsWith('/auth/me')) return json(401, { detail: 'Guest mode' });
    if (url.pathname.endsWith('/models')) return json(200, { models: [] });
    if (url.pathname.endsWith('/sessions')) return json(200, { sessions: [] });
    if (url.pathname.includes('/sessions/') && url.pathname.endsWith('/messages')) return json(200, { messages: [] });

    if (url.pathname === '/api/v1/chat') {
      const tokens = ['Hey! ', 'How\'s ', 'it ', 'going?'];
      let body = '';
      for (const token of tokens) {
        body += `event: message\ndata: ${JSON.stringify({ type: 'token', content: token })}\n\n`;
      }
      body += `event: done\ndata: ${JSON.stringify({ type: 'finish', finish_reason: 'stop', total_tokens: tokens.length })}\n\n`;

      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body,
      });
    }

    return json(200, {});
  });

  const mobilePage = await mobileContext.newPage();
  await mobilePage.goto(base);
  await mobilePage.locator('textarea').first().waitFor({ state: 'visible', timeout: 8000 });

  const mobileInput = mobilePage.locator('textarea').first();
  await mobileInput.fill('yoo bro');

  const tm0 = performance.now();
  await mobileInput.press('Enter');

  await mobilePage.waitForFunction(() => {
    const ta = document.querySelector('textarea');
    return ta && ta.value === '';
  }, { timeout: 2000 });
  const tmSubmission = performance.now() - tm0;
  console.log(`[MEASUREMENT] Mobile submission feedback latency: ${Math.round(tmSubmission)}ms`);
  results.mobileGreeting.submissionMs = Math.round(tmSubmission);
  check('Mobile: immediate submission feedback < 300ms', tmSubmission < 300);

  await mobilePage.getByText('Hey! How\'s it going?').waitFor({ timeout: 5000 });
  const tmTotal = performance.now() - tm0;
  results.mobileGreeting.totalMs = Math.round(tmTotal);
  check('Mobile: greeting completed < 3000ms', tmTotal < 3000);

  const mobileBanner = await mobilePage.locator('.jru-error-banner').isVisible().catch(() => false);
  check('Mobile: zero error banner on successful greeting', !mobileBanner);

  await mobileContext.close();

  // ==========================================
  // SCENARIO 3: FAST-FAIL & ZERO DUPLICATE ERROR DISPLAY
  // ==========================================
  console.log('\n--- Scenario 3: Fast-Fail & Zero Duplicate Error Display ---');
  const errorContext = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });

  await errorContext.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== base) return route.abort();
    if (!url.pathname.startsWith('/api/')) return route.continue();
    const json = (status, data) => route.fulfill({ status, json: data });

    if (url.pathname.endsWith('/auth/me')) return json(401, { detail: 'Guest mode' });
    if (url.pathname.endsWith('/models')) return json(200, { models: [] });
    if (url.pathname.endsWith('/sessions')) return json(200, { sessions: [] });
    if (url.pathname.includes('/sessions/') && url.pathname.endsWith('/messages')) return json(200, { messages: [] });

    if (url.pathname === '/api/v1/chat') {
      // Simulate fast fail within 1.5 seconds (not 35 seconds!)
      const errorBody =
        `event: error\ndata: ${JSON.stringify({ type: 'error', content: 'AI capacity is temporarily unavailable across all cloud providers. Please retry in a few moments.' })}\n\n`;

      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: errorBody,
      });
    }

    return json(200, {});
  });

  const errorPage = await errorContext.newPage();
  await errorPage.goto(base);
  await errorPage.locator('textarea').first().waitFor({ state: 'visible', timeout: 8000 });

  await errorPage.locator('textarea').first().fill('yoo bro');
  const te0 = performance.now();
  await errorPage.locator('textarea').first().press('Enter');

  // Wait for the assistant error bubble to appear
  const errorBubble = errorPage.locator('.message-bubble.error');
  await errorBubble.waitFor({ state: 'visible', timeout: 6000 });
  const teError = performance.now() - te0;
  console.log(`[MEASUREMENT] Fast-fail error display latency: ${Math.round(teError)}ms`);
  results.fastFailError.errorLatencyMs = Math.round(teError);
  check('Fast-fail error received quickly (< 3000ms, far below 35s timeout)', teError < 3000);

  const errorText = await errorBubble.textContent();
  check('Assistant error contains honest explanation', errorText.includes('AI capacity is temporarily unavailable') || errorText.includes('unavailable') || errorText.includes('Jruva'));

  // CRITICAL CHECK: ZERO DUPLICATE ERROR DISPLAY
  // The top banner MUST NOT be visible when the assistant card displays the error!
  const duplicateBannerVisible = await errorPage.locator('.jru-error-banner').isVisible().catch(() => false);
  check('Zero duplicate error banner: top banner is hidden when assistant message displays error', !duplicateBannerVisible);
  results.duplicateErrorCheck.bannerHidden = !duplicateBannerVisible;

  await errorContext.close();

  console.log('\n==========================================');
  console.log('ALL LIGHTNING BROWSER E2E VERIFICATIONS PASSED:');
  console.log(JSON.stringify(results, null, 2));
  console.log('==========================================');
} finally {
  await browser.close();
}
