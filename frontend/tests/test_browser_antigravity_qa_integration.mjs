// Test integration of Antigravity browser QA fixes: modal escape, error URL handling, responsive drawer.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = 'http://127.0.0.1:3129';
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
const checks = [];
const check = (name, condition) => { assert.ok(condition, name); checks.push(name); };

await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.origin !== base) return route.abort();
  if (!url.pathname.startsWith('/api/')) return route.continue();
  const json = (status, data) => route.fulfill({ status, json: data });

  if (url.pathname === '/api/v1/image/generate') {
    // Return the legacy unavailable error URL format that previously broke the modal
    return json(200, {
      success: false,
      error_type: 'unavailable',
      image_url: 'error:unavailable:Sorry%2C%20this%20image%20is%20unavailable.%20We%20only%20provide%20verified%20images%20from%20our%20university%20repository...',
      model: 'verified repository catalog',
      prompt: 'photorealistic red Ferrari'
    });
  }
  if (url.pathname === '/api/v1/image/gallery') {
    return json(200, {
      images: [
        { id: '1', prompt: 'Valid sample', file_url: '/static/demo/robotics.jpg', model: 'flux', created_at: new Date().toISOString() },
        { id: '2', prompt: 'Corrupted catalog error', file_url: 'error:unavailable:broken', model: 'none', created_at: new Date().toISOString() }
      ]
    });
  }
  if (url.pathname.endsWith('/auth/me')) return json(401, { detail: 'Guest mode' });
  return json(200, { sessions: [], files: [], models: [] });
});

try {
  const page = await context.newPage();
  await page.goto(base);

  // 1. Check Images Studio Gallery Filtering
  await page.getByTitle('Images Studio & Gallery', { exact: true }).click();
  await page.waitForTimeout(1000);
  
  await page.locator('.gallery-thumb-img').first().waitFor({ state: 'visible', timeout: 5000 });
  const galleryImgs = await page.locator('.gallery-thumb-img').all();
  check('gallery filters out corrupted error: URLs', galleryImgs.length === 1);

  // 2. Click on the valid sample image to open high-res modal
  await page.locator('.gallery-item-card').first().click();
  await page.waitForTimeout(500);
  const modalVisible = await page.locator('.image-preview-modal').isVisible();
  check('preview modal opens on gallery item click', modalVisible);

  // 3. Test Escape key dismissal
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  const modalDismissed = !(await page.locator('.image-preview-modal').isVisible());
  check('escape key successfully dismisses preview modal', modalDismissed);

  // 4. Test error URL generation handling
  await page.locator('#image-prompt-input').fill('Generate a photorealistic red Ferrari');
  await page.getByRole('button', { name: /Generate Image/ }).click();
  await page.waitForTimeout(1000);

  // Verify that preview modal did NOT open with broken error: url
  const modalStillHidden = !(await page.locator('.image-preview-modal').isVisible());
  check('broken error: url prevents high-res preview modal popup', modalStillHidden);

  // Verify that status alert banner caught and decoded the error message
  const statusText = await page.locator('.workspace-status-banner').innerText();
  check('status banner displays decoded error message', statusText.includes('Sorry, this image is unavailable'));

  console.log(JSON.stringify({ passed: checks.length, failed: 0, checks }));
} finally {
  await browser.close();
}
