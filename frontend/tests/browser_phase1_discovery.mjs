import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../..');
const EVIDENCE_DIR = path.resolve(REPO_ROOT, 'artifacts/evidence_20261011');
if (!fs.existsSync(EVIDENCE_DIR)) {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
}

async function discoverControls() {
  console.log('===============================================================');
  console.log('OPERATION JRUVA: PHASE 1 — CONTROL DISCOVERY & INVENTORY');
  console.log('Target: https://www.jruva.in');
  console.log('===============================================================');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    permissions: ['clipboard-read', 'clipboard-write'],
  });
  const page = await context.newPage();

  const consoleLogs = [];
  const networkRequests = [];
  const failedRequests = [];

  page.on('console', msg => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
  });

  page.on('response', res => {
    networkRequests.push({
      url: res.url(),
      status: res.status(),
      method: res.request().method(),
      contentType: res.headers()['content-type'] || '',
    });
  });

  page.on('requestfailed', req => {
    failedRequests.push({ url: req.url(), error: req.failure()?.errorText });
  });

  console.log('[1/6] Navigating to https://www.jruva.in...');
  const navResponse = await page.goto('https://www.jruva.in', {
    waitUntil: 'domcontentloaded',
    timeout: 30000,
  });
  console.log(`Navigation status: ${navResponse?.status()}`);

  await page.waitForTimeout(2000);

  // Dismiss splash if present
  const splash = page.locator('.jru-initial-splash-overlay');
  if (await splash.isVisible().catch(() => false)) {
    console.log('Dismissing splash overlay...');
    await splash.click();
    await page.waitForTimeout(500);
  }

  // 1. Capture Desktop Viewport
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '01_desktop_home_1920x1080.png'), fullPage: true });
  console.log('✓ Captured: 01_desktop_home_1920x1080.png');

  // 2. Discover all interactive elements
  console.log('[2/6] Extracting all actionable DOM controls...');
  const inventory = await page.evaluate(() => {
    const selector = 'button, a, input, textarea, select, [role="button"], [role="menuitem"], [role="tab"], [tabindex="0"]';
    const elements = Array.from(document.querySelectorAll(selector));

    return elements.map((el, index) => {
      const rect = el.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).visibility !== 'hidden' && window.getComputedStyle(el).display !== 'none';
      const text = el.innerText?.trim() || el.textContent?.trim() || '';
      
      return {
        index,
        tagName: el.tagName.toLowerCase(),
        id: el.id || null,
        className: el.className || null,
        ariaLabel: el.getAttribute('aria-label') || null,
        role: el.getAttribute('role') || null,
        type: el.getAttribute('type') || null,
        href: el.getAttribute('href') || null,
        placeholder: el.getAttribute('placeholder') || null,
        textSnippet: text.slice(0, 80),
        isVisible,
        rect: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
        },
        disabled: el.disabled || el.getAttribute('aria-disabled') === 'true',
      };
    });
  });

  console.log(`Total interactive elements scanned: ${inventory.length}`);
  const visibleControls = inventory.filter(c => c.isVisible);
  console.log(`Visible interactive controls: ${visibleControls.length}`);

  // 3. Inspect Model Selector Dropdown
  console.log('[3/6] Inspecting Model Selector Dropdown...');
  const modelBtn = page.locator('button:has-text("Auto"), button:has-text("Gemini"), button:has-text("Qwen"), [aria-label*="model" i]').first();
  let modelDropdownOpened = false;
  if (await modelBtn.isVisible().catch(() => false)) {
    console.log('Found model selector button. Clicking to expand...');
    await modelBtn.click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, '02_model_selector_dropdown.png') });
    console.log('✓ Captured: 02_model_selector_dropdown.png');
    modelDropdownOpened = true;
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
  } else {
    console.log('Model selector button not found with primary query.');
  }

  // 4. Inspect Sidebar Toggle & Navigation items
  console.log('[4/6] Inspecting Sidebar & Menu Controls...');
  const sidebarToggle = page.locator('button[aria-label*="sidebar" i], button[aria-label*="menu" i], button.jru-sidebar-toggle').first();
  if (await sidebarToggle.isVisible().catch(() => false)) {
    console.log('Found sidebar toggle button.');
    await sidebarToggle.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, '03_sidebar_toggled.png') });
    console.log('✓ Captured: 03_sidebar_toggled.png');
  }

  // 5. Tablet Viewport (768 x 1024)
  console.log('[5/6] Testing Tablet Viewport (768 x 1024)...');
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '04_tablet_viewport_768x1024.png'), fullPage: true });
  console.log('✓ Captured: 04_tablet_viewport_768x1024.png');

  // 6. Mobile Viewport (375 x 812)
  console.log('[6/6] Testing Mobile Viewport (375 x 812)...');
  await page.setViewportSize({ width: 375, height: 812 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '05_mobile_viewport_375x812.png'), fullPage: true });
  console.log('✓ Captured: 05_mobile_viewport_375x812.png');

  // Save Inventory JSON
  const outputData = {
    timestamp: new Date().toISOString(),
    targetUrl: 'https://www.jruva.in',
    totalElementsScanned: inventory.length,
    visibleElementsCount: visibleControls.length,
    modelDropdownOpened,
    controls: inventory,
    consoleLogs,
    networkRequestsSummary: {
      total: networkRequests.length,
      failed: failedRequests.length,
      statusCodes: networkRequests.reduce((acc, r) => {
        acc[r.status] = (acc[r.status] || 0) + 1;
        return acc;
      }, {}),
    },
    failedRequests,
  };

  const jsonPath = path.join(EVIDENCE_DIR, 'raw_control_inventory.json');
  fs.writeFileSync(jsonPath, JSON.stringify(outputData, null, 2), 'utf-8');
  console.log(`✓ Inventory written to: ${jsonPath}`);

  await browser.close();
  console.log('PHASE 1 DISCOVERY RUN COMPLETED SUCCESSFULLY.');
}

discoverControls().catch(err => {
  console.error('FATAL ERROR in discoverControls:', err);
  process.exit(1);
});
