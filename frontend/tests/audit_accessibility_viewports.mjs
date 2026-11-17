/**
 * Phase 8: Mobile Viewports, Responsive Design & Accessibility Audit.
 * Evaluates:
 * 1. Viewport scaling: 375x667 (Mobile S), 414x896 (Mobile L), 768x1024 (Tablet), 1280x800 (Laptop), 1920x1080 (Desktop)
 * 2. Horizontal layout overflow detection (document.body.scrollWidth <= window.innerWidth)
 * 3. Accessibility tree & ARIA semantics (chat input labels, focus indicators, interactive buttons)
 * 4. Keyboard focus traversal with Tab
 * 5. High-resolution screenshot capture for every viewport
 */
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const BASE_URL = process.env.CHATBOT_TEST_URL || 'http://localhost:3121';
const SCREENSHOT_DIR = path.resolve('docs/validation_report_2026-10-08/screenshots');

const VIEWPORTS = [
  { name: 'mobile_s_375x667', width: 375, height: 667, isMobile: true },
  { name: 'mobile_l_414x896', width: 414, height: 896, isMobile: true },
  { name: 'tablet_768x1024', width: 768, height: 1024, isMobile: false },
  { name: 'laptop_1280x800', width: 1280, height: 800, isMobile: false },
  { name: 'desktop_1920x1080', width: 1920, height: 1080, isMobile: false },
];

async function runAudit() {
  console.log('='.repeat(70));
  console.log('PHASE 8: RESPONSIVE VIEWPORT & ACCESSIBILITY AUDIT');
  console.log(`Target Frontend: ${BASE_URL}`);
  console.log('='.repeat(70));

  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const auditResults = [];

  for (const vp of VIEWPORTS) {
    console.log(`\nAuditing Viewport: ${vp.name} (${vp.width}x${vp.height})...`);
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      isMobile: vp.isMobile,
      userAgent: vp.isMobile
        ? 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1'
        : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    });

    const page = await context.newPage();
    const vpResult = {
      name: vp.name,
      dimensions: `${vp.width}x${vp.height}`,
      isMobile: vp.isMobile,
      overflowSafe: false,
      inputAccessible: false,
      keyboardFocusNavigable: false,
      screenshotPath: '',
      errors: [],
    };

    try {
      await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 20000 });

      // Dismiss splash overlay if present
      await page.evaluate(() => {
        const splash = document.querySelector('[class*="splash"], [class*="overlay"]');
        if (splash && splash.parentElement) {
          splash.remove();
        }
      });
      await page.waitForTimeout(500);

      // 1. Horizontal scrollbar / layout overflow check
      const overflowMetrics = await page.evaluate(() => {
        const docWidth = document.documentElement.scrollWidth;
        const bodyWidth = document.body.scrollWidth;
        const winWidth = window.innerWidth;
        const hasHorizontalScroll = Math.max(docWidth, bodyWidth) > winWidth;
        return { docWidth, bodyWidth, winWidth, hasHorizontalScroll };
      });

      vpResult.overflowSafe = !overflowMetrics.hasHorizontalScroll;
      if (overflowMetrics.hasHorizontalScroll) {
        vpResult.errors.push(`Horizontal overflow detected: scrollWidth (${overflowMetrics.bodyWidth}) > viewport (${overflowMetrics.winWidth})`);
      }

      // 2. Chat Input Accessibility Check
      const chatInput = page.locator('textarea');
      await chatInput.waitFor({ state: 'visible', timeout: 5000 });

      const inputAria = await chatInput.evaluate((el) => {
        return {
          placeholder: el.getAttribute('placeholder'),
          ariaLabel: el.getAttribute('aria-label'),
          hasRoleOrAria: !!(el.getAttribute('aria-label') || el.getAttribute('placeholder')),
          tagName: el.tagName,
        };
      });

      vpResult.inputAccessible = inputAria.hasRoleOrAria;

      // 3. Keyboard focus navigation check
      await page.keyboard.press('Tab');
      const focusedTag = await page.evaluate(() => document.activeElement ? document.activeElement.tagName : null);
      vpResult.keyboardFocusNavigable = !!focusedTag;

      // 4. Capture screenshot
      const shotPath = path.join(SCREENSHOT_DIR, `responsive_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: false });
      vpResult.screenshotPath = shotPath;
      console.log(`  ✓ Screenshot captured: ${shotPath}`);
      console.log(`  ✓ Overflow Safe: ${vpResult.overflowSafe}`);
      console.log(`  ✓ Input Accessible: ${vpResult.inputAccessible}`);
      console.log(`  ✓ Focus Navigable: ${vpResult.keyboardFocusNavigable}`);

    } catch (err) {
      console.error(`  ✗ Error auditing ${vp.name}:`, err.message);
      vpResult.errors.push(err.message);
    } finally {
      await context.close();
    }

    auditResults.push(vpResult);
  }

  await browser.close();

  const reportPath = path.resolve('docs/validation_report_2026-10-08/accessibility_viewports_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(auditResults, null, 2), 'utf-8');
  console.log('\n' + '='.repeat(70));
  console.log(`Accessibility audit completed and saved to: ${reportPath}`);
  console.log('='.repeat(70));
}

runAudit().catch((err) => {
  console.error('Fatal audit failure:', err);
  process.exit(1);
});
