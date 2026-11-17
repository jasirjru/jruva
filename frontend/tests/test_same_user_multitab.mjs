import { chromium } from 'playwright';

async function testMultiTab() {
  console.log('Testing multi-tab behavior...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  
  const tab1 = await context.newPage();
  console.log('Navigating Tab 1...');
  await tab1.goto('http://localhost:3121', { waitUntil: 'networkidle' });
  console.log('Tab 1 title:', await tab1.title());

  const splash1 = tab1.locator('.jru-initial-splash-overlay');
  if (await splash1.isVisible()) {
    await splash1.click();
  }
  const input1 = tab1.locator('textarea[aria-label="Message input"], textarea.jru-textarea').first();
  await input1.waitFor({ state: 'visible' });
  console.log('Tab 1 input found!');

  const tab2 = await context.newPage();
  console.log('Navigating Tab 2...');
  await tab2.goto('http://localhost:3121', { waitUntil: 'networkidle' });
  console.log('Tab 2 title:', await tab2.title());

  const splash2 = tab2.locator('.jru-initial-splash-overlay');
  if (await splash2.isVisible()) {
    await splash2.click();
  }
  const input2 = tab2.locator('textarea[aria-label="Message input"], textarea.jru-textarea').first();
  await input2.waitFor({ state: 'visible' });
  console.log('Tab 2 input found!');

  await browser.close();
  console.log('Multi-tab verification SUCCESS!');
}

testMultiTab().catch(err => {
  console.error('Multi-tab verification error:', err);
  process.exit(1);
});
