import { chromium } from 'playwright';

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('http://localhost:3121');
  const title = await page.title();
  console.log('Successfully navigated to http://localhost:3121. Title:', title);
  await browser.close();
}

main().catch(err => {
  console.error('Playwright verification failed:', err);
  process.exit(1);
});
