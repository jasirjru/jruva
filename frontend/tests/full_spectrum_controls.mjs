// Real production UI with synthetic API responses. No model-quality claims.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const base = process.env.BASE_URL || 'http://127.0.0.1:33619';
const out = path.resolve(__dirname, '../../docs/full-spectrum-audit-20261010');
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const chat = [], images = [], checks = [], errors = [];
const models = [
  { id: 'auto', name: 'Auto', provider: 'auto' },
  { id: 'gemini-3.5-flash', name: 'Gemini Flash', provider: 'gemini' },
  { id: 'qwen/qwen3.8-27b', name: 'Qwen 3.8', provider: 'groq' },
  { id: 'ollama/jru-specialist:v4', name: 'JRU Specialist v4', provider: 'ollama' },
];
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.origin !== base) return route.abort();
  if (!url.pathname.startsWith('/api/')) return route.continue();
  const respond = data => route.fulfill({ status: 200, json: data });
  if (url.pathname.endsWith('/models')) return respond({ models: models.map(m => ({ ...m, available: true, health: 'healthy' })) });
  if (url.pathname.endsWith('/auth/me')) return route.fulfill({ status: 401, json: { detail: 'Synthetic guest' } });
  if (url.pathname.endsWith('/projects')) return respond({ projects: [{ id: 'project-fixture', name: 'Audit Project', instructions: 'Use a concise answer.', preferred_model: 'qwen/qwen3.8-27b' }] });
  if (url.pathname.endsWith('/image/generate')) {
    images.push(route.request().postDataJSON());
    return respond({ success: true, image_id: 'synthetic', model: 'synthetic', width: 1024, height: 1024 });
  }
  if (url.pathname.endsWith('/chat')) {
    chat.push(route.request().postDataJSON());
    return route.fulfill({ status: 200, contentType: 'text/event-stream', body:
      `event: message\ndata: ${JSON.stringify({ type: 'token', content: 'Synthetic response ' + chat.length })}\n\n` +
      `event: done\ndata: ${JSON.stringify({ type: 'finish', finish_reason: 'stop' })}\n\n` });
  }
  return respond({ sessions: [], files: [], images: [], plugins: [], tasks: [] });
});
const page = await context.newPage();
page.on('pageerror', error => errors.push(error.message));
async function check(name, fn) {
  try { await fn(); checks.push({ name, passed: true }); }
  catch (error) { checks.push({ name, passed: false, error: error.message }); }
  console.log(JSON.stringify(checks.at(-1)));
}
async function send(message) {
  await page.getByRole('textbox', { name: 'Message input', exact: true }).fill(message);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByText('Synthetic response ' + chat.length, { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Stop generation', exact: true }).waitFor({ state: 'hidden' });
  return chat.at(-1);
}
try {
  await page.goto(base);
  await page.getByRole('textbox', { name: 'Message input', exact: true }).waitFor();
  for (const model of models) await check('model control transmits ' + model.id, async () => {
    await page.getByTitle('Select AI Model', { exact: true }).click();
    await page.getByRole('option').filter({ hasText: model.name }).first().click();
    assert.equal((await send('Model control probe ' + model.name)).model, model.id);
  });
  for (const [name, directive] of [['Architect', 'SYSTEMS ARCHITECT'], ['Elite Coder', 'COMPETITIVE CODER'], ['Scholar', 'RESEARCH SCIENTIST'], ['General', undefined]]) {
    await check('persona control transmits ' + name, async () => {
      await page.getByRole('button', { name: 'Select AI Persona', exact: true }).click();
      await page.getByRole('option').filter({ hasText: name }).click();
      const request = await send('Persona control probe ' + name);
      if (directive) assert.ok(request.system_instruction.includes(directive));
      else assert.equal(request.system_instruction, undefined);
    });
  }
  for (const tier of ['B', 'A', 'S']) await check('compute tier ' + tier, async () => {
    await page.getByRole('radio', { name: new RegExp('^' + tier + '( |$)') }).click();
    assert.equal((await send('Compute probe ' + tier)).tier, tier);
  });
  await check('project model preference controls next chat request', async () => {
    await page.getByTitle('Project Workspaces', { exact: true }).click();
    await page.getByRole('button', { name: /Start Chat in Project/ }).click();
    const request = await send('Project preference probe');
    assert.equal(request.model, 'qwen/qwen3.8-27b');
    assert.ok(request.message.includes('Use a concise answer.'));
  });
  await page.getByTitle('Images Studio & Gallery', { exact: true }).click();
  for (const style of ['photorealistic', 'cinematic', 'cyberpunk', 'anime', 'architecture']) await check('Vision style ' + style, async () => {
    await page.locator('#image-style-select').selectOption(style);
    await page.locator('#image-prompt-input').fill('Synthetic style probe ' + style);
    await page.getByRole('button', { name: /Generate Image/ }).click();
    await page.waitForFunction(() => document.querySelector('#image-prompt-input').value === '');
    assert.equal(images.at(-1).style, style);
  });
  for (const model of ['flux', 'schnell']) await check('Vision engine ' + model, async () => {
    await page.locator('#image-model-select').selectOption(model);
    await page.locator('#image-prompt-input').fill('Synthetic engine probe ' + model);
    await page.getByRole('button', { name: /Generate Image/ }).click();
    await page.waitForFunction(() => document.querySelector('#image-prompt-input').value === '');
    assert.equal(images.at(-1).model, model);
  });
  await check('Vision unsupported aspect ratios remain disabled', async () => {
    assert.equal(await page.getByRole('button', { name: '2:3', exact: true }).isDisabled(), true);
    assert.equal(await page.getByRole('button', { name: '3:2', exact: true }).isDisabled(), true);
    assert.ok(images.every(i => i.width === 1024 && i.height === 1024));
  });
  await check('Vision prompt enhancement toggle reaches payload', async () => {
    await page.getByRole('checkbox').uncheck();
    await page.locator('#image-prompt-input').fill('Synthetic enhancement probe');
    await page.getByRole('button', { name: /Generate Image/ }).click();
    await page.waitForFunction(() => document.querySelector('#image-prompt-input').value === '');
    assert.equal(images.at(-1).enhance, false);
  });
  for (const width of [375, 768, 1440]) await check('Vision overflow at ' + width, async () => {
    await page.setViewportSize({ width, height: 1000 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  });
  await page.screenshot({ path: out + '/vision-controls.png', fullPage: true });
  await check('no unhandled browser errors', async () => assert.deepEqual(errors, []));
} finally {
  fs.writeFileSync(out + '/browser-controls.json', JSON.stringify({ execution: 'real UI; mocked APIs', checks, chat, images, errors }, null, 2));
  await browser.close();
}
if (checks.some(c => !c.passed)) process.exitCode = 1;
