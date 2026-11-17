// Actual loopback API + production UI, synthetic account/documents, local inference.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { chromium } from 'playwright';

const base = process.env.BASE_URL || 'http://127.0.0.1:33619';
const out = '../docs/full-spectrum-audit-20261010';
const auth = JSON.parse(fs.readFileSync('../scratch/browser/auth.json', 'utf8'));
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
await context.addCookies([{ name: 'access_token', value: auth.Authorization.replace('Bearer ', ''), url: base, httpOnly: true, sameSite: 'Lax' }]);
await context.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
const page = await context.newPage();
const checks = [], runtimeErrors = [];
page.on('pageerror', error => runtimeErrors.push(error.message));
async function check(name, fn) {
  try { await fn(); checks.push({ name, passed: true }); }
  catch (error) { checks.push({ name, passed: false, error: error.message }); }
  console.log(JSON.stringify(checks.at(-1)));
}
async function json(response, status = 200) {
  assert.equal(response.status(), status, await response.text());
  return response.json();
}
try {
  await page.goto(base);
  await page.getByRole('textbox', { name: 'Message input', exact: true }).waitFor();
  await check('real authenticated cookie survives frontend proxy', async () => {
    assert.equal((await json(await page.request.get(base + '/api/v1/auth/me'))).id, 'full-spectrum-browser');
  });
  await check('real model catalog reports unavailable cloud providers and local specialist', async () => {
    const data = await json(await page.request.get(base + '/api/v1/models'));
    assert.equal(data.models.find(m => m.provider === 'gemini').available, false);
    assert.equal(data.models.find(m => m.id === 'ollama/jru-specialist:v4').available, true);
  });
  await check('real project UI creates, edits, selects and deletes owned data', async () => {
    await page.getByTitle('Project Workspaces', { exact: true }).click();
    await page.getByRole('button', { name: /New Project/ }).click();
    await page.locator('#project-name-input').fill('Full Spectrum Synthetic Project');
    await page.locator('#project-instructions-input').fill('Keep the synthetic audit answer concise.');
    await page.locator('#project-model-input').selectOption('ollama/jru-specialist:v4');
    await page.getByRole('button', { name: 'Create Project', exact: true }).click();
    const card = page.locator('.project-card').filter({ hasText: 'Full Spectrum Synthetic Project' });
    await card.waitFor();
    await card.getByTitle('Edit instructions', { exact: true }).click();
    await page.locator('#project-desc-input').fill('Synthetic audit edit');
    await page.getByRole('button', { name: 'Save Changes', exact: true }).click();
    await card.getByText('Synthetic audit edit', { exact: true }).waitFor();
    await card.getByRole('button', { name: /Start Chat in Project/ }).click();
    assert.ok((await page.getByTitle('Select AI Model', { exact: true }).innerText()).includes('Specialist'));
    await page.getByTitle('Project Workspaces', { exact: true }).click();
    await page.locator('.project-card').filter({ hasText: 'Full Spectrum Synthetic Project' }).getByTitle('Delete project', { exact: true }).click();
    await page.getByText('Full Spectrum Synthetic Project', { exact: true }).waitFor({ state: 'hidden' });
  });
  await check('real document UI upload, indexed status, retrieval and deletion', async () => {
    await page.getByTitle('Knowledge Library & Documents', { exact: true }).click();
    await page.locator('input[type=file]').setInputFiles({ name: 'full_spectrum_synthetic.txt', mimeType: 'text/plain', buffer: Buffer.from('The synthetic audit telescope calibration interval is 43 days. Locker code is STARDUST-8621.') });
    const row = page.locator('tr').filter({ hasText: 'full_spectrum_synthetic.txt' });
    await row.getByText('✓ INDEXED', { exact: true }).waitFor({ timeout: 60000 });
    const search = await json(await page.request.get(base + '/api/v1/library/search?q=synthetic%20audit%20telescope%20calibration'));
    assert.ok(search.results.some(r => r.text.includes('43 days')));
    const files = await json(await page.request.get(base + '/api/v1/library/files'));
    const file = files.files.find(f => f.filename === 'full_spectrum_synthetic.txt');
    assert.ok(file.chunk_count > 0);
    await json(await page.request.post(base + `/api/v1/library/files/${file.id}/reindex`));
    await row.getByTitle('Delete document', { exact: true }).click();
    await row.waitFor({ state: 'hidden' });
    const after = await json(await page.request.get(base + '/api/v1/library/search?q=STARDUST-8621'));
    assert.ok(!after.results.some(r => r.file_id === file.id));
  });
  await check('real scheduled task creation, pause, history and delete', async () => {
    await page.getByTitle('Scheduled AI Automations', { exact: true }).click();
    await page.locator('#task-name-input').fill('Full Spectrum Synthetic Task');
    await page.locator('#task-prompt-input').fill('Reply audit-ready.');
    await page.locator('#schedule-preset-select').selectOption('0 8 * * *');
    await page.getByRole('button', { name: /Schedule Task/ }).click();
    const card = page.locator('.schedule-task-card').filter({ hasText: 'Full Spectrum Synthetic Task' });
    await card.waitFor();
    await card.getByTitle('Toggle active / paused', { exact: true }).click();
    await card.getByText('⏸️ Paused', { exact: true }).waitFor();
    await card.getByTitle('View execution logs', { exact: true }).click();
    await page.getByText('No executions logged for this task yet.', { exact: true }).waitFor();
    await page.locator('.workspace-modal-overlay').click({ position: { x: 5, y: 5 } });
    await card.getByTitle('Delete task', { exact: true }).click();
    await card.waitFor({ state: 'hidden' });
  });
  await check('real plugin search preference persists through refresh', async () => {
    await page.getByTitle('Plugins & Tools Hub', { exact: true }).click();
    const card = page.locator('.plugin-card').filter({ hasText: 'Live Web Search' });
    if (await card.getByRole('checkbox').isChecked()) {
      const changed = page.waitForResponse(r => r.url().endsWith('/plugins/web_search/toggle'));
      await card.getByTitle('Toggle Live Web Search', { exact: true }).click();
      assert.equal((await changed).status(), 200);
    }
    const response = await json(await page.request.get(base + '/api/v1/plugins'));
    assert.equal(response.plugins.find(p => p.id === 'web_search').is_enabled, false);
    await page.reload();
    await page.getByTitle('Plugins & Tools Hub', { exact: true }).click();
    await card.waitFor();
    assert.equal(await card.getByRole('checkbox').isChecked(), false);
  });
  await check('real image generation fails closed while local flag is disabled', async () => {
    await page.getByTitle('Images Studio & Gallery', { exact: true }).click();
    await page.locator('#image-prompt-input').fill('A synthetic audit telescope');
    const response = page.waitForResponse(r => r.url().endsWith('/api/v1/image/generate'));
    await page.getByRole('button', { name: /Generate Image/ }).click();
    assert.equal((await response).status(), 503);
    await page.locator('.workspace-status-banner').filter({ hasText: /disabled|unavailable|enabled/i }).waitFor();
  });
  await check('real nonstream chat preserves explicit local model identity', async () => {
    const response = await page.request.post(base + '/api/v1/chat', { timeout: 150000, data: {
      message: 'What is 17 plus 25? Reply with just the number.', model: 'ollama/jru-specialist:v4', stream: false, web_search: false, temperature: 0,
    } });
    const data = await json(response);
    assert.equal(data.actual_provider, 'ollama');
    assert.ok(data.actual_model.endsWith('jru-specialist:v4'));
    fs.writeFileSync(out + '/live-chat-response.json', JSON.stringify(data, null, 2));
    assert.ok(data.content.includes('42'), 'Arithmetic smoke failure; response retained in live-chat-response.json');
  });
  await check('real global search sees only owned synthetic data', async () => {
    const data = await json(await page.request.get(base + '/api/v1/search/global?q=Full%20Spectrum'));
    assert.ok(data);
    await page.keyboard.press('Control+k');
    await page.getByRole('dialog', { name: 'Global Workspace Command Palette', exact: true }).waitFor();
    await page.waitForFunction(() => document.activeElement?.classList.contains('cmd-palette-input'));
    await page.keyboard.press('Escape');
    await page.getByRole('dialog', { name: 'Global Workspace Command Palette', exact: true }).waitFor({ state: 'hidden' });
  });
  await check('real Memory Card adds and deletes an owned fact', async () => {
    await page.getByTitle('Personal AI Memory Card', { exact: true }).click();
    await page.getByPlaceholder('Teach Jruva something about you (e.g. 3rd year BCA student, targeting GATE)...', { exact: true }).fill('Synthetic audit preference: use short answers.');
    await page.getByRole('button', { name: '+ Remember', exact: true }).click();
    const card = page.locator('.memory-fact-card').filter({ hasText: 'Synthetic audit preference: use short answers.' });
    await card.waitFor();
    await card.getByRole('button', { name: 'Delete memory', exact: true }).click();
    await card.waitFor({ state: 'hidden' });
    await page.keyboard.press('Escape');
  });
  await check('real logout detaches private workspace and cookie', async () => {
    await page.getByRole('button', { name: 'User profile menu', exact: true }).click();
    await page.getByRole('menuitem', { name: /Log Out/ }).click();
    await page.getByRole('button', { name: 'User profile menu', exact: true }).waitFor({ state: 'hidden' });
    assert.equal((await page.request.get(base + '/api/v1/auth/me')).status(), 401);
    assert.equal(await page.evaluate(() => localStorage.getItem('jru_auth_user')), null);
  });
  await check('no unhandled live browser errors', async () => assert.deepEqual(runtimeErrors, []));
  await page.screenshot({ path: out + '/live-workflows.png', fullPage: true });
} finally {
  fs.writeFileSync(out + '/browser-live.json', JSON.stringify({ execution: 'real local API/UI/persistence/embeddings/Ollama; synthetic account and documents', checks, runtimeErrors }, null, 2));
  await browser.close();
}
if (checks.some(c => !c.passed)) process.exitCode = 1;
