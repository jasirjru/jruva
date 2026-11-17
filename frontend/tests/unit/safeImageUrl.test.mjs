// Unit tests for the model-output image URL allowlist.
// Run: npm run test:unit   (Node >= 22.18 strips the TypeScript types itself)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isSafeImageUrl, isSameOriginPath } from '../../src/utils/safeImageUrl.ts';

const API = 'https://api.example-app.test';
const ORIGIN = 'https://app.example-app.test';

test('allows same-origin paths but not protocol-relative ones', () => {
  assert.equal(isSafeImageUrl('/api/v1/image/file/a.png', API, ORIGIN), true);
  assert.equal(isSafeImageUrl('//evil.com/a.png', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('/\\evil.com/a.png', API, ORIGIN), false);
  assert.equal(isSameOriginPath('/x'), true);
  assert.equal(isSameOriginPath('//x'), false);
  assert.equal(isSameOriginPath('/\\x'), false);
});

test('allows blob and base64 raster data URLs only', () => {
  assert.equal(isSafeImageUrl('blob:https://app.example-app.test/123', API, ORIGIN), true);
  assert.equal(isSafeImageUrl('data:image/png;base64,iVBORw0KGgo=', API, ORIGIN), true);
  assert.equal(isSafeImageUrl('data:image/webp;base64,UklGRg==', API, ORIGIN), true);
  assert.equal(isSafeImageUrl('data:image/svg+xml;base64,PHN2Zz4=', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('data:text/html;base64,PGgxPg==', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('data:image/png,<svg onload=alert(1)>', API, ORIGIN), false);
});

test('allows https only on the API host or the app origin', () => {
  assert.equal(isSafeImageUrl(`${API}/api/v1/image/file/a.png`, API, ORIGIN), true);
  assert.equal(isSafeImageUrl(`${ORIGIN}/a.png`, API, ORIGIN), true);
  assert.equal(isSafeImageUrl('https://evil.com/a.png?leak=secret', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('http://api.example-app.test/a.png', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('https://user:pw@api.example-app.test/a.png', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('https://evil.com/a.png', undefined, undefined), false);
});

test('rejects script and empty URLs', () => {
  assert.equal(isSafeImageUrl('javascript:alert(1)', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('  ', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('', API, ORIGIN), false);
  assert.equal(isSafeImageUrl('java\nscript:alert(1)', API, ORIGIN), false);
});
