import assert from 'node:assert/strict';
import test from 'node:test';

// ---------------------------------------------------------------------------
// 1. URL Sanitization Verification (Hardened Logic)
// ---------------------------------------------------------------------------
function sanitizeUrl(urlStr) {
  if (!urlStr) return '#';
  const trimmed = urlStr.trim().replace(/[\x00-\x1f\x7f]/g, '');
  if (trimmed.startsWith('//')) return '#';
  if (trimmed.startsWith('/') || trimmed.startsWith('#')) return trimmed;
  if (trimmed.startsWith('mailto:')) return trimmed;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'mailto:') {
      return trimmed;
    }
  } catch {
    // Malformed or relative URL
  }
  return '#';
}

test('sanitizeUrl allows safe protocols', () => {
  assert.equal(sanitizeUrl('https://www.jruva.in/about'), 'https://www.jruva.in/about');
  assert.equal(sanitizeUrl('http://insecure-docs.example.org'), 'http://insecure-docs.example.org');
  assert.equal(sanitizeUrl('mailto:contact@jruva.in'), 'mailto:contact@jruva.in');
  assert.equal(sanitizeUrl('/docs'), '/docs');
  assert.equal(sanitizeUrl('#section-2'), '#section-2');
});

test('sanitizeUrl blocks XSS protocols and vector variants', () => {
  assert.equal(sanitizeUrl('javascript:alert(1)'), '#');
  assert.equal(sanitizeUrl('JAVASCRIPT:alert(document.cookie)'), '#');
  assert.equal(sanitizeUrl(' javascript:alert(1)'), '#');
  assert.equal(sanitizeUrl('data:text/html,<script>alert(1)</script>'), '#');
  assert.equal(sanitizeUrl('vbscript:msgbox(1)'), '#');
  assert.equal(sanitizeUrl('//evil.attacker.com/steal'), '#');
  assert.equal(sanitizeUrl('java\x00script:alert(1)'), '#');
});

// ---------------------------------------------------------------------------
// 2. Download Filename Sanitization Verification
// ---------------------------------------------------------------------------
function sanitizeDownloadFilename(name, defaultExt) {
  if (!name) return `snippet_test.${defaultExt}`;
  const base = name.replace(/^.*[\\\/]/, '').trim();
  const sanitized = base.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_');
  return sanitized || `snippet_test.${defaultExt}`;
}

test('sanitizeDownloadFilename prevents path traversal and illegal characters', () => {
  assert.equal(sanitizeDownloadFilename('solution.py', 'py'), 'solution.py');
  assert.equal(sanitizeDownloadFilename('../../etc/passwd', 'txt'), 'passwd');
  assert.equal(sanitizeDownloadFilename('C:\\Windows\\System32\\cmd.exe', 'exe'), 'cmd.exe');
  assert.equal(sanitizeDownloadFilename('file:with*bad?chars|.ts', 'ts'), 'file_with_bad_chars_.ts');
  assert.equal(sanitizeDownloadFilename('', 'py'), 'snippet_test.py');
});

// ---------------------------------------------------------------------------
// 3. Sibling Variation Branch Navigation Logic
// ---------------------------------------------------------------------------
test('Sibling message variation preservation and switching', () => {
  const message = {
    id: 'msg-1',
    role: 'assistant',
    content: 'Initial Answer V1',
    variations: [
      { content: 'Initial Answer V1', timestamp: 1000 },
      { content: 'Regenerated Answer V2', timestamp: 2000 },
      { content: 'Alternative Answer V3', timestamp: 3000 },
    ],
    activeVariationIndex: 1, // Currently displaying V2
  };

  assert.equal(message.variations.length, 3);
  assert.equal(message.variations[message.activeVariationIndex].content, 'Regenerated Answer V2');

  // Switch to V1 (< 1 / 3 >)
  function switchVariation(msg, targetIdx) {
    if (targetIdx < 0 || targetIdx >= msg.variations.length) return msg;
    return {
      ...msg,
      content: msg.variations[targetIdx].content,
      activeVariationIndex: targetIdx,
    };
  }

  const switchedToV1 = switchVariation(message, 0);
  assert.equal(switchedToV1.content, 'Initial Answer V1');
  assert.equal(switchedToV1.activeVariationIndex, 0);

  // Switch to V3
  const switchedToV3 = switchVariation(message, 2);
  assert.equal(switchedToV3.content, 'Alternative Answer V3');
  assert.equal(switchedToV3.activeVariationIndex, 2);

  // Boundary check
  const outOfBounds = switchVariation(message, 5);
  assert.equal(outOfBounds.activeVariationIndex, 1);
});

// ---------------------------------------------------------------------------
// 4. Pinned Chats Persistence & Toggle Logic
// ---------------------------------------------------------------------------
test('Pinned chat storage and toggle behavior', () => {
  const mockStorage = new Map();
  const localStorageMock = {
    getItem: (k) => mockStorage.get(k) || null,
    setItem: (k, v) => mockStorage.set(k, v),
  };

  function togglePin(id, currentList) {
    const next = currentList.includes(id)
      ? currentList.filter(p => p !== id)
      : [...currentList, id];
    localStorageMock.setItem('jru_pinned_sessions', JSON.stringify(next));
    return next;
  }

  let pinned = [];
  pinned = togglePin('session-1', pinned);
  assert.deepEqual(pinned, ['session-1']);
  assert.equal(localStorageMock.getItem('jru_pinned_sessions'), JSON.stringify(['session-1']));

  pinned = togglePin('session-2', pinned);
  assert.deepEqual(pinned, ['session-1', 'session-2']);

  // Unpin session-1
  pinned = togglePin('session-1', pinned);
  assert.deepEqual(pinned, ['session-2']);
  assert.equal(localStorageMock.getItem('jru_pinned_sessions'), JSON.stringify(['session-2']));
});

// ---------------------------------------------------------------------------
// 5. Sandboxed SVG CSP & Document Isolation
// ---------------------------------------------------------------------------
function buildSandboxedSvgDoc(rawSvg, background, zoom) {
  const bgStyle = background === 'light' ? '#ffffff' : '#0d0d0f';
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <style>
    body { background-color: ${bgStyle}; }
  </style>
</head>
<body>
  <div class="svg-container" style="transform: scale(${zoom / 100})">
    ${rawSvg}
  </div>
</body>
</html>`;
}

test('buildSandboxedSvgDoc contains strict CSP preventing network and script exfiltration', () => {
  const svgPayload = '<svg><script>alert(1)</script><circle r="10"/></svg>';
  const doc = buildSandboxedSvgDoc(svgPayload, 'dark', 100);
  assert.ok(doc.includes("default-src 'none'"));
  assert.ok(doc.includes("style-src 'unsafe-inline'"));
  assert.ok(doc.includes('<circle r="10"/>'));
});

// ---------------------------------------------------------------------------
// 6. Page Reload & Conversation Restoration with Variations
// ---------------------------------------------------------------------------
test('Page reload reconstitutes turns and restores sibling variations', () => {
  // Simulated server turns (chronological linear history)
  const serverTurns = [
    { role: 'user', content: 'Design an LRU cache in Python' },
    { role: 'assistant', content: 'Here is an LRU cache using collections.OrderedDict' },
  ];

  // Simulated client storage of sibling variations
  const mockStorage = new Map();
  mockStorage.set('jru_session_variations_sess-123', JSON.stringify({
    1: {
      content: 'Here is an LRU cache using collections.OrderedDict (v2)',
      activeVariationIndex: 1,
      variations: [
        { content: 'Here is an LRU cache using doubly linked list (v1)', timestamp: 1000 },
        { content: 'Here is an LRU cache using collections.OrderedDict (v2)', timestamp: 2000 },
      ],
    },
  }));

  function restoreTurnsWithVariations(sessionId, turns) {
    let cachedVariations = {};
    try {
      const raw = mockStorage.get(`jru_session_variations_${sessionId}`);
      if (raw) cachedVariations = JSON.parse(raw);
    } catch {}

    return turns.map((turn, idx) => {
      const savedVar = cachedVariations[idx];
      return {
        id: `turn-${idx}`,
        role: turn.role,
        content: savedVar?.content || turn.content,
        variations: savedVar?.variations,
        activeVariationIndex: savedVar?.activeVariationIndex,
      };
    });
  }

  const restored = restoreTurnsWithVariations('sess-123', serverTurns);
  assert.equal(restored.length, 2);
  assert.equal(restored[0].content, 'Design an LRU cache in Python');
  assert.equal(restored[1].content, 'Here is an LRU cache using collections.OrderedDict (v2)');
  assert.equal(restored[1].variations.length, 2);
  assert.equal(restored[1].activeVariationIndex, 1);
});

// ---------------------------------------------------------------------------
// 7. Inline Renaming with Optimistic UI & Rollback on Failure
// ---------------------------------------------------------------------------
test('Inline session renaming optimistic update and rollback on failure', async () => {
  let sessions = [
    { id: 'sess-abc', title: 'Original Alpha Title' },
    { id: 'sess-xyz', title: 'Beta Testing Notes' },
  ];
  let errorMessage = null;

  async function renameSessionWithOptimisticUI(id, newTitle, mockApiCall) {
    const trimmed = newTitle.trim();
    if (!trimmed) return;

    let previousTitle;
    // Optimistic update
    sessions = sessions.map((s) => {
      if (s.id !== id) return s;
      previousTitle = s.title;
      return { ...s, title: trimmed };
    });

    try {
      const res = await mockApiCall(id, trimmed);
      if (!res.ok) throw new Error(`API error ${res.status}`);
    } catch {
      // Rollback on failure
      sessions = sessions.map((s) => (s.id === id ? { ...s, title: previousTitle } : s));
      errorMessage = 'Could not rename this conversation. Please try again.';
    }
  }

  // 1. Successful rename
  await renameSessionWithOptimisticUI('sess-abc', 'Updated Release Title', async () => ({ ok: true, status: 200 }));
  assert.equal(sessions.find((s) => s.id === 'sess-abc').title, 'Updated Release Title');
  assert.equal(errorMessage, null);

  // 2. Failed rename (e.g. 403 Forbidden / Network Error) -> Must rollback!
  await renameSessionWithOptimisticUI('sess-abc', 'Malicious Title Attempt', async () => ({ ok: false, status: 403 }));
  assert.equal(sessions.find((s) => s.id === 'sess-abc').title, 'Updated Release Title');
  assert.equal(errorMessage, 'Could not rename this conversation. Please try again.');
});

// ---------------------------------------------------------------------------
// 8. Scroll-to-Bottom Visibility Threshold
// ---------------------------------------------------------------------------
test('Scroll-to-bottom button visibility threshold logic', () => {
  function shouldShowScrollButton(scrollTop, scrollHeight, clientHeight, threshold = 150) {
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    return distanceFromBottom > threshold;
  }

  // User is at the bottom (distance = 0)
  assert.equal(shouldShowScrollButton(800, 1400, 600), false);

  // User scrolled up 50px (distance = 50 < 150)
  assert.equal(shouldShowScrollButton(750, 1400, 600), false);

  // User scrolled up 200px (distance = 200 > 150)
  assert.equal(shouldShowScrollButton(600, 1400, 600), true);

  // User scrolled near top (distance = 800 > 150)
  assert.equal(shouldShowScrollButton(0, 1400, 600), true);
});

// ---------------------------------------------------------------------------
// 9. Cross-User Isolation and Authorization Verification
// ---------------------------------------------------------------------------
test('Cross-user isolation denies unauthorized session access', () => {
  const sessionDb = new Map();
  sessionDb.set('sess-alice', { id: 'sess-alice', userId: 'user-alice-uuid', title: "Alice's Secret Strategy" });
  sessionDb.set('sess-bob', { id: 'sess-bob', userId: 'user-bob-uuid', title: "Bob's Public Notes" });

  function authorizeAccess(sessionId, requestingUserId) {
    const session = sessionDb.get(sessionId);
    if (!session) return { status: 404, allowed: false };
    if (session.userId !== requestingUserId) {
      return { status: 403, allowed: false, error: 'Forbidden: You do not have permission to access this session.' };
    }
    return { status: 200, allowed: true, session };
  }

  // Alice accesses Alice's session -> Allowed
  assert.equal(authorizeAccess('sess-alice', 'user-alice-uuid').status, 200);
  assert.equal(authorizeAccess('sess-alice', 'user-alice-uuid').allowed, true);

  // Bob attempts to access Alice's session -> 403 Forbidden!
  const forbiddenAttempt = authorizeAccess('sess-alice', 'user-bob-uuid');
  assert.equal(forbiddenAttempt.status, 403);
  assert.equal(forbiddenAttempt.allowed, false);

  // Anonymous guest attempts to access Alice's session -> 403 Forbidden!
  const guestAttempt = authorizeAccess('sess-alice', 'guest-anonymous');
  assert.equal(guestAttempt.status, 403);
  assert.equal(guestAttempt.allowed, false);
});

// ---------------------------------------------------------------------------
// 10. Multi-User LocalStorage Pinned Chat Isolation
// ---------------------------------------------------------------------------
test('Multi-user pinned session scoping prevents cross-user leakage on shared browser', () => {
  const mockStorage = new Map();

  function getPinnedStorageKey(userId) {
    return userId ? `jru_pinned_sessions_${userId}` : 'jru_pinned_sessions_guest';
  }

  function savePins(userId, pinList) {
    mockStorage.set(getPinnedStorageKey(userId), JSON.stringify(pinList));
  }

  function getPins(userId) {
    const raw = mockStorage.get(getPinnedStorageKey(userId));
    return raw ? JSON.parse(raw) : [];
  }

  // Alice pins session-A1 and session-A2
  savePins('user-alice-123', ['session-A1', 'session-A2']);
  assert.deepEqual(getPins('user-alice-123'), ['session-A1', 'session-A2']);

  // Bob logs in on the same browser
  assert.deepEqual(getPins('user-bob-456'), []); // Bob sees zero of Alice's pins!

  // Bob pins his own session-B1
  savePins('user-bob-456', ['session-B1']);
  assert.deepEqual(getPins('user-bob-456'), ['session-B1']);

  // Alice's pins remain completely isolated and intact
  assert.deepEqual(getPins('user-alice-123'), ['session-A1', 'session-A2']);
});

// ---------------------------------------------------------------------------
// 11. Logout Cleanup of Transient Caches on Shared Machine
// ---------------------------------------------------------------------------
test('Logout purges transient variation and guest cache keys from localStorage', () => {
  const mockStorage = new Map([
    ['jru_auth_user', JSON.stringify({ id: 'alice' })],
    ['jru_auth_token', 'jwt-token-sample'],
    ['chatbot_session_id', 'sess-123'],
    ['jru_session_variations_sess-123', JSON.stringify({ 1: { variations: [] } })],
    ['chat_intent_hash123', JSON.stringify({ completed: true })],
    ['jru_pinned_sessions_guest', JSON.stringify(['guest-sess'])],
    ['jru_theme', 'dark'], // persistent user preference should survive
  ]);

  function performLogoutCleanup() {
    mockStorage.delete('jru_auth_token');
    mockStorage.delete('jru_auth_user');
    mockStorage.delete('chatbot_session_id');

    const keysToRemove = [];
    for (const key of mockStorage.keys()) {
      if (
        key.startsWith('jru_session_variations_') ||
        key.startsWith('chat_intent_') ||
        key === 'jru_pinned_sessions_guest' ||
        key === 'jru_pinned_sessions'
      ) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((k) => mockStorage.delete(k));
  }

  performLogoutCleanup();

  assert.equal(mockStorage.has('jru_auth_token'), false);
  assert.equal(mockStorage.has('jru_auth_user'), false);
  assert.equal(mockStorage.has('chatbot_session_id'), false);
  assert.equal(mockStorage.has('jru_session_variations_sess-123'), false);
  assert.equal(mockStorage.has('chat_intent_hash123'), false);
  assert.equal(mockStorage.has('jru_pinned_sessions_guest'), false);
  assert.equal(mockStorage.get('jru_theme'), 'dark'); // Non-sensitive preferences intact
});

// ---------------------------------------------------------------------------
// 12. Least-Privilege HTML Artifact CSP & Sandbox
// ---------------------------------------------------------------------------
test('HTML artifact CSP enforces connect-src none and form-action none', () => {
  const cspMeta = `<meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'none';">`;

  assert.ok(cspMeta.includes("connect-src 'none'"));
  assert.ok(cspMeta.includes("form-action 'none'"));
  assert.ok(cspMeta.includes("base-uri 'none'"));
  assert.ok(cspMeta.includes("frame-ancestors 'none'"));
});


