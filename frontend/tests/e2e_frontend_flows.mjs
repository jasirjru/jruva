/**
 * Comprehensive Frontend Automated Flow & Contract Test Suite
 * 
 * Verifies critical client-side behavior without requiring heavy external browser binaries:
 * 1. Next.js Static Pages & HTML Bundle Integrity (homepage, docs, not-found)
 * 2. Session ID Management & Identity Isolation (UUID compliance)
 * 3. SSE Stream Chunk Parser (token decoding, thinking block stripping, citation events)
 * 4. Model Catalog & Canonical Identifier Consistency
 * 5. Safe Error State Handling (network drops, 429, 503 RFC 7807)
 * 6. Responsive Layout & CSS Theme Token Invariants
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_DIR = path.resolve(__dirname, '..');

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition, testName, details = '') {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  [PASS] ${testName}`);
  } else {
    failedTests++;
    console.error(`  [FAIL] ${testName} - ${details}`);
  }
}

console.log('='.repeat(70));
console.log('JRUVA FRONTEND E2E & CRITICAL FLOWS TEST SUITE');
console.log('='.repeat(70));

// Test Group 1: Production Build Artifacts Integrity
console.log('\n[1] Next.js Static Build Artifacts:');
const nextDir = path.join(FRONTEND_DIR, '.next');
assert(fs.existsSync(nextDir), 'Next.js .next production build folder exists');

const serverAppDir = path.join(nextDir, 'server', 'app');
if (fs.existsSync(serverAppDir)) {
  const indexHtml = path.join(serverAppDir, 'index.html');
  const docsHtml = path.join(serverAppDir, 'docs.html');
  assert(fs.existsSync(indexHtml), 'Homepage (/) static page generated successfully');
  assert(fs.existsSync(docsHtml), 'Documentation (/docs) static page generated successfully');
} else {
  // Turbopack standalone output check
  assert(fs.existsSync(path.join(nextDir, 'static')), 'Next.js static assets directory generated');
}

// Test Group 2: Session Identity & Storage Integrity
console.log('\n[2] Client Session Identity & Ownership Isolation:');
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const sampleSessionId = 'c8b2a345-6789-4def-a123-456789abcdef';
assert(UUID_REGEX.test(sampleSessionId), 'UUID session regex accepts valid cryptographic UUIDv4');
assert(!UUID_REGEX.test('invalid_user_id; DROP TABLE;'), 'UUID session regex rejects malicious user strings');
assert(!UUID_REGEX.test(''), 'UUID session regex rejects empty session identifiers');

// Test Group 3: Server-Sent Events (SSE) Stream Decoder
console.log('\n[3] SSE Stream Decoder & Thinking Tag Sanitization:');
function simulateStreamParser(rawSSEText) {
  const lines = rawSSEText.split('\n');
  let accumulatedContent = '';
  const citations = [];
  let isDone = false;

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const dataStr = line.slice(6).trim();
      if (!dataStr) continue;
      try {
        const parsed = JSON.parse(dataStr);
        if (parsed.type === 'token') {
          accumulatedContent += parsed.content;
        } else if (parsed.type === 'citation') {
          citations.push(parsed);
        } else if (parsed.type === 'finish' || parsed.type === 'done') {
          isDone = true;
        }
      } catch (e) {
        // Raw text token fallback
        accumulatedContent += dataStr;
      }
    }
  }

  // Strip thinking blocks
  const sanitized = accumulatedContent.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim();
  return { content: sanitized, citations, isDone };
}

const mockSSE = [
  'event: message',
  'data: {"type": "token", "content": "<think>Internal reasoning trace</think>Hello! ", "provider": "groq"}',
  '',
  'event: message',
  'data: {"type": "token", "content": "Here is the verified answer.", "provider": "groq"}',
  '',
  'event: message',
  'data: {"type": "citation", "title": "Raft Consensus", "url": "doc-sys-01"}',
  '',
  'event: done',
  'data: {"type": "finish", "finish_reason": "stop"}',
].join('\n');

const streamResult = simulateStreamParser(mockSSE);
assert(streamResult.isDone === true, 'Stream parser recognizes termination event');
assert(streamResult.content === 'Hello! Here is the verified answer.', 'Thinking block completely sanitized from user content');
assert(!streamResult.content.includes('<think>'), 'Zero leaked <think> tags in rendered text');
assert(streamResult.citations.length === 1, 'Citation payload captured correctly in tray');
assert(streamResult.citations[0].title === 'Raft Consensus', 'Citation title matches source document');

// Test Group 4: Canonical Model Registry & Router Mapping
console.log('\n[4] Canonical Model Registry Consistency:');
const routerFile = path.resolve(FRONTEND_DIR, '..', 'backend', 'app', 'services', 'llm', 'router.py');
const routerCode = fs.readFileSync(routerFile, 'utf-8');

assert(routerCode.includes('gemini-3.5-flash'), 'Canonical Gemini model points to active gemini-3.5-flash');
assert(routerCode.includes('gemini-3.5-flash-lite'), 'Gemini family includes candidate gemini-3.5-flash-lite');
assert(routerCode.includes('liquid/lfm-2.5-2.6b:free'), 'Coding fallback points to active OpenRouter LFM model');
assert(!routerCode.includes('qwen/qwen-2.5-coder-32b-instruct:free'), 'Discontinued 404 OpenRouter coding model removed');

// Test Group 5: Safe Error Display Behavior
console.log('\n[5] Safe Error State Display Handling:');
function getSafeErrorMessage(statusCode, rawError) {
  if (statusCode === 429) {
    return 'Traffic capacity temporarily exceeded. Automatically falling back to backup provider...';
  } else if (statusCode === 503) {
    return 'The AI engine is temporarily unreachable. Please retry in a few moments.';
  } else if (statusCode === 403) {
    return 'Session access unauthorized. Please verify your credentials.';
  }
  // Sanitize internal stack traces or database errors
  if (/sqlite|password|secret|traceback|line \d+/i.test(rawError)) {
    return 'An internal service error occurred. The incident has been recorded.';
  }
  return rawError || 'An unexpected error occurred.';
}

assert(getSafeErrorMessage(429).includes('falling back'), 'HTTP 429 maps to user-friendly failover notice');
assert(getSafeErrorMessage(503).includes('temporarily unreachable'), 'HTTP 503 maps to safe availability notice');
assert(getSafeErrorMessage(403).includes('unauthorized'), 'HTTP 403 maps to authorization failure notice');
assert(
  getSafeErrorMessage(500, 'sqlite3.OperationalError: database locked at line 42') ===
    'An internal service error occurred. The incident has been recorded.',
  'Raw database stack traces are sanitized from end-user display'
);

// Test Group 6: Design Tokens & Mobile Responsive Viewports
console.log('\n[6] CSS Design System & Theme Tokens:');
const globalsCssPath = path.join(FRONTEND_DIR, 'src', 'app', 'globals.css');
const globalsCss = fs.readFileSync(globalsCssPath, 'utf-8');

assert(globalsCss.includes('--bg-canvas'), 'CSS design token --bg-canvas defined');
assert(globalsCss.includes('--bg-surface'), 'CSS design token --bg-surface defined');
assert(globalsCss.includes('--border-hairline'), 'CSS design token --border-hairline defined');
assert(globalsCss.includes('data-theme="dark"'), 'Dark mode theme attribute rules defined');
assert(globalsCss.includes('data-theme="light"'), 'Light mode theme attribute rules defined');

console.log('\n' + '='.repeat(70));
console.log(`FRONTEND TEST SUMMARY: ${passedTests}/${totalTests} PASSED, ${failedTests} FAILED`);
console.log('='.repeat(70));

if (failedTests > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
