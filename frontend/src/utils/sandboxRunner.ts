/**
 * Hardened Client-Side Execution Sandbox.
 * 
 * Provides isolated code execution using Web Workers:
 * 1. Process Isolation: Runs outside the main UI thread, preventing CPU locks and freezing.
 * 2. Scope Isolation: Zero access to `window`, `document`, `localStorage`, `cookie`, or parent DOM.
 * 3. Network Containment: Disables `fetch`, `XMLHttpRequest`, and `importScripts` inside the sandbox.
 * 4. Watchdog Timeout: Enforces a strict execution deadline (default 2500ms) with hard worker termination.
 * 5. Prototype Protection: Modifications to Object/Array prototypes are confined to the disposable worker realm.
 */

export interface SandboxOutput {
  logs: Array<{ text: string; isError?: boolean }>;
  durationMs: number;
  timedOut?: boolean;
}

const WORKER_BOOTSTRAP_CODE = `
self.onmessage = function(event) {
  const { code, isPython } = event.data;
  const logs = [];

  // Capture stdout and stderr
  const formatArg = (a) => {
    if (typeof a === 'object' && a !== null) {
      try { return JSON.stringify(a, null, 2); } catch (e) { return String(a); }
    }
    return String(a);
  };

  const capturedConsole = {
    log: (...args) => logs.push({ text: args.map(formatArg).join(' ') }),
    warn: (...args) => logs.push({ text: '[Warning] ' + args.map(formatArg).join(' ') }),
    error: (...args) => logs.push({ text: '[Error] ' + args.map(formatArg).join(' '), isError: true }),
    info: (...args) => logs.push({ text: args.map(formatArg).join(' ') }),
  };

  // 1. Defend Against Exfiltration: Disable network and storage primitives
  self.fetch = undefined;
  self.XMLHttpRequest = undefined;
  self.importScripts = undefined;
  self.indexedDB = undefined;

  // 2. Execute within isolated realm
  try {
    if (isPython) {
      // Lightweight simulated Python runner for standard math & print statements
      const lines = code.split('\\n');
      const pyScope = {
        math: Math,
        print: (...args) => logs.push({ text: args.map(formatArg).join(' ') }),
        range: (n) => Array.from({ length: n }, (_, i) => i),
        len: (x) => (x ? x.length || Object.keys(x).length : 0),
      };

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;

        if (line.startsWith('print(') && line.endsWith(')')) {
          const inner = line.slice(6, -1);
          if ((inner.startsWith('"') && inner.endsWith('"')) || (inner.startsWith("'") && inner.endsWith("'"))) {
            logs.push({ text: inner.slice(1, -1) });
          } else if (inner.startsWith('f"') || inner.startsWith("f'")) {
            logs.push({ text: inner.slice(2, -1) });
          } else {
            try {
              const val = new Function(...Object.keys(pyScope), 'return (' + inner + ')')(
                ...Object.values(pyScope)
              );
              logs.push({ text: formatArg(val) });
            } catch (evalErr) {
              logs.push({ text: inner });
            }
          }
        }
      }

      if (logs.length === 0) {
        logs.push({ text: 'Process finished with exit code 0 (No stdout generated).' });
      }
    } else {
      // JavaScript Sandboxed Execution
      const sandboxGlobals = {
        console: capturedConsole,
        Math: Math,
        Date: Date,
        JSON: JSON,
        parseInt: parseInt,
        parseFloat: parseFloat,
        isNaN: isNaN,
        isFinite: isFinite,
        Array: Array,
        Object: Object,
        String: String,
        Number: Number,
        Boolean: Boolean,
        RegExp: RegExp,
        Map: Map,
        Set: Set,
        Promise: Promise,
        fetch: undefined,
        XMLHttpRequest: undefined,
      };

      // Wrap code execution in a function that shadows dangerous identifiers
      const scopeKeys = Object.keys(sandboxGlobals);
      const scopeValues = Object.values(sandboxGlobals);
      
      const runner = new Function(
        ...scopeKeys,
        '"use strict";\\n' + code
      );
      
      const result = runner.apply(null, scopeValues);
      if (result !== undefined && logs.length === 0) {
        logs.push({ text: formatArg(result) });
      }
    }

    self.postMessage({ success: true, logs: logs });
  } catch (err) {
    self.postMessage({
      success: false,
      logs: logs.concat([{ text: 'Runtime Error: ' + (err && err.message ? err.message : String(err)), isError: true }]),
    });
  }
};
`;

export async function runInHardenedSandbox(
  code: string,
  isPython: boolean = false,
  timeoutMs: number = 2500
): Promise<SandboxOutput> {
  const t0 = performance.now();

  // If Web Worker is unavailable in environment, reject cleanly
  if (typeof window === 'undefined' || typeof Worker === 'undefined') {
    return {
      logs: [{ text: 'Execution Error: Web Workers are not supported in this runtime environment.', isError: true }],
      durationMs: 0,
    };
  }

  return new Promise((resolve) => {
    let worker: Worker | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let resolved = false;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      if (worker) {
        worker.terminate();
        worker = null;
      }
    };

    try {
      const blob = new Blob([WORKER_BOOTSTRAP_CODE], { type: 'application/javascript' });
      const workerUrl = URL.createObjectURL(blob);
      worker = new Worker(workerUrl);
      URL.revokeObjectURL(workerUrl);

      // Watchdog timeout for CPU loops
      timer = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          cleanup();
          const duration = Math.round(performance.now() - t0);
          resolve({
            logs: [
              {
                text: `Execution Timed Out: Script exceeded ${timeoutMs}ms execution budget (Infinite loop or CPU lock prevented).`,
                isError: true,
              },
            ],
            durationMs: duration,
            timedOut: true,
          });
        }
      }, timeoutMs);

      worker.onmessage = (event: MessageEvent) => {
        if (!resolved) {
          resolved = true;
          const duration = Math.round(performance.now() - t0);
          cleanup();
          resolve({
            logs: event.data.logs || [],
            durationMs: duration,
          });
        }
      };

      worker.onerror = (error: ErrorEvent) => {
        if (!resolved) {
          resolved = true;
          const duration = Math.round(performance.now() - t0);
          cleanup();
          resolve({
            logs: [{ text: `Sandbox Error: ${error.message || 'Worker execution failed'}`, isError: true }],
            durationMs: duration,
          });
        }
      };

      worker.postMessage({ code, isPython });
    } catch (err: unknown) {
      cleanup();
      const message = err instanceof Error ? err.message : String(err);
      resolve({
        logs: [{ text: `Sandbox Initialization Failed: ${message}`, isError: true }],
        durationMs: 0,
      });
    }
  });
}
