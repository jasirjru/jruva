// Tab-scoped intent receipts contain only digests, UUIDs and completion state.
export interface IntentReceipt { id: string; completed: boolean; imageGenerationAllowed: boolean }

export async function chatIntentKey(sessionId: string, content: string): Promise<string> {
  const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(content)));
  const digest = Array.from(bytes).map(byte => byte.toString(16).padStart(2, '0')).join('');
  return `jruva:chat-intent:${sessionId}:${digest}`;
}

export function readIntent(key: string): IntentReceipt | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (value && typeof value.completed === 'boolean' && typeof value.imageGenerationAllowed === 'boolean' && typeof value.id === 'string' &&
        /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value.id)) return value;
  } catch {}
  return null;
}

export function writeIntent(key: string, value: IntentReceipt): boolean {
  try { sessionStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
}
