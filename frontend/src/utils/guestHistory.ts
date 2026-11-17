import type { HistoryTurn, Message } from '@/types/chat';

// Must stay within the backend limits in app/schemas/chat.py
// (10 turns, 8000 chars each, 20,000 chars total).
export const MAX_GUEST_HISTORY_TURNS = 10;
const MAX_TURN_CHARS = 2000;

const isFailedReply = (m: Message): boolean =>
  m.role === 'assistant' && (m.status === 'error' || m.status === 'streaming');

/**
 * Guests have no server-side history, so the browser resends its recent turns.
 * A failed reply is dropped together with the question that caused it, so the
 * model never sees two user turns in a row. Newest turn last.
 */
export function buildGuestHistory(messages: readonly Message[]): HistoryTurn[] {
  const turns = messages.reduce<HistoryTurn[]>((acc, m, i) => {
    if (m.role !== 'user' && m.role !== 'assistant') return acc;
    if (isFailedReply(m) || m.content.trim().length === 0) return acc;
    const next = messages[i + 1];
    if (m.role === 'user' && next && isFailedReply(next)) return acc;
    return [...acc, { role: m.role, content: m.content.slice(0, MAX_TURN_CHARS) }];
  }, []);
  return turns.slice(-MAX_GUEST_HISTORY_TURNS);
}
