'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, MessageVariation, ChatStreamEvent, SourceCitation, ModelOption, AttachmentItem, UserTier } from '@/types/chat';
import { AVAILABLE_PERSONAS } from '@/config/personas';
import { haptics } from '@/utils/haptics';
import { presentStatusLabel, presentUserFacingError } from '@/utils/presentAssistant';
import { chatIntentKey, readIntent, writeIntent } from '@/utils/chatIntent';
import { buildGuestHistory } from '@/utils/guestHistory';
import { useAuth } from '@/context/AuthContext';

const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
    const envUrl = process.env.NEXT_PUBLIC_API_URL || '';
    if (!isLocalhost || !envUrl || envUrl.includes('localhost') || envUrl.includes('127.0.0.1')) {
      return '';
    }
    return envUrl;
  }
  return '';
};

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

interface SavedTurn { role: 'user' | 'assistant'; content: string; createdAt?: string | number }

async function restoreTurns(id: string, turns: SavedTurn[]): Promise<Message[]> {
  let previousIntent: ReturnType<typeof readIntent> = null;
  const restored: Message[] = [];
  for (const [index, turn] of turns.entries()) {
    if (turn.role === 'user') previousIntent = readIntent(await chatIntentKey(id, turn.content));
    const parsed = turn.role === 'assistant' ? parseThoughtAndContent(turn.content) : { content: turn.content, thought: undefined };
    restored.push({
      id: turn.role === 'user' && previousIntent ? previousIntent.id : `restored-${index}`,
      role: turn.role, content: parsed.content, thought: parsed.thought,
      isThinkingActive: false, createdAt: new Date(turn.createdAt || Date.now()),
      status: turn.role === 'assistant' && previousIntent && !previousIntent.completed ? 'error' : 'completed',
      imageGenerationAllowed: previousIntent?.imageGenerationAllowed ?? false,
    });
  }
  return restored;
}

export const getClientId = (): string => {
  if (typeof window === 'undefined') return '';
  try {
    let cid = localStorage.getItem('chatbot_client_id');
    if (!cid || !UUID_REGEX.test(cid)) {
      cid = crypto.randomUUID();
      localStorage.setItem('chatbot_client_id', cid);
    }
    return cid;
  } catch {
    return '';
  }
};

// Session history is private: the API answers 401 for signed-out visitors, so skip those calls.
const hasSignedInUser = (): boolean => {
  if (typeof window === 'undefined') return false;
  try {
    return Boolean(localStorage.getItem('jru_auth_user') || localStorage.getItem('jru_auth_token'));
  } catch {
    return false;
  }
};

export const getAuthAndClientHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};
  if (typeof window !== 'undefined') {
    const clientId = getClientId();
    if (clientId) {
      headers['X-Client-Id'] = clientId;
    }
    const authToken = localStorage.getItem('jru_auth_token');
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }
  }
  return headers;
};

export function parseThoughtAndContent(rawText: string): { thought?: string; content: string; isThinkingActive: boolean } {
  if (!rawText) return { content: '', isThinkingActive: false };

  // Standard <think>...</think> tag protocol
  const thinkStart = rawText.indexOf('<think>');
  if (thinkStart !== -1) {
    const thinkEnd = rawText.indexOf('</think>', thinkStart);
    if (thinkEnd !== -1) {
      const beforeThink = rawText.slice(0, thinkStart).trim();
      const thoughtText = rawText.slice(thinkStart + 7, thinkEnd).trim();
      const afterThink = rawText.slice(thinkEnd + 8).trimStart();
      const mainContent = beforeThink ? `${beforeThink}\n\n${afterThink}` : afterThink;
      return {
        thought: thoughtText,
        content: mainContent,
        isThinkingActive: false,
      };
    } else {
      const beforeThink = rawText.slice(0, thinkStart).trim();
      const thoughtText = rawText.slice(thinkStart + 7);
      return {
        thought: thoughtText,
        content: beforeThink,
        isThinkingActive: true,
      };
    }
  }

  // Fallback for <thought>...</thought> tag variant
  const altStart = rawText.indexOf('<thought>');
  if (altStart !== -1) {
    const altEnd = rawText.indexOf('</thought>', altStart);
    if (altEnd !== -1) {
      const beforeThink = rawText.slice(0, altStart).trim();
      const thoughtText = rawText.slice(altStart + 9, altEnd).trim();
      const afterThink = rawText.slice(altEnd + 10).trimStart();
      const mainContent = beforeThink ? `${beforeThink}\n\n${afterThink}` : afterThink;
      return {
        thought: thoughtText,
        content: mainContent,
        isThinkingActive: false,
      };
    } else {
      const beforeThink = rawText.slice(0, altStart).trim();
      const thoughtText = rawText.slice(altStart + 9);
      return {
        thought: thoughtText,
        content: beforeThink,
        isThinkingActive: true,
      };
    }
  }

  return {
    content: rawText,
    isThinkingActive: false,
  };
}

export interface ChatSession {
  id: string;
  title: string;
  updatedAt: string | number;
}

export const DEFAULT_MODELS: ModelOption[] = [
  {
    id: 'auto',
    name: 'Auto',
    provider: 'auto',
    badge: '🛡️ Auto',
    description: 'Intelligent auto-routing based on prompt complexity.',
    recommended: true,
    available: true,
    health: 'healthy',
    runtime: 'cloud',
    fallback_capable: true,
  },
  {
    id: 'gemini-3.6-flash',
    name: 'Gemini Flash',
    provider: 'gemini',
    badge: '✦ Flash',
    description: 'General chat, quick Q&A, and everyday tasks.',
    recommended: true,
    available: true,
    health: 'healthy',
    runtime: 'cloud',
    fallback_capable: true,
  },
  {
    id: 'qwen/qwen3.8-27b',
    name: 'Qwen 3.8',
    provider: 'groq',
    badge: '💻 Coder',
    description: 'High-throughput coding, refactoring, and algorithmic reasoning on Groq LPU.',
    available: true,
    health: 'healthy',
    runtime: 'cloud',
    fallback_capable: true,
  },
  {
    id: 'groq/deepseek-r1-distill-llama-70b',
    name: 'DeepSeek R1',
    provider: 'groq',
    badge: '🧠 Unavailable',
    description: 'Upstream DeepSeek endpoint decommissioned on Groq.',
    available: false,
    health: 'misconfigured',
    runtime: 'cloud',
    fallback_capable: false,
    hidden: true,
  },
];

export function useChatStream() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>('s-default');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('auto');
  const [selectedTier, setSelectedTier] = useState<UserTier>('S+++');
  const [selectedPersona, setSelectedPersona] = useState<string>('general');
  const [availableModels, setAvailableModels] = useState<ModelOption[]>(DEFAULT_MODELS);

  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  // Synchronous send lock: holds the controller of the send that owns it, so a
  // double click or Enter repeat cannot start a second stream before React re-renders.
  const inFlightRef = useRef<AbortController | null>(null);
  // Incremented on every session switch; a load that finishes late is ignored.
  const sessionSwitchTokenRef = useRef(0);
  const isMountedRef = useRef(true);
  const authEpochRef = useRef(0);

  // Abort the active stream on unmount; finally blocks then skip state updates.
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      inFlightRef.current = null;
    };
  }, []);
  // Latest turns for guest history; resend paths set it directly after trimming.
  const messagesRef = useRef<Message[]>(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  // Auth is an httpOnly cookie, so the signed-in state comes from AuthContext.
  const { isAuthenticated } = useAuth();
  const isAuthenticatedRef = useRef(isAuthenticated);
  useEffect(() => {
    isAuthenticatedRef.current = isAuthenticated;
  }, [isAuthenticated]);

  // Initialize session ID and tier from localStorage on mount
  useEffect(() => {
    try {
      const storedId = localStorage.getItem('chatbot_session_id');
      if (storedId && (UUID_REGEX.test(storedId) || storedId.startsWith('s-'))) {
        setSessionId(storedId);
      } else {
        const freshId = crypto.randomUUID();
        setSessionId(freshId);
        localStorage.setItem('chatbot_session_id', freshId);
      }

      const storedModel = localStorage.getItem('chatbot_selected_model');
      if (storedModel) {
        setSelectedModel(storedModel);
      }

      const storedTier = localStorage.getItem('chatbot_selected_tier') as UserTier | null;
      if (storedTier && (storedTier === 'B' || storedTier === 'A' || storedTier === 'S' || storedTier === 'S++' || storedTier === 'S+++')) {
        setSelectedTier(storedTier);
      }
    } catch {}
  }, []);

  // Fetch real sessions from server-side API (scoped to user if authenticated, or device client id if guest)
  const fetchSessions = useCallback(async () => {
    const authEpoch = authEpochRef.current;
    if (!hasSignedInUser()) {
      setSessions([]);
      return;
    }
    try {
      const headers = getAuthAndClientHeaders();
      const res = await fetch(`${getApiBaseUrl()}/api/v1/sessions`, {
        headers,
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        if (authEpoch !== authEpochRef.current) return;
        if (data && Array.isArray(data.sessions)) {
          setSessions(data.sessions.map((s: { id: string; title?: string; updatedAt?: string | number; createdAt?: string | number }) => ({
            id: s.id,
            title: s.title || 'Untitled',
            updatedAt: s.updatedAt || s.createdAt || Date.now(),
          })));
        }
      } else if (res.status === 401) {
        setSessions([]);
      }
    } catch {
      // Silently fallback — sessions will be empty if backend is down
    }
  }, []);

  // Load sessions from server on mount and listen for login/logout events
  useEffect(() => {
    fetchSessions();

    let authChangeTimer: any = null;

    const handleAuthChange = () => {
      authEpochRef.current += 1;
      abortControllerRef.current?.abort();
      setMessages([]);
      setSessions([]);
      setError(null);
      setIsStreaming(false);
      setSessionId(crypto.randomUUID());
      fetchSessions();
    };

    const triggerAuthChange = () => {
      if (authChangeTimer) {
        clearTimeout(authChangeTimer);
      }
      authChangeTimer = setTimeout(() => {
        handleAuthChange();
      }, 50);
    };

    const handleStorageChange = (e: StorageEvent) => {
      // React only to genuine auth storage alterations or storage clearance
      const AUTH_STORAGE_KEYS = ['jru_auth_user', 'jru_auth_token', 'token'];
      if (e.key !== null && !AUTH_STORAGE_KEYS.includes(e.key)) {
        return;
      }
      if (e.key !== null && e.oldValue === e.newValue) {
        return;
      }
      triggerAuthChange();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('jru-auth-change', triggerAuthChange);
    return () => {
      if (authChangeTimer) clearTimeout(authChangeTimer);
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('jru-auth-change', triggerAuthChange);
    };
  }, [fetchSessions]);

  // Fetch real-time available models from backend
  useEffect(() => {
    fetch(`${getApiBaseUrl()}/api/v1/models`, { credentials: 'include' })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.models) && data.models.length > 0) {
          setAvailableModels(data.models);
        }
      })
      .catch(() => {});
  }, []);

  const handleSelectModel = useCallback((modelId: string) => {
    setSelectedModel(modelId);
    try {
      localStorage.setItem('chatbot_selected_model', modelId);
    } catch {}
  }, []);

  const handleSelectTier = useCallback((tier: UserTier) => {
    setSelectedTier(tier);
    try {
      localStorage.setItem('chatbot_selected_tier', tier);
    } catch {}
  }, []);

  // Persist current session id
  useEffect(() => {
    try {
      if (sessionId !== 's-default') {
        localStorage.setItem('chatbot_session_id', sessionId);
      }
    } catch {}
  }, [sessionId]);

  const stopGeneration = useCallback(() => {
    const activeController = abortControllerRef.current;
    if (activeController) {
      activeController.abort();
      abortControllerRef.current = null;
      // Release the send lock now so a new message can start; the aborted
      // send's finally sees it no longer owns the lock and leaves it alone.
      if (inFlightRef.current === activeController) inFlightRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const clearMessages = useCallback(() => {
    stopGeneration();
    setMessages([]);
    setError(null);
    const newSession = crypto.randomUUID();
    setSessionId(newSession);
  }, [stopGeneration]);

  // Load session messages from server (replaces localStorage reads)
  const switchSession = useCallback(
    async (id: string) => {
      const authEpoch = authEpochRef.current;
      const switchToken = ++sessionSwitchTokenRef.current;
      const isStale = () =>
        authEpoch !== authEpochRef.current || switchToken !== sessionSwitchTokenRef.current;
      stopGeneration();
      setError(null);
      setSessionId(id);

      // Load messages from server
      if (UUID_REGEX.test(id) && hasSignedInUser()) {
        try {
          const headers = getAuthAndClientHeaders();
          const res = await fetch(`${getApiBaseUrl()}/api/v1/sessions/${id}/messages`, {
            headers,
            credentials: 'include',
          });
          if (res.ok) {
            const data = await res.json();
            if (isStale()) return;
            if (data && Array.isArray(data.messages) && data.messages.length > 0) {
              const restored = await restoreTurns(id, data.messages);
              if (isStale()) return;
              setMessages(restored);
              return;
            }
          }
        } catch {}
      }

      if (isStale()) return;
      setMessages([]);
    },
    [stopGeneration]
  );

  // Auto-restore messages for active persisted session on initial mount / page reload
  const hasRestoredInitialSession = useRef(false);
  useEffect(() => {
    if (!hasRestoredInitialSession.current) {
      hasRestoredInitialSession.current = true;
      try {
        const storedId = localStorage.getItem('chatbot_session_id');
        if (storedId && UUID_REGEX.test(storedId) && storedId !== 's-default' && hasSignedInUser()) {
          const authEpoch = authEpochRef.current;
          fetch(`${getApiBaseUrl()}/api/v1/sessions/${storedId}/messages`, {
            headers: getAuthAndClientHeaders(),
            credentials: 'include',
          })
            .then((res) => (res.ok ? res.json() : null))
            .then(async (data) => {
              if (data && Array.isArray(data.messages) && data.messages.length > 0) {
                if (authEpoch !== authEpochRef.current) return;
                const restored = await restoreTurns(storedId, data.messages);
                if (authEpoch !== authEpochRef.current) return;
                setMessages((prev) => {
                  if (prev.length > 0) return prev; // Do not overwrite active conversation
                  return restored;
                });
              }
            })
            .catch(() => {});
        }
      } catch {}
    }
  }, []);

  const deleteSession = useCallback(
    async (id: string) => {
      // Delete from server
      if (UUID_REGEX.test(id)) {
        try {
          const headers = getAuthAndClientHeaders();
          await fetch(`${getApiBaseUrl()}/api/v1/sessions/${id}`, {
            method: 'DELETE',
            headers,
            credentials: 'include',
          });
        } catch {}
      }

      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (sessionId === id) {
        clearMessages();
      }
    },
    [clearMessages, sessionId]
  );

  const sendMessage = useCallback(
    async (
      content: string,
      attachments?: AttachmentItem[],
      webSearch?: boolean,
      retryIntentId?: string,
      imageGenerationAllowed = true,
      inheritedVariations?: MessageVariation[]
    ) => {
      const authEpoch = authEpochRef.current;
      const trimmed = content.trim();
      if ((!trimmed && (!attachments || attachments.length === 0)) || inFlightRef.current) return;

      const controller = new AbortController();
      inFlightRef.current = controller;

      setError(null);

      const userMessageId = retryIntentId || crypto.randomUUID();
      const assistantMessageId = crypto.randomUUID();

      const userMsg: Message = {
        id: userMessageId,
        role: 'user',
        content: trimmed || (attachments && attachments.length > 0 ? `Sent ${attachments.length} attachment(s)` : ''),
        attachments,
        createdAt: new Date(),
        status: 'completed',
        imageGenerationAllowed,
      };

      const initialAssistantMsg: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        createdAt: new Date(),
        status: 'streaming',
        variations: inheritedVariations,
      };

      const guestHistory = isAuthenticatedRef.current ? undefined : buildGuestHistory(messagesRef.current);

      setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
      setIsStreaming(true);
      abortControllerRef.current = controller;

      try {
        const intentStorageKey = await chatIntentKey(sessionId, trimmed || 'Analyze attached file(s)');
        const intentStored = writeIntent(intentStorageKey, { id: userMessageId, completed: false, imageGenerationAllowed });
        const requestHeaders: Record<string, string> = {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          ...getAuthAndClientHeaders(),
        };

        const response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
          method: 'POST',
          headers: requestHeaders,
          credentials: 'include',
          body: JSON.stringify({
            conversation_id: userMessageId,
            image_intent_id: userMessageId,
            image_generation_allowed: imageGenerationAllowed && intentStored,
            session_id: UUID_REGEX.test(sessionId) ? sessionId : undefined,
            history: guestHistory && guestHistory.length > 0 ? guestHistory : undefined,
            message: trimmed || 'Analyze attached file(s)',
            model: 'auto',
            tier: selectedTier,
            system_instruction: AVAILABLE_PERSONAS.find((p) => p.id === selectedPersona)?.systemInstruction || undefined,
            stream: true,
            web_search: webSearch,
            attachments: attachments && attachments.length > 0 ? attachments.map((a) => ({
              name: a.name,
              mime_type: a.mime_type,
              data: a.data,
            })) : undefined,
          }),
          signal: controller.signal,
        });

        if (response.status === 429) {
          throw new Error(
            'Jruva is receiving too many requests. Please wait a few seconds and try again.'
          );
        }

        if (!response.ok) {
          const errorJson = await response.json().catch(() => null);
          const detail =
            errorJson?.detail || `Server returned error ${response.status}`;
          if (response.status === 403 && typeof detail === 'string' && detail.includes('permission to append')) {
            const freshId = crypto.randomUUID();
            setSessionId(freshId);
            try { localStorage.setItem('chatbot_session_id', freshId); } catch {}
            throw new Error('Previous conversation expired. A fresh session was initialized — please send your message again.');
          }
          throw new Error(presentUserFacingError(typeof detail === 'string' ? detail : ''));
        }

        if (!response.body) {
          throw new Error('ReadableStream not supported by browser response.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let streamBuffer = '';
        let accumulatedTokens = '';
        let citations: SourceCitation[] = [];
        const streamStartTime = performance.now();

        const processEventBlock = (eventBlock: string) => {
          const lines = eventBlock.split('\n');
          let dataPayload = '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              dataPayload = line.slice(6).trim();
            }
          }

          if (!dataPayload) return;

          try {
            const event: ChatStreamEvent = JSON.parse(dataPayload);

            if (event.type === 'status' && event.content) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, statusText: presentStatusLabel(event.content || '') }
                    : msg
                )
              );
            } else if (event.type === 'token' && event.content) {
              const cleanChunk = event.content.replace(/<\/?(?:unstructured|structured)>/g, '');
              accumulatedTokens += cleanChunk;
              setError(null);
              const parsed = parseThoughtAndContent(accumulatedTokens);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? {
                        ...msg,
                        content: parsed.content,
                        thought: parsed.thought,
                        isThinkingActive: parsed.isThinkingActive,
                        statusText: undefined,
                        provider: event.actual_provider || event.provider || msg.provider,
                        model: event.actual_model || event.model || msg.model,
                        tier: event.tier || selectedTier,
                        cached: event.cached !== undefined ? event.cached : msg.cached,
                        failover: event.fallback_used !== undefined ? event.fallback_used : (event.failover !== undefined ? event.failover : msg.failover),
                        requested_model: event.requested_model || msg.requested_model,
                        requested_model_label: event.requested_model_label || msg.requested_model_label,
                        actual_model: event.actual_model || msg.actual_model,
                        actual_provider: event.actual_provider || msg.actual_provider,
                        fallback_used: event.fallback_used !== undefined ? event.fallback_used : msg.fallback_used,
                        fallback_reason: event.fallback_reason || msg.fallback_reason,
                      }
                    : msg
                )
              );
            } else if (event.type === 'citation' && event.sources) {
              citations = [...citations, ...event.sources];
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, sources: citations }
                    : msg
                )
              );
            } else if (event.type === 'finish') {
              if (event.finish_reason === 'stop') writeIntent(intentStorageKey, { id: userMessageId, completed: true, imageGenerationAllowed });
              const cleanFinish = event.finish_reason === 'stop';
              if (cleanFinish) {
                haptics.complete();
                setError(null);
              }
              const durationSec = Math.max(0.1, (performance.now() - streamStartTime) / 1000);
              const totalTokens = event.total_tokens || accumulatedTokens.split(/\s+/).filter(Boolean).length;
              const tokensPerSec = Math.round(totalTokens / durationSec);
              const parsed = parseThoughtAndContent(accumulatedTokens);

              setMessages((prev) =>
                prev.map((msg) => {
                  if (msg.id !== assistantMessageId) return msg;

                  const newVariation: MessageVariation = {
                    id: assistantMessageId,
                    content: parsed.content,
                    thought: parsed.thought,
                    createdAt: new Date(),
                    model: event.actual_model || event.model || msg.model,
                    provider: event.actual_provider || event.provider || msg.provider,
                    sources: citations,
                  };

                  const allVariations = msg.variations ? [...msg.variations, newVariation] : [newVariation];

                  return {
                    ...msg,
                    content: parsed.content,
                    thought: parsed.thought,
                    isThinkingActive: false,
                    status: cleanFinish ? 'completed' : 'error',
                    errorMessage: cleanFinish ? undefined : "Jruva couldn't finish this response. Please retry.",
                    statusText: undefined,
                    provider: event.actual_provider || event.provider || msg.provider,
                    model: event.actual_model || event.model || msg.model,
                    tier: event.tier || selectedTier,
                    cached: event.cached !== undefined ? event.cached : msg.cached,
                    failover: event.fallback_used !== undefined ? event.fallback_used : (event.failover !== undefined ? event.failover : msg.failover),
                    requested_model: event.requested_model || msg.requested_model,
                    requested_model_label: event.requested_model_label || msg.requested_model_label,
                    actual_model: event.actual_model || msg.actual_model,
                    actual_provider: event.actual_provider || msg.actual_provider,
                    fallback_used: event.fallback_used !== undefined ? event.fallback_used : msg.fallback_used,
                    fallback_reason: event.fallback_reason || msg.fallback_reason,
                    metrics: {
                      tokensPerSec,
                      totalTokens,
                      durationSec: parseFloat(durationSec.toFixed(1)),
                    },
                    variations: allVariations.length > 1 ? allVariations : undefined,
                    activeVariationIndex: allVariations.length > 1 ? allVariations.length - 1 : undefined,
                  };
                })
              );
            } else if (event.type === 'error') {
              throw new Error(event.content || (event as { error?: string }).error || 'Server reported generation error');
            }
          } catch (parseErr) {
            if (
              parseErr instanceof Error &&
              parseErr.message !== 'Unexpected end of JSON input'
            ) {
              throw parseErr;
            }
          }
        };

        while (true) {
          if (authEpoch !== authEpochRef.current) return;
          const { done, value } = await reader.read();
          if (authEpoch !== authEpochRef.current) return;
          if (done) {
            if (streamBuffer.trim()) {
              const flushed = streamBuffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
              const remainingEvents = flushed.split('\n\n');
              for (const eventBlock of remainingEvents) {
                if (eventBlock.trim()) {
                  processEventBlock(eventBlock);
                }
              }
              streamBuffer = '';
            }
            break;
          }

          streamBuffer += decoder.decode(value, { stream: true });
          const normalized = streamBuffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
          const events = normalized.split('\n\n');
          streamBuffer = events.pop() || '';

          for (const eventBlock of events) {
            processEventBlock(eventBlock);
          }
        }

        // Finalize message status
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId && msg.status === 'streaming'
              ? { ...msg, status: 'error' as const,
                  errorMessage: "Jruva couldn't finish this response. Please retry." }
              : msg
          )
        );

        // Refresh sessions list from server to pick up auto-generated titles
        fetchSessions();

      } catch (err) {
        if (!isMountedRef.current) return;
        if (err instanceof Error && err.name === 'AbortError') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId && msg.status !== 'completed'
                ? { ...msg, status: 'error',
                    errorMessage: "Jruva couldn't finish this response. Please retry." }
                : msg
            )
          );
        } else {
          const messageText =
            err instanceof Error ? err.message : 'Unknown communication error';
          const safeUserMessage = presentUserFacingError(messageText);
          setError(null);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    status: 'error',
                    errorMessage: safeUserMessage,
                    content: msg.content?.trim() ? msg.content : safeUserMessage,
                    statusText: undefined,
                  }
                : msg
            )
          );
        }
      } finally {
        // Only the send that still owns the stream may release it; a newer send
        // (after Stop) has replaced these refs and must keep its own state.
        if (inFlightRef.current === controller) inFlightRef.current = null;
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
          if (isMountedRef.current) setIsStreaming(false);
        }
      }
    },
    [selectedModel, selectedTier, sessionId, fetchSessions, selectedPersona]
  );

  // Edit & resend: replace messages from editIndex onward, resend with new content
  const editAndResend = useCallback(
    async (messageId: string, newContent: string, attachments?: AttachmentItem[], webSearch?: boolean) => {
      if (inFlightRef.current) return;

      // Find the message index
      const msgIndex = messages.findIndex((m) => m.id === messageId);
      if (msgIndex < 0) return;

      // Remove this message and everything after it
      messagesRef.current = messages.slice(0, msgIndex);
      setMessages((prev) => prev.slice(0, msgIndex));

      // messagesRef is already trimmed and the queued slice runs before
      // sendMessage's append, so no delay is needed; inFlightRef blocks repeats.
      await sendMessage(newContent, attachments, webSearch);
    },
    [messages, sendMessage]
  );

  // Regenerate: resend the last user message to get a fresh response with variation preservation
  const regenerateLastResponse = useCallback(async () => {
    if (inFlightRef.current) return;

    // Find the last user message
    const lastUserIdx = messages.findLastIndex((m) => m.role === 'user');
    if (lastUserIdx < 0) return;
    const lastUserMsg = messages[lastUserIdx];
    const prevAssistantMsg = messages[lastUserIdx + 1];

    // Capture existing variations
    const existingVariations: MessageVariation[] = prevAssistantMsg?.variations
      ? [...prevAssistantMsg.variations]
      : (prevAssistantMsg && prevAssistantMsg.content
          ? [{
              id: prevAssistantMsg.id,
              content: prevAssistantMsg.content,
              thought: prevAssistantMsg.thought,
              createdAt: prevAssistantMsg.createdAt,
              model: prevAssistantMsg.model,
              provider: prevAssistantMsg.provider,
              sources: prevAssistantMsg.sources,
            }]
          : []);

    // Remove user message and all subsequent messages
    messagesRef.current = messages.slice(0, lastUserIdx);
    setMessages((prev) => prev.slice(0, lastUserIdx));

    // Re-dispatch sendMessage; the queued slice runs before its append.
    await sendMessage(
      lastUserMsg.content,
      lastUserMsg.attachments,
      undefined,
      prevAssistantMsg?.status === 'error' && UUID_REGEX.test(lastUserMsg.id) ? lastUserMsg.id : undefined,
      lastUserMsg.imageGenerationAllowed !== false,
      existingVariations.length > 0 ? existingVariations : undefined
    );
  }, [messages, sendMessage]);

  // Switch between sibling response variations (< 1/3 >)
  const switchMessageVariation = useCallback((messageId: string, targetIndex: number) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id === messageId && msg.variations && msg.variations[targetIndex]) {
          const target = msg.variations[targetIndex];
          return {
            ...msg,
            content: target.content,
            thought: target.thought,
            model: target.model || msg.model,
            provider: target.provider || msg.provider,
            sources: target.sources || msg.sources,
            activeVariationIndex: targetIndex,
          };
        }
        return msg;
      })
    );
  }, []);

  // Rename session in local state and persist to backend
  const renameSession = useCallback(
    async (id: string, newTitle: string) => {
      const trimmed = newTitle.trim();
      if (!trimmed) return;

      let previousTitle: string | undefined;
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== id) return s;
          previousTitle = s.title;
          return { ...s, title: trimmed };
        })
      );

      const rollback = () => {
        if (previousTitle === undefined) return;
        const restore = previousTitle;
        setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title: restore } : s)));
        setError('Could not rename this conversation. Please try again.');
      };

      if (!UUID_REGEX.test(id)) return;
      try {
        const res = await fetch(`${getApiBaseUrl()}/api/v1/sessions/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json', ...getAuthAndClientHeaders() },
          credentials: 'include',
          body: JSON.stringify({ title: trimmed }),
        });
        const data = res.ok ? await res.json().catch(() => null) : null;
        if (!res.ok || data?.success === false) rollback();
      } catch {
        rollback();
      }
    },
    []
  );

  return {
    messages,
    isStreaming,
    sessionId,
    sessions,
    error,
    selectedModel,
    setSelectedModel: handleSelectModel,
    selectedTier,
    setSelectedTier: handleSelectTier,
    availableModels,
    selectedPersona,
    setSelectedPersona,
    availablePersonas: AVAILABLE_PERSONAS,
    sendMessage,
    stopGeneration,
    clearMessages,
    switchSession,
    deleteSession,
    renameSession,
    editAndResend,
    regenerateLastResponse,
    switchMessageVariation,
  };
}
