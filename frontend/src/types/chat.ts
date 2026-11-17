/**
 * Chat and SSE Stream Types adhering to OpenAPI 3.1.0 Contract.
 */

export interface SourceCitation {
  id: string;
  title: string;
  url?: string | null;
  snippet?: string | null;
  file_id?: string | null;
  page?: number | null;
  score?: number | null;
  source_type?: 'document' | 'web' | 'domain' | string;
}

export interface UsageMetadata {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface ModelOption {
  id: string;
  name: string;
  provider: 'auto' | 'gemini' | 'groq' | string;
  badge: string;
  icon?: string;
  speed?: string;
  description: string;
  recommended?: boolean;
  available?: boolean;
  health?: 'healthy' | 'degraded' | 'rate_limited' | 'down' | 'misconfigured' | string;
  runtime?: 'cloud' | 'local' | string;
  actual_backend_model?: string;
  fallback_capable?: boolean;
  hidden?: boolean;
}

export interface PersonaOption {
  id: string;
  name: string;
  icon: string;
  badge: string;
  description: string;
  /** Short line shown under the name in the persona menu. */
  tagline?: string;
  systemInstruction?: string;
}

export interface AttachmentItem {
  name: string;
  mime_type: string;
  data: string; // Base64 string without data URL prefix
  size?: number;
  previewUrl?: string; // Optional local object URL for instant image previews
}

export type UserTier = 'B' | 'A' | 'S' | 'S++' | 'S+++';

export interface TierInfo {
  tier: UserTier;
  label: string;
  badge: string;
  tagline: string;
  description: string;
  routing: string;
  maxTokens: string;
  ragChunks: string;
  targetTtft: string;
  rateLimit: string;
  color: string;
}

export const USER_TIERS: Record<UserTier, TierInfo> = {
  'S+++': {
    tier: 'S+++',
    label: 'Tier S+++',
    badge: 'Frontier',
    tagline: 'Supreme Intelligence & Speed',
    description: 'Frontier reasoning with GPT-OSS 120B on Groq LPUs, Gemini 3.8 Flash backup, and 4k context.',
    routing: 'Primary: Groq 120B → Backup: Gemini 3.8 Flash',
    maxTokens: '4,096 tokens',
    ragChunks: '12 chunks',
    targetTtft: '< 100 ms',
    rateLimit: '300 req/min',
    color: 'rose',
  },
  'S++': {
    tier: 'S++',
    label: 'Tier S++',
    badge: 'Supreme',
    tagline: 'Zero-Lag, Max Performance',
    description: 'Ultra-fast LPU inference (300+ tok/s) with Qwen 27B and Gemini 3.8 Flash zero-lag failover.',
    routing: 'Primary: Groq Qwen 27B → Backup: Gemini 3.8 Flash',
    maxTokens: '4,096 tokens',
    ragChunks: '8 chunks',
    targetTtft: '< 120 ms',
    rateLimit: '240 req/min',
    color: 'mocha',
  },
  S: {
    tier: 'S',
    label: 'Tier S',
    badge: 'Frontier',
    tagline: 'High Throughput, Max Compute',
    description: 'High-throughput cloud reasoning and frontier priority compute.',
    routing: 'Primary: Groq Qwen 27B → Backup: Gemini Flash Lite',
    maxTokens: '4,096 tokens',
    ragChunks: '10 chunks',
    targetTtft: '< 180 ms',
    rateLimit: '120 req/min',
    color: 'amber',
  },
  A: {
    tier: 'A',
    label: 'Tier A',
    badge: 'Pro / Dev',
    tagline: 'Deep Reasoning, Long Context',
    description: 'Extended reasoning with Google Gemini 3.8 Flash and Groq failover.',
    routing: 'Primary: Gemini 3.8 Flash → Backup: Groq Qwen 27B',
    maxTokens: '4,096 tokens',
    ragChunks: '6 chunks',
    targetTtft: '< 500 ms',
    rateLimit: '60 req/min',
    color: 'indigo',
  },
  B: {
    tier: 'B',
    label: 'Tier B',
    badge: 'Standard',
    tagline: 'High Throughput, Free',
    description: 'Zero-cost inference biased toward Groq LPUs with Google Gemini 3.8 Flash failover.',
    routing: 'Primary: Groq Qwen 27B → Backup: Gemini 3.8 Flash',
    maxTokens: '2,048 tokens',
    ragChunks: '3 chunks',
    targetTtft: '< 300 ms',
    rateLimit: '15 req/min',
    color: 'emerald',
  },
};

export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  history?: HistoryTurn[];
  session_id?: string;
  conversation_id?: string;
  message: string;
  model?: string;
  tier?: UserTier;
  stream?: boolean;
  system_instruction?: string | null;
  temperature?: number;
  attachments?: AttachmentItem[];
  web_search?: boolean;
}

export interface ChatResponse {
  session_id: string;
  message_id: string;
  conversation_id?: string;
  role: 'assistant';
  content: string;
  provider?: string;
  model?: string;
  tier?: UserTier;
  cached?: boolean;
  failover?: boolean;
  requested_model?: string;
  requested_model_label?: string;
  actual_model?: string;
  actual_provider?: string;
  fallback_used?: boolean;
  fallback_reason?: string;
  sources: SourceCitation[];
  usage?: UsageMetadata;
  created_at: string;
}

export type StreamEventType = 'token' | 'citation' | 'finish' | 'error' | 'metadata' | 'status';

export interface ChatStreamEvent {
  type: StreamEventType;
  session_id?: string;
  conversation_id?: string;
  content?: string;
  provider?: string;
  model?: string;
  tier?: UserTier;
  cached?: boolean;
  failover?: boolean;
  requested_model?: string;
  requested_model_label?: string;
  actual_model?: string;
  actual_provider?: string;
  fallback_used?: boolean;
  fallback_reason?: string;
  sources?: SourceCitation[];
  finish_reason?: 'stop' | 'length' | 'safety' | 'error' | 'capacity';
  total_tokens?: number;
}

export interface MessageVariation {
  id: string;
  content: string;
  thought?: string;
  createdAt: Date;
  model?: string;
  provider?: string;
  sources?: SourceCitation[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  imageGenerationAllowed?: boolean;
  content: string;
  thought?: string;
  isThinkingActive?: boolean;
  attachments?: AttachmentItem[];
  provider?: string;
  model?: string;
  tier?: UserTier;
  cached?: boolean;
  failover?: boolean;
  requested_model?: string;
  requested_model_label?: string;
  actual_model?: string;
  actual_provider?: string;
  fallback_used?: boolean;
  fallback_reason?: string;
  sources?: SourceCitation[];
  createdAt: Date;
  status?: 'sending' | 'streaming' | 'completed' | 'error';
  statusText?: string; // Processing indicator text (e.g. "Searching the web...")
  errorMessage?: string;
  metrics?: {
    tokensPerSec?: number;
    totalTokens?: number;
    durationSec?: number;
  };
  variations?: MessageVariation[];
  activeVariationIndex?: number;
}
