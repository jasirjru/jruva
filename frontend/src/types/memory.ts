export type MemoryCategory = 'all' | 'academic' | 'goal' | 'preference' | 'interest' | 'general';

export interface UserMemory {
  id: string;
  userId: string;
  category: string;
  fact: string;
  source: 'auto' | 'manual';
  confidence: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface MemoryCategoryConfig {
  id: MemoryCategory;
  label: string;
  icon: string;
  color: string;
}

export const MEMORY_CATEGORIES: MemoryCategoryConfig[] = [
  { id: 'all', label: 'All Facts', icon: '✦', color: 'var(--accent-primary)' },
  { id: 'academic', label: 'Academic', icon: '🎓', color: '#3B82F6' },
  { id: 'goal', label: 'Goals', icon: '🎯', color: '#10B981' },
  { id: 'preference', label: 'Preferences', icon: '⚙️', color: '#F59E0B' },
  { id: 'interest', label: 'Interests', icon: '💡', color: '#8B5CF6' },
];
