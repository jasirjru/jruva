import { GraduationCap, Lightbulb, SlidersHorizontal, StickyNote, Target, type LucideIcon } from 'lucide-react';

export type FactCategory = 'academic' | 'goal' | 'preference' | 'interest' | 'general';

export interface FactCategoryConfig {
  id: FactCategory;
  label: string;
  plural: string;
  Icon: LucideIcon;
}

/** Categories the memory API accepts. "general" has no filter in the old card, so facts there were hidden. */
export const FACT_CATEGORIES: FactCategoryConfig[] = [
  { id: 'academic', label: 'Academic', plural: 'Academic', Icon: GraduationCap },
  { id: 'goal', label: 'Goal', plural: 'Goals', Icon: Target },
  { id: 'preference', label: 'Preference', plural: 'Preferences', Icon: SlidersHorizontal },
  { id: 'interest', label: 'Interest', plural: 'Interests', Icon: Lightbulb },
  { id: 'general', label: 'General', plural: 'General', Icon: StickyNote },
];

const FALLBACK = FACT_CATEGORIES[FACT_CATEGORIES.length - 1];

/** Unknown categories from the API are shown as General. */
export function categoryOf(raw: string): FactCategoryConfig {
  const id = raw.toLowerCase();
  return FACT_CATEGORIES.find((c) => c.id === id) ?? FALLBACK;
}

/** Server and network errors arrive without end punctuation; make them read as a sentence. */
export function asSentence(text: string | undefined, fallback: string): string {
  const clean = (text || fallback).trim();
  return /[.!?]$/.test(clean) ? clean : `${clean}.`;
}
