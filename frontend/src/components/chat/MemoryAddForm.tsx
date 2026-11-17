'use client';

import React, { useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { Selector } from '@astryxdesign/core/Selector';
import { TextInput } from '@astryxdesign/core/TextInput';
import { AlertCircle } from 'lucide-react';
import { asSentence, FACT_CATEGORIES, FactCategory } from './memoryCategories';

const MAX_FACT_LENGTH = 200;
const COUNTER_FROM = 150;

/** Starting points for a first fact. Clicking one fills the field; nothing is saved until "Remember". */
const EXAMPLES: { fact: string; category: FactCategory }[] = [
  { fact: 'I am a second-year computer science student', category: 'academic' },
  { fact: 'I am preparing for campus placements', category: 'goal' },
  { fact: 'I prefer short answers with code examples', category: 'preference' },
];

const CATEGORY_OPTIONS = FACT_CATEGORIES.map(({ id, label, Icon }) => ({
  value: id,
  label,
  icon: <Icon size={15} strokeWidth={1.75} aria-hidden="true" />,
}));

type MemoryAddFormProps = {
  showExamples: boolean;
  onAdd: (fact: string, category: string) => Promise<{ success: boolean; error?: string }>;
};

/** Add one fact to memory, with the category it belongs to. */
export function MemoryAddForm({ showExamples, onAdd }: MemoryAddFormProps) {
  const [fact, setFact] = useState('');
  const [category, setCategory] = useState<FactCategory>('general');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const trimmed = fact.trim();

  const submit = async () => {
    if (!trimmed || isSaving) return;
    setIsSaving(true);
    setError(null);
    const res = await onAdd(trimmed.slice(0, MAX_FACT_LENGTH), category);
    setIsSaving(false);
    if (res.success) setFact('');
    else setError(asSentence(res.error, 'The server did not save it'));
  };

  return (
    <form
      className="pop-mem-form"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <div className="pop-mem-form-row">
        <div className="pop-field pop-field-grow">
          <TextInput
            label="Add a fact"
            size="lg"
            value={fact}
            onChange={(value) => setFact(value.slice(0, MAX_FACT_LENGTH))}
            placeholder="For example: I'm preparing for GATE"
            autoComplete="off"
            isDisabled={isSaving}
          />
        </div>
        <div className="pop-field">
          <Selector
            label="Category"
            size="lg"
            options={CATEGORY_OPTIONS}
            value={category}
            onChange={(value) => setCategory(value as FactCategory)}
            isDisabled={isSaving}
          />
        </div>
        <Button type="submit" label="Remember" variant="primary" size="lg" isLoading={isSaving} isDisabled={!trimmed} />
      </div>

      {fact.length >= COUNTER_FROM && (
        <p className="pop-hint" aria-live="polite">
          <span />
          <span>{fact.length} / {MAX_FACT_LENGTH}</span>
        </p>
      )}

      {showExamples && !trimmed && (
        <div className="pop-examples">
          <span className="pop-hint">Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.fact}
              type="button"
              className="pop-chip"
              onClick={() => {
                setFact(ex.fact);
                setCategory(ex.category);
              }}
            >
              {ex.fact}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="pop-error" role="alert">
          <AlertCircle size={15} strokeWidth={1.75} aria-hidden="true" />
          <span>Couldn&apos;t save that fact. {error} Check your connection and try again.</span>
        </p>
      )}
    </form>
  );
}
