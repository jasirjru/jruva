'use client';

import React, { useState } from 'react';
import { Tooltip } from '@astryxdesign/core/Tooltip';
import { Brain, Trash2 } from 'lucide-react';
import { UserMemory } from '@/types/memory';
import { categoryOf, FACT_CATEGORIES, FactCategory } from './memoryCategories';

type Filter = 'all' | FactCategory;

type MemoryFactListProps = {
  memories: UserMemory[];
  onRequestDelete: (memory: UserMemory) => void;
};

/** Category filter plus the list of saved facts. */
export function MemoryFactList({ memories, onRequestDelete }: MemoryFactListProps) {
  const [filter, setFilter] = useState<Filter>('all');

  if (memories.length === 0) {
    return (
      <div className="pop-empty">
        <Brain className="pop-empty-icon" size={28} strokeWidth={1.5} aria-hidden="true" />
        <p className="pop-empty-title">Nothing saved yet</p>
        <p className="pop-empty-text">
          Facts Jruva learns as you chat show up here.
        </p>
      </div>
    );
  }

  const counts = new Map<Filter, number>([['all', memories.length]]);
  for (const m of memories) {
    const id = categoryOf(m.category).id;
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  const filters = [
    { id: 'all' as Filter, label: 'All' },
    ...FACT_CATEGORIES.filter((c) => counts.has(c.id)).map((c) => ({ id: c.id as Filter, label: c.plural })),
  ];
  // A filter can empty out after a delete; fall back to All instead of an empty list.
  const activeFilter = counts.has(filter) ? filter : 'all';
  const visible =
    activeFilter === 'all' ? memories : memories.filter((m) => categoryOf(m.category).id === activeFilter);

  return (
    <>
      {filters.length > 2 && (
        <div className="pop-mem-filters" role="group" aria-label="Filter by category">
          {filters.map((f) => (
            <button
              key={f.id}
              type="button"
              className="pop-chip"
              aria-pressed={activeFilter === f.id}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
              <span className="pop-chip-count">{counts.get(f.id)}</span>
            </button>
          ))}
        </div>
      )}

      <ul className={`pop-mem-list${filters.length > 2 ? '' : ' pop-mem-list-spaced'}`} aria-label="Saved facts">
        {visible.map((mem) => {
          const cat = categoryOf(mem.category);
          return (
            <li key={mem.id} className="pop-mem-item">
              <cat.Icon className="pop-mem-icon" size={16} strokeWidth={1.75} aria-hidden="true" />
              <div className="pop-mem-main">
                <p className="pop-mem-fact">{mem.fact}</p>
                <p className="pop-mem-meta">
                  {cat.label}
                  {mem.source === 'auto' ? ' · Learned from a chat' : ''}
                </p>
              </div>
              <Tooltip content="Forget" placement="above">
                <button
                  type="button"
                  className="pop-icon-btn"
                  aria-label={`Forget: ${mem.fact}`}
                  onClick={() => onRequestDelete(mem)}
                >
                  <Trash2 size={16} strokeWidth={1.75} aria-hidden="true" />
                </button>
              </Tooltip>
            </li>
          );
        })}
      </ul>
    </>
  );
}
