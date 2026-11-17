'use client';

import '@/styles/pages/popups.css';
import React, { useEffect, useId, useState } from 'react';
import { Dialog } from '@astryxdesign/core/Dialog';
import { AlertCircle, ArrowDown, ArrowUp, Loader2, Search } from 'lucide-react';
import { WorkspaceView } from '@/types/workspace';
import { buildActions, buildResults, PaletteItem, SearchResults } from './paletteItems';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectView: (view: WorkspaceView) => void;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onToggleTheme: () => void;
  theme: 'dark' | 'light';
}

type SearchStatus = 'idle' | 'loading' | 'error' | 'done';

const SEARCH_DEBOUNCE_MS = 200;
const RESULT_LIMIT = 5;
const ICON = { size: 16, strokeWidth: 1.75 } as const;

/** Ctrl+K palette: quick actions plus search across chats, projects, files and images. */
export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose, ...handlers }) => (
  <Dialog
    isOpen={isOpen}
    onOpenChange={(open) => {
      if (!open) onClose();
    }}
    width={600}
    maxHeight="min(560px, 80dvh)"
    padding={0}
    position={{ top: 'min(14vh, 120px)' }}
    className="pop-dialog pop-pal"
    aria-label="Search and commands"
    data-jruva-skin=""
  >
    {/* Mounted only while open, so every open starts from an empty query. */}
    {isOpen ? <PaletteBody onClose={onClose} {...handlers} /> : null}
  </Dialog>
);

function PaletteBody({ onClose, ...handlers }: Omit<CommandPaletteProps, 'isOpen'>) {
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<SearchStatus>('idle');
  const [results, setResults] = useState<SearchResults | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [retryToken, setRetryToken] = useState(0);
  const listId = useId();

  const trimmed = query.trim();

  useEffect(() => {
    if (!trimmed) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const url = `/api/v1/search/global?q=${encodeURIComponent(trimmed)}&limit=${RESULT_LIMIT}`;
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Search returned ${res.status}`);
        const data = await res.json();
        setResults(data.results ?? null);
        setStatus('done');
      } catch (err) {
        if (controller.signal.aborted) return;
        console.warn('Global search failed:', err);
        setStatus('error');
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [trimmed, retryToken]);

  const actions = buildActions(query, handlers);
  const found = trimmed ? buildResults(results, handlers) : [];
  const items: PaletteItem[] = [...actions, ...found];
  const active = items.length ? Math.min(activeIndex, items.length - 1) : -1;
  const activeId = active >= 0 ? `${listId}-${items[active].id}` : undefined;

  // Keep the highlighted row visible while moving with the arrow keys.
  useEffect(() => {
    if (!activeId) return;
    document.getElementById(activeId)?.scrollIntoView({ block: 'nearest' });
  }, [activeId]);

  const runItem = (item: PaletteItem) => {
    item.run();
    onClose();
  };

  const handleQueryChange = (value: string) => {
    setQuery(value);
    setActiveIndex(0);
    if (value.trim()) setStatus('loading');
    else {
      setStatus('idle');
      setResults(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!items.length) return;
    const last = items.length - 1;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const step = e.key === 'ArrowDown' ? 1 : -1;
      setActiveIndex(active + step > last ? 0 : active + step < 0 ? last : active + step);
    } else if (e.key === 'Enter' && active >= 0) {
      e.preventDefault();
      runItem(items[active]);
    }
  };

  const isSearching = trimmed && status === 'loading';
  const noMatches = trimmed && status === 'done' && items.length === 0;

  return (
    <div className="pop-shell">
      <div className="pop-pal-search">
        <Search {...ICON} size={18} aria-hidden="true" />
        <input
          className="pop-pal-input"
          type="text"
          role="combobox"
          aria-expanded="true"
          aria-controls={listId}
          aria-activedescendant={activeId}
          aria-autocomplete="list"
          aria-label="Search chats, projects, files and images"
          placeholder="Search or jump to…"
          autoComplete="off"
          spellCheck={false}
          data-autofocus=""
          value={query}
          onChange={(e) => handleQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        {isSearching && <Loader2 {...ICON} className="pop-spin" aria-label="Searching" />}
      </div>

      <div className="pop-pal-list" id={listId} role="listbox" aria-label="Results">
        {renderGroups(items, active, listId, setActiveIndex, runItem)}
      </div>

      {status === 'error' && trimmed && (
        <div className="pop-error" role="alert">
          <AlertCircle size={15} strokeWidth={1.75} aria-hidden="true" />
          <span>
            Search isn&apos;t reachable right now, so only actions are shown.{' '}
            <button type="button" className="pop-text-btn" onClick={() => setRetryToken((n) => n + 1)}>
              Try again
            </button>
          </span>
        </div>
      )}
      {noMatches && <p className="pop-pal-empty">Nothing matches &ldquo;{trimmed}&rdquo;.</p>}
      <p className="pop-sr-only" aria-live="polite">
        {isSearching ? 'Searching' : trimmed && status === 'done' ? `${items.length} results` : ''}
      </p>

      <div className="pop-foot pop-pal-foot" aria-hidden="true">
        <span>
          <kbd className="pop-kbd"><ArrowUp size={12} strokeWidth={1.75} /></kbd>
          <kbd className="pop-kbd"><ArrowDown size={12} strokeWidth={1.75} /></kbd> Move
        </span>
        <span><kbd className="pop-kbd">Enter</kbd> Open</span>
        <span><kbd className="pop-kbd">Esc</kbd> Close</span>
      </div>
    </div>
  );
}

function renderGroups(
  items: PaletteItem[],
  active: number,
  listId: string,
  setActive: (index: number) => void,
  run: (item: PaletteItem) => void,
) {
  const groups: { name: string; rows: { item: PaletteItem; index: number }[] }[] = [];
  items.forEach((item, index) => {
    const group = groups.find((g) => g.name === item.group);
    if (group) group.rows.push({ item, index });
    else groups.push({ name: item.group, rows: [{ item, index }] });
  });

  return groups.map((group) => (
    <div key={group.name} role="group" aria-labelledby={`${listId}-g-${group.name}`}>
      <div className="pop-pal-group-label" id={`${listId}-g-${group.name}`} role="presentation">
        {group.name}
      </div>
      {group.rows.map(({ item, index }) => (
        <div
          key={item.id}
          id={`${listId}-${item.id}`}
          role="option"
          aria-selected={index === active}
          className="pop-pal-item"
          onMouseMove={() => {
            if (index !== active) setActive(index);
          }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => run(item)}
        >
          <span className="pop-pal-item-icon" aria-hidden="true">
            <item.Icon {...ICON} />
          </span>
          <span className="pop-pal-item-text">
            <span className="pop-pal-item-label">{item.label}</span>
            {item.sub && <span className="pop-pal-item-sub">{item.sub}</span>}
          </span>
        </div>
      ))}
    </div>
  ));
}
