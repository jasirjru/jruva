'use client';

import React, { useId, useState } from 'react';
import { BookOpen, ChevronDown, FileText } from 'lucide-react';
import { SourceCitation } from '@/types/chat';
import { EvidenceInspectorDrawer } from './EvidenceInspectorDrawer';

interface SourcesTrayProps {
  sources?: SourceCitation[];
}

const SNIPPET_OVERLAP_CHARS = 24;

function extractDomain(url?: string | null): string | null {
  if (!url) return null;
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

function isWebUrl(url?: string | null): boolean {
  return Boolean(url && (url.startsWith('http://') || url.startsWith('https://')));
}

/** Drop leading emoji that some source documents put in their titles. */
function cleanTitle(title: string): string {
  return title.replace(/^[\p{Extended_Pictographic}\u{FE0F}\s]+/u, '').trim() || title;
}

/** Hide a snippet that only repeats the title (common for document chunks). */
function usefulSnippet(title: string, snippet?: string | null): string | null {
  if (!snippet) return null;
  const head = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().slice(0, SNIPPET_OVERLAP_CHARS);
  return head(snippet) === head(title) ? null : snippet;
}

export const SourcesTray: React.FC<SourcesTrayProps> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);
  const [selectedSource, setSelectedSource] = useState<SourceCitation | null>(null);
  const listId = useId();

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-tray-container">
      <button
        type="button"
        className="sources-tray-header"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        aria-controls={listId}
      >
        <span className="sources-tray-header-left">
          <BookOpen className="sources-globe-icon" size={14} strokeWidth={1.75} aria-hidden="true" />
          <span className="sources-title">
            {sources.length} {sources.length === 1 ? 'source' : 'sources'}
          </span>
        </span>
        <ChevronDown className={`sources-chevron ${isOpen ? 'open' : ''}`} size={14} strokeWidth={2} aria-hidden="true" />
      </button>

      {isOpen && (
        <ol id={listId} className="sources-grid" aria-label="Sources">
          {sources.map((source, index) => {
            const hasWeb = isWebUrl(source.url);
            const domain = extractDomain(source.url);
            const faviconUrl = hasWeb && domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : null;
            const origin = hasWeb && domain ? domain : 'Jruva library';
            const title = cleanTitle(source.title);
            const snippet = usefulSnippet(title, source.snippet);

            return (
              <li key={source.id || index}>
                <button
                  type="button"
                  className="source-card interactive-source-card"
                  onClick={() => setSelectedSource(source)}
                  aria-label={`Source ${index + 1}: ${title}. Open evidence`}
                >
                  <span className="source-card-top">
                    <span className="source-index-badge" aria-hidden="true">
                      {index + 1}
                    </span>
                    <span className="source-favicon-wrapper" aria-hidden="true">
                      {faviconUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element -- 16px remote favicon
                        <img
                          src={faviconUrl}
                          alt=""
                          width={14}
                          height={14}
                          className="source-favicon"
                          loading="lazy"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />
                      ) : (
                        <FileText className="source-doc-icon" size={13} strokeWidth={1.75} />
                      )}
                    </span>
                    <span className="source-domain">{origin}</span>
                    {source.page ? <span className="source-page-badge">p. {source.page}</span> : null}
                  </span>
                  <span className="source-card-title">{title}</span>
                  {snippet && <span className="source-card-snippet">{snippet}</span>}
                </button>
              </li>
            );
          })}
        </ol>
      )}

      {selectedSource && (
        <EvidenceInspectorDrawer source={selectedSource} onClose={() => setSelectedSource(null)} />
      )}
    </div>
  );
};
