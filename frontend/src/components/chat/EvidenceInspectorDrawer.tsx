'use client';

import React, { useEffect, useState } from 'react';
import { SourceCitation } from '@/types/chat';

interface EvidenceInspectorDrawerProps {
  source: SourceCitation | null;
  onClose: () => void;
}

interface PassageDetails {
  passage_id: string;
  file_id: string;
  doc_name: string;
  page?: number;
  section?: string;
  breadcrumb?: string;
  text: string;
  download_url?: string | null;
}

export const EvidenceInspectorDrawer: React.FC<EvidenceInspectorDrawerProps> = ({
  source,
  onClose,
}) => {
  const [details, setDetails] = useState<PassageDetails | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    if (!source) {
      setDetails(null);
      return;
    }

    // If source has a file_id, fetch full passage excerpt from authenticated endpoint
    const fileId = source.file_id;
    const passageId = source.id;

    if (fileId && passageId && !source.url?.startsWith('http')) {
      setLoading(true);
      fetch(`/api/v1/library/files/${encodeURIComponent(fileId)}/passages/${encodeURIComponent(passageId)}`, {
        headers: {
          Accept: 'application/json',
        },
      })
        .then((res) => {
          if (res.ok) return res.json();
          throw new Error('Failed to load passage');
        })
        .then((data: PassageDetails) => {
          setDetails(data);
        })
        .catch(() => {
          // Fallback to client snippet if fetch fails or user is unauthenticated
          setDetails({
            passage_id: passageId,
            file_id: fileId,
            doc_name: source.title,
            page: source.page || 1,
            text: source.snippet || 'No excerpt available.',
            download_url: fileId !== 'domain' ? `/api/v1/library/files/${encodeURIComponent(fileId)}/content` : null,
          });
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setDetails({
        passage_id: source.id,
        file_id: source.file_id || '',
        doc_name: source.title,
        page: source.page || undefined,
        text: source.snippet || 'No excerpt available.',
        download_url: source.url?.startsWith('http') ? source.url : null,
      });
    }
  }, [source]);

  if (!source) return null;

  const handleCopy = () => {
    const textToCopy = details?.text || source.snippet || '';
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const displayText = details?.text || source.snippet || 'No passage excerpt available.';
  const pageNumber = details?.page ?? source.page ?? 1;
  const scorePercent = source.score ? Math.round(source.score * 100) : null;

  return (
    <div className="evidence-drawer-backdrop" onClick={onClose} role="presentation">
      <div
        className="evidence-drawer-container"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Source passage"
      >
        <div className="evidence-drawer-header">
          <div className="evidence-drawer-header-left">
            <span className="evidence-shield-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </span>
            <div>
              <h3 className="evidence-drawer-title">Source passage</h3>
              <p className="evidence-drawer-subtitle">Verbatim passage verification & document grounding</p>
            </div>
          </div>
          <button
            type="button"
            className="evidence-drawer-close-btn"
            onClick={onClose}
            aria-label="Close evidence inspection"
          >
            ✕
          </button>
        </div>

        <div className="evidence-drawer-body">
          <div className="evidence-metadata-card">
            <div className="evidence-meta-row">
              <span className="evidence-meta-label">Document:</span>
              <span className="evidence-meta-value doc-name">{details?.doc_name || source.title}</span>
            </div>

            <div className="evidence-badges-row">
              <span className="evidence-badge page-badge">
                Page {pageNumber}
              </span>

              <span className="evidence-badge type-badge">
                {source.source_type === 'domain'
                  ? 'Institutional Knowledge'
                  : source.source_type === 'web'
                  ? 'Web Grounding'
                  : 'Library Document'}
              </span>

              {scorePercent !== null && (
                <span className="evidence-badge score-badge" title="Reranked semantic + lexical relevance score">
                  Relevance: {scorePercent}%
                </span>
              )}
            </div>

            {details?.section && (
              <div className="evidence-meta-row section-row">
                <span className="evidence-meta-label">Section:</span>
                <span className="evidence-meta-value">{details.section}</span>
              </div>
            )}
          </div>

          <div className="evidence-excerpt-section">
            <div className="evidence-excerpt-header">
              <span className="evidence-excerpt-title">Verbatim Grounded Passage:</span>
              <button
                type="button"
                className="evidence-copy-btn"
                onClick={handleCopy}
                title="Copy grounded excerpt"
              >
                {copied ? '✓ Copied' : 'Copy Excerpt'}
              </button>
            </div>

            {loading ? (
              <div className="evidence-loading-state">
                <div className="processing-spinner" />
                <span>Loading passage...</span>
              </div>
            ) : (
              <div className="evidence-excerpt-box">
                <blockquote className="evidence-quote">{displayText}</blockquote>
              </div>
            )}
          </div>

          <div className="evidence-calibration-note">
            <span className="evidence-note-icon">ℹ</span>
            <span>
              <strong>Check it:</strong> This is the passage search returned. It may not support every claim in the answer, so compare them yourself.
            </span>
          </div>
        </div>

        <div className="evidence-drawer-footer">
          {details?.download_url && (
            <a
              href={details.download_url}
              target="_blank"
              rel="noopener noreferrer"
              className="evidence-action-btn primary"
            >
              {source.url?.startsWith('http') ? 'Visit Web Source' : 'View / Download Original File'}
            </a>
          )}
          <button type="button" className="evidence-action-btn secondary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
