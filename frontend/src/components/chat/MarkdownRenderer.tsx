'use client';

import React, { useState } from 'react';
import { useArtifact } from '@/context/ArtifactContext';
import { GeneratedImageCard } from './GeneratedImageCard';
import { isSameOriginPath } from '@/utils/safeImageUrl';

// The server refuses arbitrary code execution unless it is explicitly enabled, so only show Run when it is.
const SERVER_CODE_EXECUTION_ENABLED = process.env.NEXT_PUBLIC_ENABLE_CODE_EXECUTION === 'true';

interface MarkdownRendererProps {
  content: string;
  isStreaming?: boolean;
}

interface CodeBlockProps {
  language: string;
  filename?: string;
  code: string;
}

export function sanitizeDownloadFilename(name: string | undefined, defaultExt: string): string {
  if (!name) return `snippet_${Date.now().toString(36)}.${defaultExt}`;
  const base = name.replace(/^.*[\\\/]/, '').trim();
  const sanitized = base.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_');
  return sanitized || `snippet_${Date.now().toString(36)}.${defaultExt}`;
}

export function sanitizeUrl(urlStr: string): string {
  if (!urlStr) return '#';
  const trimmed = urlStr.trim().replace(/[\x00-\x1f\x7f]/g, '');
  // Browsers treat "//host" and "/\host" as protocol-relative links to another site.
  if (trimmed.startsWith('/')) return isSameOriginPath(trimmed) ? trimmed : '#';
  if (trimmed.startsWith('#')) return trimmed;
  if (trimmed.startsWith('mailto:')) return trimmed;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'mailto:') {
      return trimmed;
    }
  } catch {
    // Malformed or relative URL without valid protocol
  }
  return '#';
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, filename, code }) => {
  const [copied, setCopied] = useState(false);
  const [isRunningSandbox, setIsRunningSandbox] = useState(false);
  const [sandboxResult, setSandboxResult] = useState<any>(null);
  const [showSandbox, setShowSandbox] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [complexityResult, setComplexityResult] = useState<any>(null);
  const [showComplexity, setShowComplexity] = useState(false);

  let artifactCtx: any = null;
  try {
    // Safely invoke useArtifact within ArtifactProvider
    artifactCtx = useArtifact();
  } catch {}

  const trimmedCode = code.trim();
  const lowerCode = trimmedCode.toLowerCase();
  const langLower = (language || '').toLowerCase();
  const isPython = !langLower || langLower === 'python' || langLower === 'py';

  const isHtml =
    language === 'html' ||
    lowerCode.startsWith('<!doctype') ||
    lowerCode.startsWith('<html') ||
    (lowerCode.includes('</') && (lowerCode.includes('<div') || lowerCode.includes('<button') || lowerCode.includes('<canvas')));

  const isSvg =
    language === 'svg' ||
    lowerCode.startsWith('<svg') ||
    (lowerCode.includes('<svg') && lowerCode.includes('</svg>'));

  const isReact = language === 'jsx' || language === 'tsx' || language === 'react';
  const isRenderable = isHtml || isSvg || isReact;

  // Infer title from filename, title tag, heading, or language
  const artifactTitle = React.useMemo(() => {
    if (filename && filename.trim().length > 0) {
      return filename.trim();
    }
    if (isHtml) {
      const titleMatch = code.match(/<title>([^<]+)<\/title>/i);
      if (titleMatch) return titleMatch[1].trim();
      const h1Match = code.match(/<h1[^>]*>([^<]+)<\/h1>/i);
      if (h1Match) return h1Match[1].trim();
      return 'Interactive Web Application';
    }
    if (isSvg) {
      return 'Vector SVG Graphic';
    }
    if (isReact) {
      return 'React Component Preview';
    }
    return `${(language || 'code').toUpperCase()} Artifact`;
  }, [code, filename, isHtml, isSvg, isReact, language]);

  const handleOpenArtifact = () => {
    if (!artifactCtx) return;
    artifactCtx.openArtifact({
      id: `art-${Math.random().toString(36).substring(2, 9)}`,
      title: artifactTitle,
      type: isSvg ? 'svg' : isHtml ? 'html' : isReact ? 'react' : 'code',
      language: language || (isSvg ? 'svg' : 'html'),
      content: code,
    });
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  const [downloaded, setDownloaded] = useState(false);
  const handleDownload = () => {
    try {
      const extMap: Record<string, string> = {
        python: 'py', py: 'py',
        javascript: 'js', js: 'js',
        typescript: 'ts', ts: 'ts',
        tsx: 'tsx', jsx: 'jsx',
        html: 'html', css: 'css',
        json: 'json', sql: 'sql',
        markdown: 'md', md: 'md',
        sh: 'sh', bash: 'sh',
        rust: 'rs', rs: 'rs',
        go: 'go', cpp: 'cpp', c: 'c',
        yaml: 'yaml', yml: 'yaml',
        xml: 'xml', svg: 'svg',
      };
      const ext = extMap[langLower] || (filename && filename.includes('.') ? filename.split('.').pop() : 'txt') || 'txt';
      const cleanName = sanitizeDownloadFilename(filename, ext);
      const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = cleanName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 2000);
    } catch (e) {
      console.warn('Download code snippet failed:', e);
    }
  };

  const handleRunSandbox = async () => {
    setIsRunningSandbox(true);
    setShowSandbox(true);
    try {
      const res = await fetch('/api/v1/coding/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: trimmedCode, timeout_seconds: 1.0 }),
      });
      const data = await res.json();
      setSandboxResult(data);
    } catch (err: any) {
      setSandboxResult({ success: false, error: err?.message || 'Network error executing sandbox' });
    } finally {
      setIsRunningSandbox(false);
    }
  };

  const handleAnalyzeComplexity = async () => {
    setIsAnalyzing(true);
    setShowComplexity(true);
    try {
      const res = await fetch('/api/v1/coding/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: trimmedCode, language: language || 'python' }),
      });
      const data = await res.json();
      setComplexityResult(data);
    } catch (err: any) {
      setComplexityResult({ valid: false, summary: 'Failed to analyze complexity' });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="code-block-wrapper">
      {/* Claude-style Interactive Artifact Launch Card */}
      {isRenderable && artifactCtx && (
        <div className="artifact-banner-card" onClick={handleOpenArtifact}>
          <div className="artifact-banner-left">
            <span className="artifact-banner-sparkle">✦</span>
            <div className="artifact-banner-info">
              <span className="artifact-banner-title">{artifactTitle}</span>
              <span className="artifact-banner-type">
                {isSvg ? 'Vector SVG Graphic' : isHtml ? 'Interactive Web App' : 'UI Component'} • Click to preview in canvas
              </span>
            </div>
          </div>
          <button
            type="button"
            className="artifact-banner-btn"
            onClick={(e) => {
              e.stopPropagation();
              handleOpenArtifact();
            }}
            title="Open Interactive Canvas"
          >
            <span>Open Canvas</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="7" y1="17" x2="17" y2="7" />
              <polyline points="7 7 17 7 17 17" />
            </svg>
          </button>
        </div>
      )}

      <div className="code-block-container">
        <div className="code-block-header">
          <div className="macos-traffic-lights">
            <span className="traffic-dot close" />
            <span className="traffic-dot minimize" />
            <span className="traffic-dot maximize" />
          </div>
          <span className="code-lang-label">{language || 'code'}</span>

          <div className="code-header-actions">
            {isPython && SERVER_CODE_EXECUTION_ENABLED && (
              <button
                type="button"
                onClick={handleRunSandbox}
                className="code-action-btn sandbox-run-btn"
                title="Execute code in safe instruction-traced sandbox"
                disabled={isRunningSandbox}
              >
                {isRunningSandbox ? (
                  <span>Running...</span>
                ) : (
                  <>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    <span>Run</span>
                  </>
                )}
              </button>
            )}

            <button
              type="button"
              onClick={handleAnalyzeComplexity}
              className="code-action-btn complexity-btn"
              title="Analyze AST Big-O complexity & code smells"
              disabled={isAnalyzing}
            >
              {isAnalyzing ? <span>Analyzing...</span> : <span>Complexity</span>}
            </button>

            {isRenderable && artifactCtx && (
              <button
                type="button"
                onClick={handleOpenArtifact}
                className="code-preview-btn"
                title="Open in Sandboxed Canvas"
                aria-label="Preview in canvas"
              >
                <span>Canvas</span>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                  <polyline points="15 3 21 3 21 9" />
                  <line x1="10" y1="14" x2="21" y2="3" />
                </svg>
              </button>
            )}

            <button
              type="button"
              onClick={handleCopy}
              className="code-copy-btn"
              title="Copy to clipboard"
              aria-label="Copy code"
            >
              {copied ? (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  <span>Copy</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleDownload}
              className="code-copy-btn code-download-btn"
              title="Download code as file"
              aria-label="Download code snippet"
            >
              {downloaded ? (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Saved</span>
                </>
              ) : (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Save</span>
                </>
              )}
            </button>
          </div>
        </div>
        <pre className="code-pre">
          <code>{code}</code>
        </pre>

        {/* Interactive Safe Sandbox Terminal Drawer */}
        {showSandbox && sandboxResult && (
          <div className="sandbox-terminal-drawer">
            <div className="sandbox-terminal-header">
              <div className="sandbox-status-badge">
                <span className={`status-dot ${sandboxResult.success ? 'success' : 'error'}`} />
                <span>{sandboxResult.success ? 'Sandbox Passed' : 'Execution Failed'}</span>
                {sandboxResult.execution_time_ms != null && (
                  <span className="sandbox-latency-pill">{sandboxResult.execution_time_ms} ms</span>
                )}
              </div>
              <div className="sandbox-header-actions">
                <button type="button" onClick={handleRunSandbox} className="sandbox-rerun-btn" title="Re-run">
                  ↻ Re-run
                </button>
                <button type="button" onClick={() => setShowSandbox(false)} className="sandbox-close-btn" title="Close console">
                  ✕
                </button>
              </div>
            </div>
            <pre className="sandbox-terminal-pre">
              <code>
                {sandboxResult.stdout
                  ? sandboxResult.stdout
                  : sandboxResult.error
                  ? `❌ ${sandboxResult.error}`
                  : '[Process completed with 0 output]'}
              </code>
            </pre>
          </div>
        )}

        {/* Interactive AST Algorithmic Complexity Profile Drawer */}
        {showComplexity && complexityResult && (
          <div className="complexity-inspector-drawer">
            <div className="complexity-drawer-header">
              <span className="complexity-drawer-title">⚡ AST Algorithmic Complexity Profile</span>
              <button type="button" onClick={() => setShowComplexity(false)} className="sandbox-close-btn">
                ✕
              </button>
            </div>
            <div className="complexity-metrics-grid">
              <div className="metric-chip">
                <span className="chip-label">Time Complexity</span>
                <span className="chip-val highlight">{complexityResult.estimated_time_complexity || 'O(N)'}</span>
              </div>
              <div className="metric-chip">
                <span className="chip-label">Aux Space</span>
                <span className="chip-val">{complexityResult.estimated_space_complexity || 'O(1)'}</span>
              </div>
              <div className="metric-chip">
                <span className="chip-label">Max Nesting</span>
                <span className="chip-val">{complexityResult.max_nesting_depth ?? 0} loops</span>
              </div>
              <div className="metric-chip">
                <span className="chip-label">Cyclomatic</span>
                <span className="chip-val">{complexityResult.cyclomatic_complexity ?? 1}</span>
              </div>
            </div>
            {complexityResult.algorithmic_pattern && (
              <div className="complexity-pattern-badge">
                <strong>Pattern:</strong> {complexityResult.algorithmic_pattern.pattern} — {complexityResult.algorithmic_pattern.description}
              </div>
            )}
            {complexityResult.code_smells && complexityResult.code_smells.length > 0 && (
              <div className="complexity-smells-list">
                <span className="smells-header">⚠️ Optimization Opportunities:</span>
                {complexityResult.code_smells.map((s: any, idx: number) => (
                  <div key={idx} className="smell-item">
                    <code>Line {s.line}</code>: {s.message}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

function cleanMarkdownContent(content: string): string {
  if (!content) return '';
  // 1. Strip all closed <think>...</think> blocks completely
  let clean = content.replace(/<think>[\s\S]*?<\/think>/gi, '');
  // 2. If an unclosed <think> tag exists (e.g. streaming in progress), strip it and everything after it
  const openIdx = clean.indexOf('<think>');
  if (openIdx !== -1) {
    clean = clean.slice(0, openIdx);
  }
  return clean.trim();
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  isStreaming = false,
}) => {
  if (!content) return null;

  const cleanContent = cleanMarkdownContent(content);
  if (!cleanContent) return null;

  return (
    <div className="md-content">
      {renderMarkdownBlocks(cleanContent)}
    </div>
  );
};

function renderMarkdownBlocks(rawContent: string): React.ReactNode[] {
  // Normalize if model wrote 'html code block' without backticks
  let content = rawContent.replace(/html code block\s*(<(?:!DOCTYPE|html)[\s\S]*?<\/html>)\s*html code block/gi, '```html\n$1\n```');
  
  // If raw complete HTML tag exists outside backticks, wrap it in ```html ... ```
  if (!content.includes('```') && (content.includes('<html') || content.includes('<!DOCTYPE') || content.includes('<!doctype')) && content.includes('</html>')) {
    content = content.replace(/(<(?:!DOCTYPE|html)[\s\S]*?<\/html>)/gi, '```html\n$1\n```');
  }

  // Split by code blocks: ```lang filename.ext\n ... ```
  const codeBlockRegex = /```([a-zA-Z0-9_.-]*)(?:[ \t]+([^\n]+))?\n([\s\S]*?)```/g;
  const elements: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  let blockIdx = 0;
  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Text before the code block
    if (match.index > lastIndex) {
      const textChunk = content.slice(lastIndex, match.index);
      elements.push(<span key={`text-${blockIdx}`}>{renderFormattedText(textChunk)}</span>);
    }

    const language = match[1] || '';
    const filename = match[2] ? match[2].trim() : undefined;
    const code = match[3];
    elements.push(
      <CodeBlock
        key={`code-${blockIdx}`}
        language={language}
        filename={filename}
        code={code.trimEnd()}
      />
    );

    lastIndex = match.index + match[0].length;
    blockIdx++;
  }

  // Trailing text after final code block
  if (lastIndex < content.length) {
    const remainingText = content.slice(lastIndex);
    elements.push(<span key={`tail-${blockIdx}`}>{renderFormattedText(remainingText)}</span>);
  }

  return elements;
}

// ─── Table Parser ──────────────────────────────────────────────────────────

function parseTable(lines: string[], startIdx: number): { table: React.ReactNode; endIdx: number } | null {
  // Check if current line and next line form a valid markdown table header
  if (startIdx + 1 >= lines.length) return null;

  const headerLine = lines[startIdx].trim();
  const separatorLine = lines[startIdx + 1]?.trim() || '';

  // Must have pipes and separator must be dashes
  if (!headerLine.includes('|') || !separatorLine.match(/^[\s|:-]+$/)) return null;
  if (!separatorLine.includes('---')) return null;

  const parseCells = (line: string): string[] => {
    return line.split('|').map((c) => c.trim()).filter((c, i, arr) => {
      // Remove empty first/last cells from leading/trailing pipes
      if (i === 0 && c === '') return false;
      if (i === arr.length - 1 && c === '') return false;
      return true;
    });
  };

  const headers = parseCells(headerLine);
  if (headers.length === 0) return null;

  // Parse alignment from separator
  const separatorCells = parseCells(separatorLine);
  const alignments = separatorCells.map((cell) => {
    const left = cell.startsWith(':');
    const right = cell.endsWith(':');
    if (left && right) return 'center';
    if (right) return 'right';
    return 'left';
  });

  // Collect data rows
  const dataRows: string[][] = [];
  let endIdx = startIdx + 2;
  while (endIdx < lines.length) {
    const line = lines[endIdx].trim();
    if (!line.includes('|') || line === '') break;
    dataRows.push(parseCells(line));
    endIdx++;
  }

  const table = (
    <div className="md-table-wrap" key={`table-${startIdx}`}>
      <table className="md-table">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} style={{ textAlign: alignments[i] || 'left' }}>{formatInline(h)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, ri) => (
            <tr key={ri}>
              {headers.map((_, ci) => (
                <td key={ci} style={{ textAlign: alignments[ci] || 'left' }}>
                  {formatInline(row[ci] || '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return { table, endIdx };
}

// ─── Main Text Renderer ────────────────────────────────────────────────────

function renderFormattedText(text: string): React.ReactNode {
  const lines = text.split('\n');
  const renderedLines: React.ReactNode[] = [];

  let inList: 'ul' | 'ol' | null = null;
  let listItems: React.ReactNode[] = [];
  let listKey = 0;

  const flushList = () => {
    if (inList && listItems.length > 0) {
      if (inList === 'ul') {
        renderedLines.push(<ul key={`ul-${listKey++}`}>{listItems}</ul>);
      } else {
        renderedLines.push(<ol key={`ol-${listKey++}`}>{listItems}</ol>);
      }
      listItems = [];
      inList = null;
    }
  };

  let lineIdx = 0;
  while (lineIdx < lines.length) {
    const line = lines[lineIdx];
    const trimmed = line.trim();

    // Try to parse a markdown table starting at this line
    const tableResult = parseTable(lines, lineIdx);
    if (tableResult) {
      flushList();
      renderedLines.push(tableResult.table);
      lineIdx = tableResult.endIdx;
      continue;
    }

    // Standalone Markdown image: ![alt](url) or ![error:...](prompt)
    const imgMatch = trimmed.match(/^!\[([\s\S]*?)\]\(([\s\S]+)\)$/);
    if (imgMatch) {
      flushList();
      const rawAlt = imgMatch[1].trim();
      const rawSrc = imgMatch[2].trim();

      // Check if alt or src carries structured error directive
      let cardSrc = rawSrc;
      let cardAlt = rawAlt;

      if (rawAlt.startsWith('error:')) {
        cardSrc = rawAlt;
        cardAlt = rawSrc;
      } else if (!cardSrc || cardSrc === '#' || cardSrc === 'about:blank') {
        cardSrc = 'error:empty_url:Upstream generation returned an empty payload';
      }

      renderedLines.push(
        <GeneratedImageCard
          key={`img-${lineIdx}`}
          alt={cardAlt}
          src={cardSrc}
        />
      );
      lineIdx++;
      continue;
    }

    // Horizontal rule
    if (trimmed.match(/^(-{3,}|\*{3,}|_{3,})$/)) {
      flushList();
      renderedLines.push(<hr key={`hr-${lineIdx}`} className="md-hr" />);
      lineIdx++;
      continue;
    }

    // Headers
    if (trimmed.startsWith('### ')) {
      flushList();
      renderedLines.push(<h3 key={`h3-${lineIdx}`}>{formatInline(trimmed.slice(4))}</h3>);
      lineIdx++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      flushList();
      renderedLines.push(<h2 key={`h2-${lineIdx}`}>{formatInline(trimmed.slice(3))}</h2>);
      lineIdx++;
      continue;
    }
    if (trimmed.startsWith('# ')) {
      flushList();
      renderedLines.push(<h1 key={`h1-${lineIdx}`}>{formatInline(trimmed.slice(2))}</h1>);
      lineIdx++;
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      flushList();
      renderedLines.push(
        <blockquote key={`quote-${lineIdx}`}>{formatInline(trimmed.slice(2))}</blockquote>
      );
      lineIdx++;
      continue;
    }

    // Task list (checkbox items)
    const taskMatch = trimmed.match(/^[-*]\s+\[([ xX])\]\s+(.*)/);
    if (taskMatch) {
      if (inList !== 'ul') {
        flushList();
        inList = 'ul';
      }
      const checked = taskMatch[1] !== ' ';
      listItems.push(
        <li key={`task-${lineIdx}`} className="md-task-item">
          <input type="checkbox" checked={checked} readOnly className="md-task-checkbox" />
          <span className={checked ? 'md-task-done' : ''}>{formatInline(taskMatch[2])}</span>
        </li>
      );
      lineIdx++;
      continue;
    }

    // Unordered list
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (inList !== 'ul') {
        flushList();
        inList = 'ul';
      }
      listItems.push(<li key={`li-${lineIdx}`}>{formatInline(trimmed.slice(2))}</li>);
      lineIdx++;
      continue;
    }

    // Numbered list
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      if (inList !== 'ol') {
        flushList();
        inList = 'ol';
      }
      listItems.push(<li key={`li-${lineIdx}`}>{formatInline(numMatch[2])}</li>);
      lineIdx++;
      continue;
    }

    // Standard paragraph or empty line
    flushList();
    if (!trimmed) {
      lineIdx++;
      continue;
    }

    renderedLines.push(<p key={`p-${lineIdx}`}>{formatInline(line)}</p>);
    lineIdx++;
  }

  flushList();
  return <>{renderedLines}</>;
}

// ─── Inline Formatting ─────────────────────────────────────────────────────

/**
 * Citation chips sit inside the answer text; the matching source card lives in the
 * same message's SourcesTray. Clicking the chip clicks that card, which opens its evidence.
 */
function openCitedSource(chip: HTMLElement, citeNumber: number) {
  const message = chip.closest('.message-wrapper');
  const card = message?.querySelectorAll<HTMLButtonElement>('.sources-grid .source-card')[citeNumber - 1];
  if (!card) return;
  card.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  card.click();
}

function formatInline(str: string): React.ReactNode {
  // Suppress solitary '!' artifacts that might result from failed/interrupted streams
  if (str.trim() === '!') {
    return null;
  }

  // Parses: inline code `code`, bold **bold**, italic *italic*, strikethrough ~~text~~, images ![alt](url), links [text](url), citations [3]
  const parts: React.ReactNode[] = [];
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|~~[^~]+~~|!\[([^\]]*)\]\(([^)]*)\)|\[([^\]]+)\]\(([^)]+)\)|\[\d{1,3}\](?!\())/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIndex) {
      parts.push(str.slice(lastIndex, match.index));
    }

    const token = match[0];

    // Image: ![alt](url)
    if (token.startsWith('![')) {
      const imgAlt = match[2] || 'Generated Image';
      const imgSrc = match[3] || 'error:empty_url:Image generation produced an empty payload';
      parts.push(
        <GeneratedImageCard
          key={`inline-img-${i++}`}
          alt={imgAlt}
          src={imgSrc}
        />
      );
    }
    // Link: [text](url) - strictly validate URL protocol against javascript: and unsafe URI schemes
    else if (token.startsWith('[') && match[4] && match[5]) {
      const safeHref = sanitizeUrl(match[5]);
      parts.push(
        <a key={`link-${i++}`} href={safeHref} target="_blank" rel="noopener noreferrer" className="md-link">
          {match[4]}
        </a>
      );
    }
    // Citation: [3] -> numbered chip that opens that source's evidence
    else if (/^\[\d{1,3}\]$/.test(token)) {
      const citeNumber = Number(token.slice(1, -1));
      parts.push(
        <button
          key={`cite-${i++}`}
          type="button"
          className="md-cite"
          onClick={(e) => openCitedSource(e.currentTarget, citeNumber)}
          aria-label={`Open source ${citeNumber}`}
        >
          {citeNumber}
        </button>
      );
    }
    // Inline code: `code`
    else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code key={`code-${i++}`} className="md-inline-code">
          {token.slice(1, -1)}
        </code>
      );
    }
    // Bold: **text**
    else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={`bold-${i++}`}>{token.slice(2, -2)}</strong>);
    }
    // Strikethrough: ~~text~~
    else if (token.startsWith('~~') && token.endsWith('~~')) {
      parts.push(<del key={`del-${i++}`} className="md-strikethrough">{token.slice(2, -2)}</del>);
    }
    // Italic: *text*
    else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(<em key={`italic-${i++}`}>{token.slice(1, -1)}</em>);
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < str.length) {
    const trailing = str.slice(lastIndex);
    // Don't append a trailing standalone '!' if it was part of an incomplete image markdown
    if (trailing !== '!') {
      parts.push(trailing);
    }
  }

  return <>{parts}</>;
}
