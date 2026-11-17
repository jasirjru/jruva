'use client';

import React, { useState, useMemo } from 'react';
import { ArtifactItem, useArtifact } from '@/context/ArtifactContext';
import { runInHardenedSandbox } from '@/utils/sandboxRunner';

interface ArtifactCanvasProps {
  artifact: ArtifactItem;
  onClose: () => void;
}

interface DiffLine {
  type: 'added' | 'removed' | 'neutral';
  oldLineNum?: number;
  newLineNum?: number;
  text: string;
}

// Compute line-by-line diff between two text strings
function computeLineDiff(oldText: string, newText: string): DiffLine[] {
  const oldLines = oldText.split('\n');
  const newLines = newText.split('\n');
  const diff: DiffLine[] = [];

  let o = 0;
  let n = 0;

  while (o < oldLines.length || n < newLines.length) {
    if (o < oldLines.length && n < newLines.length && oldLines[o] === newLines[n]) {
      diff.push({
        type: 'neutral',
        oldLineNum: o + 1,
        newLineNum: n + 1,
        text: oldLines[o],
      });
      o++;
      n++;
    } else {
      // Simple heuristic for insertions and deletions
      let matchFound = false;
      for (let lookAhead = 1; lookAhead <= 3; lookAhead++) {
        if (n + lookAhead < newLines.length && oldLines[o] === newLines[n + lookAhead]) {
          for (let k = 0; k < lookAhead; k++) {
            diff.push({
              type: 'added',
              newLineNum: n + 1,
              text: newLines[n],
            });
            n++;
          }
          matchFound = true;
          break;
        }
        if (o + lookAhead < oldLines.length && oldLines[o + lookAhead] === newLines[n]) {
          for (let k = 0; k < lookAhead; k++) {
            diff.push({
              type: 'removed',
              oldLineNum: o + 1,
              text: oldLines[o],
            });
            o++;
          }
          matchFound = true;
          break;
        }
      }

      if (!matchFound) {
        if (o < oldLines.length) {
          diff.push({
            type: 'removed',
            oldLineNum: o + 1,
            text: oldLines[o],
          });
          o++;
        }
        if (n < newLines.length) {
          diff.push({
            type: 'added',
            newLineNum: n + 1,
            text: newLines[n],
          });
          n++;
        }
      }
    }
  }

  return diff;
}

function buildSandboxedSvgDoc(rawSvg: string, background: 'dark' | 'light' | 'checker', zoom: number): string {
  const bgStyle =
    background === 'light'
      ? '#ffffff'
      : background === 'checker'
      ? '#18181b'
      : '#0d0d0f';
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: ${bgStyle};
      overflow: auto;
    }
    .svg-container {
      display: flex;
      align-items: center;
      justify-content: center;
      transform: scale(${zoom / 100});
      transform-origin: center center;
      transition: transform 0.15s ease;
    }
    svg {
      max-width: 90vw;
      max-height: 85vh;
    }
  </style>
</head>
<body>
  <div class="svg-container">
    ${rawSvg}
  </div>
</body>
</html>`;
}

export const ArtifactCanvas: React.FC<ArtifactCanvasProps> = ({ artifact, onClose }) => {
  const { setVersion, restoreVersion } = useArtifact();
  const [activeTab, setActiveTab] = useState<'preview' | 'code' | 'diff'>('preview');
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [reloadKey, setReloadKey] = useState<number>(0);
  const [svgBackground, setSvgBackground] = useState<'dark' | 'light' | 'checker'>('dark');
  const [svgZoom, setSvgZoom] = useState<number>(100);

  // Terminal & Code Execution State
  const [isTerminalOpen, setIsTerminalOpen] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [terminalOutput, setTerminalOutput] = useState<Array<{ text: string; isError?: boolean }>>([]);
  const [executionDuration, setExecutionDuration] = useState<number | null>(null);

  const versions = artifact.versions || [{ version: 1, content: artifact.content, timestamp: new Date() }];
  const currentVerIdx = artifact.currentVersionIndex !== undefined ? artifact.currentVersionIndex : versions.length - 1;
  const currentVerNum = versions[currentVerIdx]?.version || 1;

  // Previous version content for diffing
  const previousContent = useMemo(() => {
    if (versions.length > 1) {
      if (currentVerIdx > 0) {
        return versions[currentVerIdx - 1].content;
      }
      return versions[0].content;
    }
    return '';
  }, [versions, currentVerIdx]);

  const diffLines = useMemo(() => {
    if (!previousContent) return [];
    return computeLineDiff(previousContent, artifact.content);
  }, [previousContent, artifact.content]);

  // Format HTML content with fallback boilerplate if needed
  const formattedHtml = useMemo(() => {
    if (artifact.type !== 'html') return artifact.content;

    const trimmed = artifact.content.trim();
    const cspMeta = `<meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'none';">`;

    if (trimmed.toLowerCase().startsWith('<!doctype') || trimmed.toLowerCase().startsWith('<html')) {
      if (!trimmed.toLowerCase().includes('content-security-policy')) {
        return trimmed.replace(/<head[^>]*>/i, `$& \n  ${cspMeta}`);
      }
      return trimmed;
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  ${cspMeta}
  <title>${artifact.title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      margin: 0;
      padding: 16px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      color: #1e293b;
      background-color: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
      body {
        color: #f8fafc;
        background-color: #0f172a;
      }
    }
  </style>
</head>
<body>
  ${trimmed}
</body>
</html>`;
  }, [artifact.content, artifact.type, artifact.title]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(artifact.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  const handleDownload = () => {
    try {
      const ext =
        artifact.type === 'svg'
          ? 'svg'
          : artifact.type === 'html'
          ? 'html'
          : artifact.language || 'txt';
      const filename = `${artifact.title.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase()}_v${currentVerNum}.${ext}`;
      const blob = new Blob([artifact.content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {}
  };

  const handleReload = () => {
    setReloadKey((prev) => prev + 1);
  };

  // Run Code in Hardened Client-Side Web Worker Sandbox
  const handleRunCode = async () => {
    setIsRunning(true);
    setIsTerminalOpen(true);
    setTerminalOutput([{ text: `[Executing ${artifact.language || 'code'} in isolated Web Worker sandbox...]` }]);

    try {
      const isPython = artifact.language?.toLowerCase() === 'python';
      const result = await runInHardenedSandbox(artifact.content, isPython, 2500);

      setExecutionDuration(result.durationMs / 1000);
      setTerminalOutput(result.logs.length > 0 ? result.logs : [{ text: 'Process finished with exit code 0.' }]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setTerminalOutput([{ text: `Execution Failed: ${message}`, isError: true }]);
    } finally {
      setIsRunning(false);
    }
  };

  const isExecutable = artifact.language?.toLowerCase() === 'python' || artifact.language?.toLowerCase() === 'javascript' || artifact.type === 'code';

  return (
    <aside className={`artifact-canvas-drawer ${isFullscreen ? 'fullscreen' : 'split-screen'}`}>
      {/* Canvas Top Bar */}
      <div className="artifact-canvas-header">
        <div className="artifact-header-left">
          <div className="artifact-icon-box">
            <span className="artifact-sparkle-icon">✦</span>
          </div>
          <div className="artifact-title-box">
            <span className="artifact-canvas-title" title={artifact.title}>
              {artifact.title}
            </span>
            <span className="artifact-type-badge">{artifact.language?.toUpperCase() || artifact.type.toUpperCase()}</span>
          </div>
        </div>

        {/* Center: Segmented Preview / Code / Diff Toggle */}
        <div className="artifact-tabs-pill">
          <button
            type="button"
            className={`artifact-tab-btn ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => setActiveTab('preview')}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            <span>Preview</span>
          </button>
          <button
            type="button"
            className={`artifact-tab-btn ${activeTab === 'code' ? 'active' : ''}`}
            onClick={() => setActiveTab('code')}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
            <span>Code</span>
          </button>
          {versions.length > 1 && (
            <button
              type="button"
              className={`artifact-tab-btn ${activeTab === 'diff' ? 'active' : ''}`}
              onClick={() => setActiveTab('diff')}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="18" cy="18" r="3" />
                <circle cx="6" cy="6" r="3" />
                <path d="M13 6h3a2 2 0 0 1 2 2v7" />
                <line x1="6" y1="9" x2="6" y2="21" />
              </svg>
              <span>Diff</span>
            </button>
          )}
        </div>

        {/* Right Actions Strip */}
        <div className="artifact-header-right">
          {/* Run Code Button for Python / Code */}
          {isExecutable && (
            <button
              type="button"
              className={`artifact-action-btn run-btn ${isRunning ? 'running' : ''}`}
              onClick={handleRunCode}
              disabled={isRunning}
              title="Run Code in Sandbox"
              aria-label="Run code"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              <span>{isRunning ? 'Running...' : 'Run'}</span>
            </button>
          )}

          {activeTab === 'preview' && artifact.type === 'html' && (
            <button
              type="button"
              className="artifact-action-btn"
              onClick={handleReload}
              title="Restart / Reload Preview"
              aria-label="Reload preview"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10" />
                <polyline points="1 20 1 14 7 14" />
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
              </svg>
            </button>
          )}

          {activeTab === 'preview' && artifact.type === 'svg' && (
            <div className="artifact-svg-controls">
              <button
                type="button"
                className="artifact-action-btn"
                onClick={() => setSvgZoom((z) => Math.max(50, z - 25))}
                title="Zoom Out"
              >
                -
              </button>
              <span className="artifact-zoom-label">{svgZoom}%</span>
              <button
                type="button"
                className="artifact-action-btn"
                onClick={() => setSvgZoom((z) => Math.min(300, z + 25))}
                title="Zoom In"
              >
                +
              </button>
              <button
                type="button"
                className="artifact-action-btn"
                onClick={() =>
                  setSvgBackground((b) => (b === 'dark' ? 'light' : b === 'light' ? 'checker' : 'dark'))
                }
                title={`Toggle Canvas Background (Current: ${svgBackground})`}
              >
                🎨
              </button>
            </div>
          )}

          <button
            type="button"
            className="artifact-action-btn"
            onClick={handleCopy}
            title="Copy Code to Clipboard"
            aria-label="Copy code"
          >
            {copied ? (
              <span style={{ fontSize: '0.7rem', color: '#10b981' }}>Copied!</span>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="artifact-action-btn"
            onClick={handleDownload}
            title="Download Artifact File"
            aria-label="Download file"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>

          <button
            type="button"
            className="artifact-action-btn"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title={isFullscreen ? 'Exit Fullscreen' : 'Expand Fullscreen'}
            aria-label="Toggle fullscreen"
          >
            {isFullscreen ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="4 14 10 14 10 20" />
                <polyline points="20 10 14 10 14 4" />
                <line x1="14" y1="10" x2="21" y2="3" />
                <line x1="3" y1="21" x2="10" y2="14" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 3 21 3 21 9" />
                <polyline points="9 21 3 21 3 15" />
                <line x1="21" y1="3" x2="14" y2="10" />
                <line x1="3" y1="21" x2="10" y2="14" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="artifact-action-btn close"
            onClick={onClose}
            title="Close Canvas"
            aria-label="Close canvas"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Multi-Turn Version Switcher Bar (Claude 3.5 Style) */}
      {versions.length > 1 && (
        <div className="artifact-version-strip">
          <span className="version-strip-label">Revisions:</span>
          <div className="version-pills-row">
            {versions.map((ver, idx) => (
              <button
                key={idx}
                type="button"
                className={`artifact-version-pill ${idx === currentVerIdx ? 'active' : ''}`}
                onClick={() => setVersion(idx)}
                title={`Version ${ver.version} (${new Date(ver.timestamp).toLocaleTimeString()})`}
              >
                v{ver.version}
              </button>
            ))}
          </div>

          {currentVerIdx < versions.length - 1 && (
            <button
              type="button"
              className="artifact-restore-btn"
              onClick={() => restoreVersion(currentVerIdx)}
              title="Restore this version as the latest"
            >
              ↺ Restore v{currentVerNum}
            </button>
          )}
        </div>
      )}

      {/* Main Canvas Body */}
      <div className="artifact-canvas-body">
        {activeTab === 'preview' ? (
          <div className="artifact-preview-stage">
            {artifact.type === 'html' ? (
              <iframe
                key={reloadKey}
                title={artifact.title}
                srcDoc={formattedHtml}
                className="artifact-iframe"
                sandbox="allow-scripts"
              />
            ) : artifact.type === 'svg' ? (
              <iframe
                key={`svg-${reloadKey}-${svgBackground}`}
                title={artifact.title}
                srcDoc={buildSandboxedSvgDoc(artifact.content, svgBackground, svgZoom)}
                className="artifact-iframe"
                sandbox=""
              />
            ) : (
              <div className="artifact-code-preview-stage">
                <div className="artifact-code-stage">
                  <div className="artifact-code-lines">
                    {artifact.content.split('\n').map((_, lineIdx) => (
                      <span key={lineIdx} className="artifact-line-num">
                        {lineIdx + 1}
                      </span>
                    ))}
                  </div>
                  <pre className="artifact-code-pre">
                    <code>{artifact.content}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>
        ) : activeTab === 'code' ? (
          <div className="artifact-code-stage">
            <div className="artifact-code-lines">
              {artifact.content.split('\n').map((_, lineIdx) => (
                <span key={lineIdx} className="artifact-line-num">
                  {lineIdx + 1}
                </span>
              ))}
            </div>
            <pre className="artifact-code-pre">
              <code>{artifact.content}</code>
            </pre>
          </div>
        ) : (
          /* Visual Diff Viewer */
          <div className="artifact-diff-stage">
            <div className="artifact-diff-header">
              <span>Comparing v{currentVerNum > 1 ? currentVerNum - 1 : 1} ➔ v{currentVerNum}</span>
              <span className="diff-stats">
                <span className="diff-stat-add">+{diffLines.filter(d => d.type === 'added').length} lines</span>
                <span className="diff-stat-sub">-{diffLines.filter(d => d.type === 'removed').length} lines</span>
              </span>
            </div>
            <div className="artifact-diff-content">
              {diffLines.map((line, idx) => (
                <div key={idx} className={`artifact-diff-line ${line.type}`}>
                  <span className="diff-line-marker">
                    {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
                  </span>
                  <span className="diff-line-number">
                    {line.type === 'removed' ? line.oldLineNum : line.newLineNum || ' '}
                  </span>
                  <span className="diff-line-text">{line.text || ' '}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Collapsible Sandbox Terminal Drawer */}
        {isTerminalOpen && (
          <div className="artifact-terminal-drawer">
            <div className="artifact-terminal-bar">
              <div className="terminal-title">
                <span className="terminal-dot green" />
                <span>Console Terminal {executionDuration !== null ? `(${executionDuration}s)` : ''}</span>
              </div>
              <div className="terminal-actions">
                <button
                  type="button"
                  className="terminal-clear-btn"
                  onClick={() => setTerminalOutput([])}
                  title="Clear Console"
                >
                  Clear
                </button>
                <button
                  type="button"
                  className="terminal-close-btn"
                  onClick={() => setIsTerminalOpen(false)}
                  title="Close Terminal"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="artifact-terminal-output">
              {terminalOutput.length === 0 ? (
                <div className="terminal-empty">No output generated. Click &quot;Run&quot; to execute.</div>
              ) : (
                terminalOutput.map((out, idx) => (
                  <div key={idx} className={`terminal-line ${out.isError ? 'error' : ''}`}>
                    {out.text}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
