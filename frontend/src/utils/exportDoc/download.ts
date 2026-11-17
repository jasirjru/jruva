import { exportFilename, stripHiddenContent } from './markdownBlocks';

/** Triggers a browser download for a generated file. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.rel = 'noopener';
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Give the browser time to start the download before releasing the URL.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const EXPORT_NOTICE =
  'AI-generated answer from Jruva. AI answers can contain mistakes. Check important facts before relying on them.';

export function formatExportDate(date: Date): string {
  return new Intl.DateTimeFormat('en-GB', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

/** Downloads the answer as written (Markdown), minus hidden reasoning. */
export function exportAnswerAsMarkdown(markdown: string, exportedAt: Date = new Date()): void {
  const body = `${stripHiddenContent(markdown)}\n\n---\n_${EXPORT_NOTICE}_\n`;
  saveBlob(new Blob([body], { type: 'text/markdown;charset=utf-8' }), exportFilename('md', exportedAt));
}
