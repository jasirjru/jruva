/**
 * Converts an assistant answer (Markdown) into a small, format-neutral block
 * model that the PDF and DOCX exporters both render.
 *
 * Pure and dependency-free on purpose: it runs in the browser and in
 * `node --test` (type stripping), so it must not import anything.
 */

export interface InlineRun {
  text: string;
  bold?: boolean;
  italic?: boolean;
  code?: boolean;
}

export interface ListItem {
  level: number;
  runs: InlineRun[];
}

export type DocBlock =
  | { type: 'heading'; level: number; runs: InlineRun[] }
  | { type: 'paragraph'; runs: InlineRun[] }
  | { type: 'list'; ordered: boolean; start: number; items: ListItem[] }
  | { type: 'code'; language: string; text: string }
  | { type: 'quote'; runs: InlineRun[] }
  | { type: 'table'; header: InlineRun[][]; rows: InlineRun[][][] }
  | { type: 'rule' };

const FENCE_RE = /^\s*(```|~~~)\s*([\w+#.-]*)/;
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE_RE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_SEPARATOR_RE = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;
const INDENT_PER_LEVEL = 2;
const MAX_LIST_LEVEL = 5;

/** Removes reasoning blocks the chat UI also hides. */
export function stripHiddenContent(markdown: string): string {
  let clean = (markdown || '').replace(/<think>[\s\S]*?<\/think>/gi, '');
  const openIdx = clean.indexOf('<think>');
  if (openIdx !== -1) clean = clean.slice(0, openIdx);
  return clean.replace(/\r\n?/g, '\n').trim();
}

/**
 * Parses `**bold**`, `*italic*`, `` `code` ``, links and images into runs.
 * Underscore emphasis is left as text so identifiers like `__init__` or
 * `snake_case_name` survive intact.
 */
export function parseInline(text: string): InlineRun[] {
  const runs: InlineRun[] = [];
  const pattern =
    /(`+)([\s\S]*?)\1|!\[([^\]]*)\]\(([^)\s]*)[^)]*\)|\[([^\]]+)\]\(([^)\s]*)[^)]*\)|\*\*([\s\S]+?)\*\*|\*([^\s*][^*]*?)\*|~~([\s\S]+?)~~/g;
  let last = 0;
  let match: RegExpExecArray | null;

  const pushText = (value: string, style: Omit<InlineRun, 'text'> = {}) => {
    if (!value) return;
    runs.push({ text: value, ...style });
  };

  while ((match = pattern.exec(text)) !== null) {
    pushText(text.slice(last, match.index));
    if (match[1]) {
      pushText(match[2].trim() || match[2], { code: true });
    } else if (match[4] !== undefined && match[0].startsWith('!')) {
      pushText(match[3] ? `[Image: ${match[3]}]` : '[Image]', { italic: true });
    } else if (match[5] !== undefined) {
      const label = match[5];
      const url = match[6];
      pushText(label);
      if (url && url !== label && /^https?:\/\//i.test(url)) pushText(` (${url})`);
    } else if (match[7] !== undefined) {
      for (const inner of parseInline(match[7])) runs.push({ ...inner, bold: true });
    } else if (match[8] !== undefined) {
      for (const inner of parseInline(match[8])) runs.push({ ...inner, italic: true });
    } else if (match[9] !== undefined) {
      pushText(match[9]);
    }
    last = pattern.lastIndex;
  }
  pushText(text.slice(last));
  return mergeRuns(runs);
}

function sameStyle(a: InlineRun, b: InlineRun): boolean {
  return !!a.bold === !!b.bold && !!a.italic === !!b.italic && !!a.code === !!b.code;
}

function mergeRuns(runs: InlineRun[]): InlineRun[] {
  const merged: InlineRun[] = [];
  for (const run of runs) {
    const prev = merged[merged.length - 1];
    if (prev && sameStyle(prev, run)) {
      merged[merged.length - 1] = { ...prev, text: prev.text + run.text };
    } else {
      merged.push(run);
    }
  }
  return merged;
}

function splitTableRow(line: string): string[] {
  let trimmed = line.trim();
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
  if (trimmed.endsWith('|') && !trimmed.endsWith('\\|')) trimmed = trimmed.slice(0, -1);
  const cells: string[] = [];
  let current = '';
  for (let k = 0; k < trimmed.length; k += 1) {
    const char = trimmed[k];
    if (char === '\\' && trimmed[k + 1] === '|') {
      current += '|';
      k += 1;
    } else if (char === '|') {
      cells.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
}

function isTableStart(lines: string[], i: number): boolean {
  const next = lines[i + 1];
  return lines[i].includes('|') && next !== undefined && next.includes('-') && TABLE_SEPARATOR_RE.test(next);
}

function isBlockStart(lines: string[], i: number): boolean {
  const line = lines[i];
  return (
    FENCE_RE.test(line) ||
    HEADING_RE.test(line) ||
    RULE_RE.test(line) ||
    LIST_RE.test(line) ||
    /^\s*>/.test(line) ||
    isTableStart(lines, i)
  );
}

interface ParseResult<T> {
  block: T;
  next: number;
}

function parseCode(lines: string[], i: number): ParseResult<DocBlock> {
  const open = FENCE_RE.exec(lines[i]);
  const fence = open ? open[1] : '```';
  const language = open ? open[2] : '';
  const body: string[] = [];
  let j = i + 1;
  while (j < lines.length && !lines[j].trim().startsWith(fence)) {
    body.push(lines[j]);
    j += 1;
  }
  return { block: { type: 'code', language, text: body.join('\n') }, next: j + 1 };
}

function parseTable(lines: string[], i: number): ParseResult<DocBlock> {
  const header = splitTableRow(lines[i]).map(parseInline);
  const rows: InlineRun[][][] = [];
  let j = i + 2;
  while (j < lines.length && lines[j].trim() !== '' && lines[j].includes('|')) {
    const cells = splitTableRow(lines[j]);
    const padded = header.map((_, idx) => parseInline(cells[idx] ?? ''));
    rows.push(padded);
    j += 1;
  }
  return { block: { type: 'table', header, rows }, next: j };
}

function parseList(lines: string[], i: number): ParseResult<DocBlock> {
  const first = LIST_RE.exec(lines[i]);
  const ordered = !!first && /\d/.test(first[2]);
  const start = ordered && first ? parseInt(first[2], 10) : 1;
  const baseIndent = first ? first[1].length : 0;
  const items: ListItem[] = [];
  const texts: string[] = [];
  let j = i;

  while (j < lines.length) {
    const line = lines[j];
    const m = LIST_RE.exec(line);
    if (m) {
      const isOrdered = /\d/.test(m[2]);
      const indent = m[1].length;
      if (indent <= baseIndent && isOrdered !== ordered && items.length > 0) break;
      const level = Math.min(MAX_LIST_LEVEL, Math.max(0, Math.floor((indent - baseIndent) / INDENT_PER_LEVEL)));
      items.push({ level, runs: [] });
      texts.push(m[3]);
      j += 1;
      continue;
    }
    if (line.trim() === '') {
      const following = lines[j + 1];
      if (following !== undefined && LIST_RE.test(following)) {
        j += 1;
        continue;
      }
      break;
    }
    if (/^\s+\S/.test(line) && texts.length > 0 && !isBlockStart(lines, j)) {
      texts[texts.length - 1] += ` ${line.trim()}`;
      j += 1;
      continue;
    }
    break;
  }

  const parsedItems = items.map((item, idx) => ({ ...item, runs: parseInline(texts[idx]) }));
  return { block: { type: 'list', ordered, start, items: parsedItems }, next: j };
}

function parseQuote(lines: string[], i: number): ParseResult<DocBlock> {
  const parts: string[] = [];
  let j = i;
  while (j < lines.length && /^\s*>/.test(lines[j])) {
    parts.push(lines[j].replace(/^\s*>\s?/, '').trim());
    j += 1;
  }
  return { block: { type: 'quote', runs: parseInline(parts.filter(Boolean).join(' ')) }, next: j };
}

function parseParagraph(lines: string[], i: number): ParseResult<DocBlock> {
  const parts: string[] = [lines[i].trim()];
  let j = i + 1;
  while (j < lines.length && lines[j].trim() !== '' && !isBlockStart(lines, j)) {
    parts.push(lines[j].trim());
    j += 1;
  }
  return { block: { type: 'paragraph', runs: parseInline(parts.join(' ')) }, next: j };
}

/** Converts a Markdown answer into export blocks. */
export function markdownToBlocks(markdown: string): DocBlock[] {
  const lines = stripHiddenContent(markdown).split('\n');
  const blocks: DocBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    let result: ParseResult<DocBlock>;
    const heading = HEADING_RE.exec(line);
    if (FENCE_RE.test(line)) {
      result = parseCode(lines, i);
    } else if (heading) {
      result = { block: { type: 'heading', level: heading[1].length, runs: parseInline(heading[2]) }, next: i + 1 };
    } else if (RULE_RE.test(line)) {
      result = { block: { type: 'rule' }, next: i + 1 };
    } else if (isTableStart(lines, i)) {
      result = parseTable(lines, i);
    } else if (LIST_RE.test(line)) {
      result = parseList(lines, i);
    } else if (/^\s*>/.test(line)) {
      result = parseQuote(lines, i);
    } else {
      result = parseParagraph(lines, i);
    }
    blocks.push(result.block);
    i = result.next;
  }
  return blocks;
}

export interface ScriptRun {
  text: string;
  malayalam: boolean;
}

const MALAYALAM_CHAR_RE = /[ഀ-ൿ]/;
const JOINER_RE = /[‌‍]/;

/**
 * Splits text into Malayalam / non-Malayalam runs so the PDF can use a
 * Malayalam font only where needed. Zero-width joiners stay with the
 * Malayalam run they belong to; spaces stay with the surrounding run.
 */
export function splitScriptRuns(text: string): ScriptRun[] {
  const runs: ScriptRun[] = [];
  for (const char of Array.from(text)) {
    const prev = runs[runs.length - 1];
    let malayalam = MALAYALAM_CHAR_RE.test(char);
    if (!malayalam && prev && (JOINER_RE.test(char) || (char === ' ' && prev.malayalam))) {
      malayalam = prev.malayalam;
    }
    if (prev && prev.malayalam === malayalam) {
      prev.text += char;
    } else {
      runs.push({ text: char, malayalam });
    }
  }
  return runs;
}

/** Plain text of a run list, for alt text and tests. */
export function runsToText(runs: InlineRun[]): string {
  return runs.map((run) => run.text).join('');
}

function pad(value: number): string {
  return String(value).padStart(2, '0');
}

/** `jruva-answer-YYYYMMDD-HHMM.<ext>` in local time. */
export function exportFilename(extension: string, date: Date = new Date()): string {
  const stamp = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}`;
  return `jruva-answer-${stamp}.${extension}`;
}
