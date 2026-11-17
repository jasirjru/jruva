/**
 * Real PDF export of one assistant answer, generated in the browser with
 * pdfmake (MIT). Loaded only through `await import()` from the export menu.
 *
 * Fonts:
 * - Roboto (bundled with pdfmake) for Latin, Greek and Cyrillic text.
 * - Noto Sans Malayalam (SIL OFL 1.1, served from /public/fonts) for
 *   Malayalam runs. fontkit shapes conjuncts and vowel-sign reordering.
 *   Mark-positioning features are switched off because fontkit 2.0.4 crashes
 *   on this font's null GPOS anchors; Malayalam rarely stacks marks, so the
 *   visible effect is small.
 * - Courier (PDF standard font) for code that is plain Latin-1.
 * Scripts outside these fonts (emoji, CJK, Arabic, Devanagari...) render as
 * blank or missing glyphs. The Word export does not have this limit.
 */
import type { PdfMakeBrowser } from 'pdfmake/build/pdfmake';
import type { DocBlock, InlineRun, ListItem } from './markdownBlocks';
import { markdownToBlocks, splitScriptRuns, exportFilename } from './markdownBlocks';
import { EXPORT_NOTICE, formatExportDate, saveBlob } from './download';

const MALAYALAM_FONT = 'NotoSansMalayalam';
const MALAYALAM_FILES = {
  normal: 'noto-sans-malayalam-malayalam-400-normal.woff',
  bold: 'noto-sans-malayalam-malayalam-700-normal.woff',
};
const DISABLED_MARK_FEATURES = { mark: false, mkmk: false, abvm: false, blwm: false };
const LATIN1_RE = /^[\t\n\x20-\x7E\xA0-\xFF]*$/;
const HEADING_SIZES = [18, 15, 13, 12, 11, 11];
const BODY_SIZE = 10.5;
const CODE_SIZE = 9;
const CODE_BACKGROUND = '#f1f5f9';
const PAGE_CONTENT_WIDTH = 515;

type PdfNode = Record<string, unknown>;

let pdfMakePromise: Promise<PdfMakeBrowser> | null = null;

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function fetchFontBase64(file: string): Promise<string> {
  const response = await fetch(`/fonts/${file}`);
  if (!response.ok) throw new Error(`Could not load font ${file} (HTTP ${response.status})`);
  return arrayBufferToBase64(await response.arrayBuffer());
}

async function loadPdfMake(): Promise<PdfMakeBrowser> {
  const [pdfMakeModule, robotoModule, courierModule, malayalamNormal, malayalamBold] = await Promise.all([
    import('pdfmake/build/pdfmake'),
    import('pdfmake/build/fonts/Roboto'),
    import('pdfmake/build/standard-fonts/Courier'),
    fetchFontBase64(MALAYALAM_FILES.normal),
    fetchFontBase64(MALAYALAM_FILES.bold),
  ]);
  const pdfMake = pdfMakeModule.default;
  pdfMake.addFontContainer(robotoModule.default);
  pdfMake.addFontContainer(courierModule.default);
  pdfMake.addVirtualFileSystem({
    [MALAYALAM_FILES.normal]: malayalamNormal,
    [MALAYALAM_FILES.bold]: malayalamBold,
  });
  pdfMake.addFonts({
    [MALAYALAM_FONT]: {
      normal: MALAYALAM_FILES.normal,
      bold: MALAYALAM_FILES.bold,
      italics: MALAYALAM_FILES.normal,
      bolditalics: MALAYALAM_FILES.bold,
    },
  });
  // The document never references images or remote files.
  pdfMake.setUrlAccessPolicy(() => false);
  return pdfMake;
}

function getPdfMake(): Promise<PdfMakeBrowser> {
  if (!pdfMakePromise) {
    pdfMakePromise = loadPdfMake().catch((error: unknown) => {
      pdfMakePromise = null;
      throw error;
    });
  }
  return pdfMakePromise;
}

function textNodes(text: string, style: PdfNode, monospace: boolean): PdfNode[] {
  return splitScriptRuns(text).map((segment) => {
    if (segment.malayalam) {
      return { ...style, text: segment.text, font: MALAYALAM_FONT, fontFeatures: DISABLED_MARK_FEATURES };
    }
    const useCourier = monospace && LATIN1_RE.test(segment.text);
    return useCourier ? { ...style, text: segment.text, font: 'Courier' } : { ...style, text: segment.text };
  });
}

function runsToPdf(runs: InlineRun[]): PdfNode[] {
  return runs.flatMap((run) => {
    const style: PdfNode = {};
    if (run.bold) style.bold = true;
    if (run.italic) style.italics = true;
    if (run.code) {
      style.background = CODE_BACKGROUND;
      style.fontSize = CODE_SIZE;
    }
    return textNodes(run.text, style, !!run.code);
  });
}

interface ListTreeNode {
  runs: InlineRun[];
  children: ListTreeNode[];
}

function buildListTree(items: ListItem[]): ListTreeNode[] {
  const roots: ListTreeNode[] = [];
  const stack: { level: number; node: ListTreeNode }[] = [];
  for (const item of items) {
    const node: ListTreeNode = { runs: item.runs, children: [] };
    while (stack.length > 0 && stack[stack.length - 1].level >= item.level) stack.pop();
    const parent = stack[stack.length - 1];
    if (parent) parent.node.children.push(node);
    else roots.push(node);
    stack.push({ level: item.level, node });
  }
  return roots;
}

function listToPdf(nodes: ListTreeNode[], ordered: boolean, start: number): PdfNode {
  const entries = nodes.map((node) => {
    const label: PdfNode = { text: runsToPdf(node.runs) };
    if (node.children.length === 0) return label;
    return { stack: [label, listToPdf(node.children, ordered, 1)] };
  });
  return ordered ? { ol: entries, start, margin: [0, 2, 0, 6] } : { ul: entries, margin: [0, 2, 0, 6] };
}

function codeToPdf(text: string): PdfNode {
  return {
    table: {
      widths: ['*'],
      body: [[{ text: textNodes(text || ' ', { fontSize: CODE_SIZE }, true), preserveLeadingSpaces: true, fillColor: CODE_BACKGROUND }]],
    },
    layout: {
      hLineWidth: () => 0,
      vLineWidth: () => 0,
      paddingLeft: () => 8,
      paddingRight: () => 8,
      paddingTop: () => 6,
      paddingBottom: () => 6,
    },
    margin: [0, 4, 0, 8],
  };
}

function tableToPdf(header: InlineRun[][], rows: InlineRun[][][]): PdfNode {
  const headerRow = header.map((cell) => ({ text: runsToPdf(cell), bold: true, fillColor: CODE_BACKGROUND }));
  const bodyRows = rows.map((row) => row.map((cell) => ({ text: runsToPdf(cell) })));
  return {
    table: { headerRows: 1, widths: header.map(() => '*'), body: [headerRow, ...bodyRows] },
    layout: 'lightHorizontalLines',
    fontSize: BODY_SIZE - 1,
    margin: [0, 4, 0, 8],
  };
}

function blockToPdf(block: DocBlock): PdfNode {
  switch (block.type) {
    case 'heading':
      return {
        text: runsToPdf(block.runs),
        bold: true,
        fontSize: HEADING_SIZES[block.level - 1] ?? BODY_SIZE,
        margin: [0, 10, 0, 4],
      };
    case 'paragraph':
      return { text: runsToPdf(block.runs), margin: [0, 0, 0, 6] };
    case 'list':
      return listToPdf(buildListTree(block.items), block.ordered, block.start);
    case 'code':
      return codeToPdf(block.text);
    case 'quote':
      return { text: runsToPdf(block.runs), italics: true, color: '#475569', margin: [14, 2, 0, 8] };
    case 'table':
      return tableToPdf(block.header, block.rows);
    case 'rule':
      return {
        canvas: [{ type: 'line', x1: 0, y1: 0, x2: PAGE_CONTENT_WIDTH, y2: 0, lineWidth: 0.5, lineColor: '#cbd5e1' }],
        margin: [0, 6, 0, 8],
      };
  }
}

export function buildPdfDefinition(markdown: string, exportedAt: Date): PdfNode {
  const blocks = markdownToBlocks(markdown);
  return {
    info: { title: 'Jruva answer', author: 'Jruva', creator: 'Jruva' },
    pageSize: 'A4',
    pageMargins: [40, 56, 40, 56],
    defaultStyle: { font: 'Roboto', fontSize: BODY_SIZE, lineHeight: 1.25, color: '#1e293b' },
    header: { text: `Jruva answer · Exported ${formatExportDate(exportedAt)}`, fontSize: 8, color: '#64748b', margin: [40, 24, 40, 0] },
    footer: (currentPage: number, pageCount: number) => ({
      columns: [
        { text: EXPORT_NOTICE, fontSize: 7, color: '#94a3b8' },
        { text: `${currentPage} / ${pageCount}`, fontSize: 7, color: '#94a3b8', alignment: 'right', width: 40 },
      ],
      margin: [40, 16, 40, 0],
    }),
    content: blocks.length > 0 ? blocks.map(blockToPdf) : [{ text: '(empty answer)' }],
  };
}

/** Builds the PDF in the browser and downloads it. */
export async function exportAnswerAsPdf(markdown: string, exportedAt: Date = new Date()): Promise<void> {
  const pdfMake = await getPdfMake();
  const blob = await pdfMake.createPdf(buildPdfDefinition(markdown, exportedAt)).getBlob();
  saveBlob(blob, exportFilename('pdf', exportedAt));
}
