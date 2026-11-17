/**
 * Word (.docx) export of one assistant answer, generated in the browser with
 * the `docx` package (MIT). Loaded only through `await import()`.
 *
 * Word renders the text with the reader's installed fonts, so Malayalam and
 * other scripts display as well as the reader's system supports them.
 */
import type * as Docx from 'docx';
import type { DocBlock, InlineRun, ListItem } from './markdownBlocks';
import { markdownToBlocks, exportFilename } from './markdownBlocks';
import { EXPORT_NOTICE, formatExportDate, saveBlob } from './download';

const CODE_FONT = 'Consolas';
const CODE_SHADING = 'F1F5F9';
const CODE_SIZE_HALF_POINTS = 19;
const BULLET_INDENT_TWIPS = 360;
const MAX_LEVELS = 6;

type DocxModule = typeof Docx;
type BodyChild = Docx.Paragraph | Docx.Table;

function runsToDocx(docx: DocxModule, runs: InlineRun[]): Docx.TextRun[] {
  return runs.map(
    (run) =>
      new docx.TextRun({
        text: run.text,
        bold: run.bold,
        italics: run.italic,
        ...(run.code
          ? {
              font: CODE_FONT,
              size: CODE_SIZE_HALF_POINTS,
              shading: { type: docx.ShadingType.CLEAR, color: 'auto', fill: CODE_SHADING },
            }
          : {}),
      }),
  );
}

const HEADING_LEVELS = ['HEADING_1', 'HEADING_2', 'HEADING_3', 'HEADING_4', 'HEADING_5', 'HEADING_6'] as const;

function listToDocx(docx: DocxModule, items: ListItem[], ordered: boolean, reference: string): Docx.Paragraph[] {
  return items.map(
    (item) =>
      new docx.Paragraph({
        children: runsToDocx(docx, item.runs),
        ...(ordered ? { numbering: { reference, level: item.level } } : { bullet: { level: item.level } }),
      }),
  );
}

function codeToDocx(docx: DocxModule, text: string): Docx.Paragraph[] {
  const lines = (text || ' ').split('\n');
  return lines.map(
    (line, index) =>
      new docx.Paragraph({
        children: [new docx.TextRun({ text: line || ' ', font: CODE_FONT, size: CODE_SIZE_HALF_POINTS })],
        shading: { type: docx.ShadingType.CLEAR, color: 'auto', fill: CODE_SHADING },
        spacing: { before: index === 0 ? 120 : 0, after: index === lines.length - 1 ? 160 : 0 },
      }),
  );
}

function tableToDocx(docx: DocxModule, header: InlineRun[][], rows: InlineRun[][][]): Docx.Table {
  const makeRow = (cells: InlineRun[][], isHeader: boolean) =>
    new docx.TableRow({
      tableHeader: isHeader,
      children: cells.map(
        (cell) =>
          new docx.TableCell({
            children: [
              new docx.Paragraph({
                children: runsToDocx(
                  docx,
                  cell.map((run) => (isHeader ? { ...run, bold: true } : run)),
                ),
              }),
            ],
            ...(isHeader ? { shading: { type: docx.ShadingType.CLEAR, color: 'auto', fill: CODE_SHADING } } : {}),
          }),
      ),
    });
  return new docx.Table({
    width: { size: 100, type: docx.WidthType.PERCENTAGE },
    rows: [makeRow(header, true), ...rows.map((row) => makeRow(row, false))],
  });
}

function blockToDocx(docx: DocxModule, block: DocBlock, listReference: string): BodyChild[] {
  switch (block.type) {
    case 'heading':
      return [
        new docx.Paragraph({
          heading: docx.HeadingLevel[HEADING_LEVELS[Math.min(block.level, MAX_LEVELS) - 1]],
          children: runsToDocx(docx, block.runs),
        }),
      ];
    case 'paragraph':
      return [new docx.Paragraph({ children: runsToDocx(docx, block.runs), spacing: { after: 120 } })];
    case 'list':
      return listToDocx(docx, block.items, block.ordered, listReference);
    case 'code':
      return codeToDocx(docx, block.text);
    case 'quote':
      return [
        new docx.Paragraph({
          children: runsToDocx(docx, block.runs.map((run) => ({ ...run, italic: true }))),
          indent: { left: BULLET_INDENT_TWIPS },
          border: { left: { style: docx.BorderStyle.SINGLE, size: 12, color: 'CBD5E1', space: 8 } },
        }),
      ];
    case 'table':
      return [tableToDocx(docx, block.header, block.rows), new docx.Paragraph({ children: [] })];
    case 'rule':
      return [
        new docx.Paragraph({
          children: [],
          border: { bottom: { style: docx.BorderStyle.SINGLE, size: 6, color: 'CBD5E1', space: 1 } },
        }),
      ];
  }
}

function orderedListNumbering(docx: DocxModule, reference: string, start: number) {
  return {
    reference,
    levels: Array.from({ length: MAX_LEVELS }, (_, level) => ({
      level,
      format: docx.LevelFormat.DECIMAL,
      text: `%${level + 1}.`,
      start: level === 0 ? start : 1,
      alignment: docx.AlignmentType.START,
      style: { paragraph: { indent: { left: BULLET_INDENT_TWIPS * (level + 1), hanging: BULLET_INDENT_TWIPS } } },
    })),
  };
}

export async function buildDocxBlob(markdown: string, exportedAt: Date): Promise<Blob> {
  const docx: DocxModule = await import('docx');
  const blocks = markdownToBlocks(markdown);
  const numberingConfig: ReturnType<typeof orderedListNumbering>[] = [];
  const children: BodyChild[] = [
    new docx.Paragraph({
      children: [new docx.TextRun({ text: `Jruva answer · Exported ${formatExportDate(exportedAt)}`, size: 16, color: '64748B' })],
      spacing: { after: 240 },
    }),
  ];

  blocks.forEach((block, index) => {
    const reference = `jruva-ol-${index}`;
    if (block.type === 'list' && block.ordered) numberingConfig.push(orderedListNumbering(docx, reference, block.start));
    children.push(...blockToDocx(docx, block, reference));
  });
  if (blocks.length === 0) children.push(new docx.Paragraph({ text: '(empty answer)' }));

  const document = new docx.Document({
    creator: 'Jruva',
    title: 'Jruva answer',
    description: EXPORT_NOTICE,
    numbering: { config: numberingConfig },
    sections: [
      {
        footers: {
          default: new docx.Footer({
            children: [new docx.Paragraph({ children: [new docx.TextRun({ text: EXPORT_NOTICE, size: 14, color: '94A3B8' })] })],
          }),
        },
        children,
      },
    ],
  });
  return docx.Packer.toBlob(document);
}

/** Builds the .docx in the browser and downloads it. */
export async function exportAnswerAsDocx(markdown: string, exportedAt: Date = new Date()): Promise<void> {
  const blob = await buildDocxBlob(markdown, exportedAt);
  saveBlob(blob, exportFilename('docx', exportedAt));
}
