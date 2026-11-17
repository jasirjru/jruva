// Minimal types for the pdfmake 0.3 browser build, limited to what
// src/utils/exportDoc/pdfExport.ts uses. pdfmake ships no TypeScript types.

declare module 'pdfmake/build/pdfmake' {
  export interface PdfFontFamily {
    normal: string;
    bold: string;
    italics: string;
    bolditalics: string;
  }

  export interface PdfFontContainer {
    vfs?: Record<string, string | { data: string; encoding?: string }>;
    fonts?: Record<string, PdfFontFamily>;
    standardFonts?: unknown[];
  }

  export interface PdfOutputDocument {
    getBlob(): Promise<Blob>;
  }

  export interface PdfMakeBrowser {
    addFontContainer(container: PdfFontContainer): void;
    addVirtualFileSystem(vfs: Record<string, string>): void;
    addFonts(fonts: Record<string, PdfFontFamily>): void;
    setUrlAccessPolicy(callback: (url: string) => boolean): void;
    createPdf(docDefinition: object): PdfOutputDocument;
  }

  const pdfMake: PdfMakeBrowser;
  export default pdfMake;
}

declare module 'pdfmake/build/fonts/Roboto' {
  import type { PdfFontContainer } from 'pdfmake/build/pdfmake';
  const container: PdfFontContainer;
  export default container;
}

declare module 'pdfmake/build/standard-fonts/Courier' {
  import type { PdfFontContainer } from 'pdfmake/build/pdfmake';
  const container: PdfFontContainer;
  export default container;
}
