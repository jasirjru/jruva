// Unit tests for the answer-export Markdown parser.
// Run: npm run test:unit   (Node >= 22.18 strips the TypeScript types itself)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  markdownToBlocks,
  parseInline,
  splitScriptRuns,
  exportFilename,
  stripHiddenContent,
  runsToText,
} from '../../src/utils/exportDoc/markdownBlocks.ts';

test('parses headings with their level and inline text', () => {
  const blocks = markdownToBlocks('# Title\n\n### Sub **part**');
  assert.equal(blocks.length, 2);
  assert.deepEqual(blocks[0], { type: 'heading', level: 1, runs: [{ text: 'Title' }] });
  assert.equal(blocks[1].type, 'heading');
  assert.equal(blocks[1].level, 3);
  assert.deepEqual(blocks[1].runs, [{ text: 'Sub ' }, { text: 'part', bold: true }]);
});

test('joins soft-wrapped lines into one paragraph and splits on blank lines', () => {
  const blocks = markdownToBlocks('first line\nsecond line\n\nnext para');
  assert.equal(blocks.length, 2);
  assert.equal(runsToText(blocks[0].runs), 'first line second line');
  assert.equal(runsToText(blocks[1].runs), 'next para');
});

test('parses bullet lists with nesting levels and continuation lines', () => {
  const blocks = markdownToBlocks('- one\n  - nested\n    more text\n- two');
  assert.equal(blocks.length, 1);
  const list = blocks[0];
  assert.equal(list.type, 'list');
  assert.equal(list.ordered, false);
  assert.deepEqual(
    list.items.map((item) => [item.level, runsToText(item.runs)]),
    [[0, 'one'], [1, 'nested more text'], [0, 'two']],
  );
});

test('parses ordered lists and keeps the start number', () => {
  const [list] = markdownToBlocks('3. three\n4. four');
  assert.equal(list.type, 'list');
  assert.equal(list.ordered, true);
  assert.equal(list.start, 3);
  assert.equal(list.items.length, 2);
});

test('keeps fenced code verbatim, including indentation and markdown symbols', () => {
  const md = 'Intro\n```python\ndef f():\n    return **x**\n```\nAfter';
  const blocks = markdownToBlocks(md);
  assert.deepEqual(blocks.map((b) => b.type), ['paragraph', 'code', 'paragraph']);
  assert.equal(blocks[1].language, 'python');
  assert.equal(blocks[1].text, 'def f():\n    return **x**');
});

test('parses pipe tables, padding short rows to the header width', () => {
  const md = '| Name | Value |\n|---|:---:|\n| a | 1 |\n| b |';
  const [table] = markdownToBlocks(md);
  assert.equal(table.type, 'table');
  assert.deepEqual(table.header.map(runsToText), ['Name', 'Value']);
  assert.deepEqual(table.rows.map((row) => row.map(runsToText)), [['a', '1'], ['b', '']]);
});

test('handles escaped pipes inside table cells', () => {
  const [table] = markdownToBlocks('| expr |\n|---|\n| a \\| b |');
  assert.equal(runsToText(table.rows[0][0]), 'a | b');
});

test('parses quotes and horizontal rules', () => {
  const blocks = markdownToBlocks('> quoted\n> text\n\n---\n\nend');
  assert.deepEqual(blocks.map((b) => b.type), ['quote', 'rule', 'paragraph']);
  assert.equal(runsToText(blocks[0].runs), 'quoted text');
});

test('inline parser handles code, italic, links and images', () => {
  assert.deepEqual(parseInline('use `npm ci` *now*'), [
    { text: 'use ' },
    { text: 'npm ci', code: true },
    { text: ' ' },
    { text: 'now', italic: true },
  ]);
  assert.equal(runsToText(parseInline('see [docs](https://example.com/x)')), 'see docs (https://example.com/x)');
  assert.equal(runsToText(parseInline('![chart](a.png)')), '[Image: chart]');
});

test('underscores in identifiers are not treated as emphasis', () => {
  assert.deepEqual(parseInline('call __init__ on snake_case_name'), [{ text: 'call __init__ on snake_case_name' }]);
});

test('strips <think> reasoning blocks, including an unclosed one', () => {
  assert.equal(stripHiddenContent('<think>secret</think>Answer'), 'Answer');
  assert.equal(stripHiddenContent('Answer<think>partial'), 'Answer');
  assert.deepEqual(markdownToBlocks('<think>x</think>'), []);
});

test('splits Malayalam and Latin text into script runs', () => {
  const runs = splitScriptRuns('Kerala: കേരളം സംസ്ഥാനം (2024)');
  assert.deepEqual(runs, [
    { text: 'Kerala: ', malayalam: false },
    { text: 'കേരളം സംസ്ഥാനം ', malayalam: true },
    { text: '(2024)', malayalam: false },
  ]);
});

test('keeps zero-width joiners inside the Malayalam run', () => {
  const runs = splitScriptRuns('ന്‍റ');
  assert.equal(runs.length, 1);
  assert.equal(runs[0].malayalam, true);
});

test('builds jruva-answer-YYYYMMDD-HHMM filenames in local time', () => {
  const date = new Date(2026, 0, 5, 9, 7);
  assert.equal(exportFilename('pdf', date), 'jruva-answer-20260105-0907.pdf');
  assert.equal(exportFilename('docx', date), 'jruva-answer-20260105-0907.docx');
});
