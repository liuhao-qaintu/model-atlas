import test from 'node:test'
import assert from 'node:assert/strict'
import { renderMarkdown, normalizeMathCommands } from '../src/markdownRendering.js'

test('projection formulas have single slashes and render unchanged multiplication', () => {
  const answer = String.raw`投影函数为 \(g(\cdot)\)，视觉特征投影为 \(H_v = W \cdot Z_v\)。`
  assert.ok(answer.includes(String.raw`\cdot`))
  assert.ok(!answer.includes(String.raw`\\cdot`))
  const rendered = renderMarkdown(answer)
  const math = [...rendered.matchAll(/<annotation encoding="application\/x-tex">([^<]*)<\/annotation>/g)].map(match => match[1])
  assert.ok(math.includes(String.raw`g(\cdot)`))
  assert.ok(math.includes(String.raw`H_v = W \cdot Z_v`))
  assert.ok(!rendered.includes('katex-error'))
})

test('duplicated slash in a supported math command is handled only inside math', () => {
  const content = '$' + String.raw`H_v=W\\cdot Z_v` + '$；`' + String.raw`W\\cdot Z` + '`'
  const rendered = renderMarkdown(content)
  assert.match(rendered, /<annotation encoding="application\/x-tex">H_v=W\\cdot Z_v<\/annotation>/)
  assert.ok(rendered.includes('<code>' + String.raw`W\\cdot Z` + '</code>'))
})

test('normal formulas and environment row separators remain unchanged', () => {
  const matrix = String.raw`\begin{bmatrix}1&2\\3&4\end{bmatrix}`
  const aligned = String.raw`\begin{aligned}a&=b\\\theta&=c\end{aligned}`
  const content = '$$\n' + matrix + '\n$$\n\n$$\n' + aligned + '\n$$'
  assert.equal(normalizeMathCommands(matrix), matrix)
  assert.equal(normalizeMathCommands(aligned), aligned)
  assert.equal(normalizeMathCommands(String.raw`a\cdot b`), String.raw`a\cdot b`)
  const rendered = renderMarkdown(content)
  assert.ok(rendered.includes(matrix.replaceAll('&', '&amp;')))
  assert.ok(rendered.includes(aligned.replaceAll('&', '&amp;')))
  assert.ok(!rendered.includes('katex-error'))
})

test('fences, double backticks and indented code retain literal formula delimiters', () => {
  const contents = [
    '```text\n' + String.raw`\(x\\cdot y\)` + '\n```',
    '~~~text\n' + String.raw`\[x\\cdot y\]` + '\n~~~',
    '``' + String.raw`\(x\\cdot y\)` + '``',
    '    ' + String.raw`\(x\\cdot y\)`,
  ]
  for (const content of contents) {
    const rendered = renderMarkdown(content)
    assert.ok(!rendered.includes('class="katex"'))
    assert.ok(rendered.includes(String.raw`\\cdot`))
    assert.ok(rendered.includes('<code'))
  }
})

test('parentheses and brackets render inside prose, tables and standalone blocks', () => {
  const inline = renderMarkdown(String.raw`投影 \(H = W \cdot Z\) 完成。`)
  const table = renderMarkdown('| 位置 | 数学式 |\n|---|---|\n| 投影 | ' + String.raw`\(H=W\cdot Z\)` + ' |')
  const block = renderMarkdown(String.raw`\[` + '\n' + String.raw`H=W\cdot Z` + '\n' + String.raw`\]`)
  assert.match(inline, /class="katex"/)
  assert.match(table, /<td><span class="katex">/)
  assert.match(block, /^<span class="katex-display">/)
  assert.ok(![inline, table, block].some(html => html.includes('katex-error')))
})

test('unknown commands, literal double slash and triple-slash row separator are preserved', () => {
  for (const text of [String.raw`a\\ b`, String.raw`a\\3`, String.raw`a\\\alpha`, String.raw`\\unknown{a}`]) {
    assert.equal(normalizeMathCommands(text), text)
  }
})

test('link cleanup and syntax highlighting still apply with the shared parser', () => {
  const content = '来源：https://example.org/source。正文\n\n```python\nx = a[0]\n```\n\n$x[0]$'
  const rendered = renderMarkdown(content)
  assert.match(rendered, /href="https:\/\/example.org\/source"/)
  assert.match(rendered, /<\/a>。正文/)
  assert.match(rendered, /class="hljs"/)
  assert.match(rendered, /<annotation encoding="application\/x-tex">x\[0\]<\/annotation>/)
})

test('saved CNN formula renders between adjacent prose and citations', () => {
  const answer = '输出长度按公式\n$$\n' + String.raw`L_{out}=\left\lfloor\frac{L_{in}+2p-d(k-1)-1}{s}+1\right\rfloor` + '\n$$\n计算，其中 p 是 padding [4][3]。'
  const targets = Array.from({ length: 5 }, (_, i) => `#evidence-cnn-${i + 1}`)
  const html = renderMarkdown(answer, targets)
  assert.ok(answer.includes('\n$$\n'))
  assert.match(html, /class="katex-display"/)
  assert.ok(!html.includes('$$'))
  assert.ok(!html.includes('katex-error'))
  assert.match(html, /<annotation encoding="application\/x-tex">L_\{out\}=/)
  assert.match(html, /href="#evidence-cnn-4"/)
  assert.match(html, /href="#evidence-cnn-3"/)
})

test('adjacent multiline dollar math supports CRLF, lists and blockquotes', () => {
  for (const source of [
    'Before\n$$\nx^2\n$$\nafter.',
    'Before\r\n$$ \r\nx^2\r\n$$\r\nafter.',
    '- Before\n  $$\n  x^2\n  $$\n  after.',
    '> Before\n> $$\n> x^2\n> $$\n> after.',
  ]) {
    const html = renderMarkdown(source)
    assert.match(html, /class="katex-display"/)
    assert.ok(!html.includes('$$'))
    assert.ok(!html.includes('katex-error'))
    assert.ok(html.includes('Before') && html.includes('after.'))
  }
})

test('multiline dollar literals stay intact inside code and incomplete formulas stay visible', () => {
  for (const source of [
    '```text\n$$\nx^2\n$$\n```',
    '~~~text\n$$\nx^2\n$$\n~~~',
    '`literal\n$$\nx^2\n$$\nend`',
    '    $$\n    x^2\n    $$',
  ]) {
    const html = renderMarkdown(source)
    assert.ok(html.includes('<code'))
    assert.ok(html.includes('$$'))
    assert.ok(!html.includes('class="katex'))
  }
  const incomplete = renderMarkdown('Before\n$$\nx^2\nafter.')
  assert.ok(incomplete.includes('$$'))
  assert.ok(!incomplete.includes('class="katex'))
})
