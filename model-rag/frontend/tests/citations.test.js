import test from 'node:test'
import assert from 'node:assert/strict'
import { renderMarkdown } from '../src/markdownRendering.js'
import { evidenceTargets } from '../src/citations.js'

test('numeric intervals remain text while evidence references remain clickable', () => {
  const html = renderMarkdown('Sigmoid 映射到 [0,1] 区间，范围为 [1,5]，依据[1,2]。', ['#e-1', '#e-2'])
  assert.match(html, /\[0,1\]/)
  assert.match(html, /\[1,5\]/)
  assert.equal((html.match(/data-citation=/g) || []).length, 2)
})

test('legacy grouped citations become individual links without rewriting numerical shapes', () => {
  const html = renderMarkdown('结论[1, 3，2]；形状为 [2, 3]；`[1,3]`；$[1,3]$；x[1,3]。', ['#e-1', '#e-2', '#e-3'])
  assert.equal((html.match(/data-citation=/g) || []).length, 3)
  assert.match(html, /data-citation="3"/)
  assert.match(html, /形状为 \[2, 3\]/)
  assert.match(html, /x\[1,3\]/)
})

test('citations target the correct answer and retain adjacent references', () => {
  const message = { metadata: { turn_id: 'abc-123', evidence: [{}, {}] } }
  const targets = evidenceTargets(message, 0)
  const html = renderMarkdown('结论[1][2]，缺失[99]。', targets)
  assert.match(html, /href="#evidence-abc-123-1"/)
  assert.match(html, /href="#evidence-abc-123-2"/)
  assert.match(html, /aria-label="查看依据 1"/)
  assert.ok(html.includes('[99]'))
  assert.equal((html.match(/data-citation=/g) || []).length, 2)
  assert.notDeepEqual(targets, evidenceTargets({ metadata: { turn_id: 'next', evidence: [{}] } }, 1))
})

test('code, math, indexed variables and explicit links are not converted', () => {
  const content = 'x[1]，`x[1]`，$x[1]$\n\n```python\na[1]\n```\n\n[1](https://example.org)\n\n[1][source]\n\n[source]: https://example.org'
  const html = renderMarkdown(content, ['#evidence-x-1'])
  assert.ok(!html.includes('data-citation='))
  assert.match(html, /href="https:\/\/example.org"/)
  assert.match(html, /application\/x-tex">x\[1\]/)
})

test('unsafe anchors cannot be injected and context never leaks to the next render', () => {
  const html = renderMarkdown('[1]', ['#x" onclick="alert(1)'])
  assert.ok(!html.includes('<a'))
  renderMarkdown('[1]', ['#evidence-valid-1'])
  assert.ok(!renderMarkdown('[1]').includes('<a'))
})
