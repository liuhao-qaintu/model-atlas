import test from 'node:test'
import assert from 'node:assert/strict'
import { readingSections, supplementarySections, readingNeighbors } from '../src/readingPresentation.js'

const sections = [
  { id: 'cnn:overview', slug: 'overview', title: '原始概览标题' },
  { id: 'cnn:io', slug: 'io', title: '输入输出' },
  { id: 'cnn:code', slug: 'code', title: '最小代码' },
  { id: 'cnn:figures', slug: 'figures', title: '图示准备状态' },
  { id: 'cnn:sources', slug: 'sources', title: '来源与定位' },
  { id: 'cnn:custom', slug: 'custom', title: '用户补充内容' },
  { id: 'doc:1:0', slug: 'doc-1', title: '原文第1页', document_id: 'doc:1' },
]

test('reading menu omits raw fragments and empty figure placeholders, preserving content and IDs', () => {
  const detail = { sections, figures: [] }
  const menu = readingSections(detail)
  assert.deepEqual(menu.map(s => s.id), ['cnn:overview', 'cnn:io', 'cnn:code'])
  assert.equal(menu[0].title, '原始概览标题')
  assert.equal(menu[0].label, '能做什么')
  assert.equal(sections[0].label, undefined)
  assert.equal(readingSections({ sections, figures: [{}] }).at(-1).id, 'cnn:figures')
})

test('reading order prioritizes the task and data before detail, without changing stored chapters', () => {
  const source = ['formula', 'code', 'relations', 'mechanism', 'overview', 'io', 'practice'].map(slug => ({ id: `detr:${slug}`, slug, title: slug }))
  const menu = readingSections({ sections: source })
  assert.deepEqual(menu.map(s => s.slug), ['overview', 'io', 'mechanism', 'code', 'formula', 'practice', 'relations'])
  assert.deepEqual(menu.filter(s => s.primary).map(s => s.slug), ['overview', 'io', 'mechanism', 'code'])
  assert.equal(readingNeighbors({ sections: source }, 'detr:overview').next.id, 'detr:io')
  assert.deepEqual(source.map(s => s.slug), ['formula', 'code', 'relations', 'mechanism', 'overview', 'io', 'practice'])
})

test('source and custom chapters remain reachable in supplementary reading', () => {
  assert.deepEqual(supplementarySections({ sections }).map(s => s.id), ['cnn:sources', 'cnn:custom'])
  assert.deepEqual(supplementarySections(null), [])
})

test('previous and next point only to existing reading chapters', () => {
  const detail = { sections }
  assert.equal(readingNeighbors(detail, 'cnn:overview').previous, null)
  assert.equal(readingNeighbors(detail, 'cnn:overview').next.slug, 'io')
  assert.equal(readingNeighbors(detail, 'cnn:code').next, null)
  assert.equal(readingNeighbors(detail, 'cnn:io').previous.slug, 'overview')
  assert.deepEqual(readingNeighbors(detail, 'doc:1:0'), { previous: null, next: null })
  assert.deepEqual(readingNeighbors(null, 'missing'), { previous: null, next: null })
})
