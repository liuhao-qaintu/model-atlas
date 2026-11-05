import test from 'node:test'
import assert from 'node:assert/strict'
import { scrollPosition } from '../src/navigation.js'
import { groupSources, sourceTitle, sourceType, documentGroupTitle } from '../src/sourcePresentation.js'

test('changing a graph edge or reader section preserves scroll', () => {
  assert.equal(scrollPosition({path:'/graph',query:{edge:'b'}}, {path:'/graph',query:{edge:'a'}}), false)
  assert.equal(scrollPosition({path:'/models/vit'}, {path:'/models/vit'}), false)
  assert.deepEqual(scrollPosition({path:'/models'}, {path:'/'}), {top:0})
  assert.deepEqual(scrollPosition({path:'/graph'}, {path:'/models'}, {top:420}), {top:420})
})

test('sources show human labels and preserve genuinely different types', () => {
  assert.equal(sourceTitle('S01：Conv1d Shape'), 'Conv1d Shape')
  assert.equal(sourceType({kind:'external',url:'https://arxiv.org/pdf/x'}), '研究论文')
  assert.equal(sourceType({kind:'internal'}), '教学示例')
  assert.equal(sourceType({kind:'external',url:'https://example.com'}), '参考资料')
})

test('one original source can show related models and an imported reading entry', () => {
  const result = groupSources([{id:'S1',kind:'external',url:'https://arxiv.org/pdf/x#page=3',topic:'vit',scope:'方法'},
    {id:'D1',kind:'document',url:'/documents/a',original_url:'https://arxiv.org/pdf/x',topic:'clip',scope:'已入库'}])
  assert.equal(result.length, 1); assert.equal(result[0].url, '/documents/a')
  assert.deepEqual(result[0].related_topics, ['vit','clip'])
  assert.ok(result[0].scope.includes('方法'))
})

test('document headings preserve distinguishing tutorial or implementation names', () => {
  assert.equal(documentGroupTitle({title:'PyTorch 官方教程 · 自动梯度 · 第1节', locator:'第1节'}), 'PyTorch 官方教程 · 自动梯度')
  assert.equal(documentGroupTitle({title:'ViT 原文 · 第3页 · 片段1', locator:'第3页 · 片段1'}), 'ViT 原文')
})
