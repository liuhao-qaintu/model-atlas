import test from 'node:test'
import assert from 'node:assert/strict'
import { evidenceOrigin, evidenceExcerpt, rewrittenQuery, citationGroups } from '../src/answerPresentation.js'

test('evidence summary counts documents and cards separately, retaining all citation positions', () => {
  const metadata = { evidence: [{ document_id: 'a' }, { document_id: 'a' }, { document_id: 'b' }, { model_id: 'cnn' }, { model_id: 'cnn' }] }
  assert.equal(evidenceOrigin(metadata), '2 份导入资料、1 张整理模型卡')
  assert.equal(metadata.evidence.length, 5)
  assert.equal(evidenceOrigin(), '')
})

test('query preview exposes actual follow-up rewriting without repeating an unchanged question', () => {
  assert.equal(rewrittenQuery({ original_query: ' 它输出什么？ ', normalized_query: 'ResNet-18 去掉 fc 后输出什么？' }), 'ResNet-18 去掉 fc 后输出什么？')
  assert.equal(rewrittenQuery({ original_query: 'CNN', normalized_query: ' CNN ' }), '')
  assert.equal(rewrittenQuery(), '')
})

test('excerpt prefers actual retrieval match; short evidence does not gain a misleading ellipsis', () => {
  assert.equal(evidenceExcerpt({ generation_excerpt: '提供给生成的正文', retrieval_excerpt: '仅用于排序的窗口' }), '提供给生成的正文')
  assert.equal(evidenceExcerpt({ retrieval_excerpt: 'forward 函数', content: '很长的文件头' }), 'forward 函数')
  assert.equal(evidenceExcerpt({ content: 'abcdef' }, 4), 'abcd…')
  assert.equal(evidenceExcerpt({ content: 'abc' }, 4), 'abc')
  assert.equal(evidenceExcerpt({}), '')
})

test('source groups shorten repeated titles without merging different documents or losing citation numbers', () => {
  const evidence = [
    { id: '1', document_id: 'a', title: 'CNN · PyTorch 官方实现 · Conv1d · 片段 1', locator: 'Conv1d · 片段 1', sources: [{ kind: 'document', title: 'D01 · PyTorch Conv1d v2.8.0' }] },
    { id: '2', document_id: 'a', title: 'CNN · Conv1d.forward' },
    { id: '3', document_id: 'b', title: 'CNN · PyTorch 官方实现 · Conv1d · 片段 1', locator: 'Conv1d · 片段 1' },
    { id: '4', model_id: 'cnn', title: '1D CNN · 输入输出' },
  ]
  const groups = citationGroups(evidence)
  assert.equal(groups.length, 3)
  assert.equal(groups[0].title, 'PyTorch Conv1d v2.8.0')
  assert.deepEqual(groups[0].references.map(r => r.number), [1, 2])
  assert.equal(groups[1].title, 'CNN · PyTorch 官方实现')
  assert.equal(groups[2].references[0].number, 4)
  assert.equal(groups[2].title, '1D CNN · 整理模型卡')
  assert.equal(evidence[0].title, 'CNN · PyTorch 官方实现 · Conv1d · 片段 1')
})
