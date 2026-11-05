import test from 'node:test'
import assert from 'node:assert/strict'
import { conversationContext, mergeMessages, sessionListPath } from '../src/sessionHistory.js'

test('restored context belongs to the latest answer and preserves research mode', () => {
  const messages = [
    { role: 'assistant', metadata: { model_id: 'cnn-1d', mode: 'learning' } },
    { role: 'assistant', metadata: { model_id: 'lstm', section_id: 'lstm:io', mode: 'research' } },
    { role: 'user', content: '未完成的提问' },
  ]
  assert.deepEqual(conversationContext(messages, [{ id: 'cnn-1d' }, { id: 'lstm' }]),
    { modelId: 'lstm', sectionId: 'lstm:io', mode: 'research' })
  assert.deepEqual(conversationContext(messages, [{ id: 'cnn-1d' }]),
    { modelId: '', sectionId: '', mode: 'research' })
  assert.deepEqual(conversationContext([], []), { modelId: '', sectionId: '', mode: 'learning' })
})

test('older pages and refreshed saved turns merge by message identity and sequence', () => {
  const older = [{ id: 'a', sequence: 1 }, { id: 'b', sequence: 2, content: 'old' }]
  const current = [{ id: 'b', sequence: 2, content: 'saved' }, { id: 'c', sequence: 9 }]
  const merged = mergeMessages(older, current)
  assert.deepEqual(merged.map(m => m.id), ['a', 'b', 'c'])
  assert.equal(merged[1].content, 'saved')
  assert.equal(older[1].content, 'old')
  const pending = [{ role: 'user', content: '待同步提问' }, { role: 'assistant', content: '已生成回答' }]
  assert.deepEqual(mergeMessages(older, pending).slice(-2), pending)
})

test('search text is encoded as one parameter and pagination retains it', () => {
  const url = new URL('http://localhost' + sessionListPath(' 中文 & before=1 ', 99))
  assert.equal(url.searchParams.get('query'), '中文 & before=1')
  assert.equal(url.searchParams.get('before'), '99')
  assert.equal(url.searchParams.get('limit'), '20')
  assert.equal(new URL('http://localhost' + sessionListPath(' ')).searchParams.has('query'), false)
})
