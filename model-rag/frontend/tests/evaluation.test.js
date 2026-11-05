import test from 'node:test'
import assert from 'node:assert/strict'
import { filteredCases, judgmentFor, metricText, precisionStatus } from '../src/evaluationPresentation.js'

test('unjudged and measured zero have different display states', () => {
  assert.equal(metricText(null), '待标注'); assert.equal(metricText(0), '0.0%')
  assert.equal(judgmentFor({}, 'a').label, 'unjudged')
  assert.equal(judgmentFor({ judgments: { a: { label: 'irrelevant', origin: 'user' } } }, 'a').label, 'irrelevant')
})
test('development selection cannot silently include challenge cases', () => {
  const all = [{ id: 'dev' }, { id: 'hold', split: 'challenge' }]
  assert.deepEqual(filteredCases(all, 'development').map(c => c.id), ['dev'])
  assert.deepEqual(filteredCases(all, 'challenge').map(c => c.id), ['hold'])
  assert.equal(filteredCases(all, 'all').length, 2)
})
test('row status differentiates pending, empty and completed evidence judgments', () => {
  assert.equal(precisionStatus({ metrics: { retrieved: 2, unjudged: 1, returned_precision: null } }), '待判断 1 条')
  assert.equal(precisionStatus({ metrics: { retrieved: 0, unjudged: 0 } }), '未返回证据')
  assert.equal(precisionStatus({ metrics: { retrieved: 2, unjudged: 0, returned_precision: .5 } }), '实际精确率 50.0%')
})
