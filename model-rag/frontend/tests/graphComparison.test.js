import test from 'node:test'
import assert from 'node:assert/strict'
import { comparisonPair, comparisonRoute } from '../src/graphComparison.js'

const component = { name: '残差相加', implementations: [
  { edge_id: 'r', model_name: 'ResNet-18', entity: 'BasicBlock' },
  { edge_id: 'c1', model_name: 'CLIP', entity: 'CLIP文本Transformer' },
  { edge_id: 'c2', model_name: 'CLIP', entity: 'CLIP视觉ViT' },
  { edge_id: 'v', model_name: 'ViT', entity: 'EncoderBlock' },
  { edge_id: 't', model_name: 'Transformer', entity: 'Transformer层' },
] }
test('comparison defaults to two different model owners', () => assert.deepEqual(comparisonPair(component), ['r', 'c1']))
test('comparison contains only selected implementations', () => {
  const route = comparisonRoute(component, ['r', 'v'])
  assert.match(route.query.question, /ResNet-18 中的 BasicBlock 与 ViT 中的 EncoderBlock/)
  assert.doesNotMatch(route.query.question, /CLIP|Transformer/)
})
test('comparison rejects duplicate or unknown implementations', () => {
  assert.equal(comparisonRoute(component, ['r', 'r']), null)
  assert.equal(comparisonRoute(component, ['r', 'missing']), null)
})
