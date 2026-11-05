import test from 'node:test'
import assert from 'node:assert/strict'
import { AE_PAPER_URL, AE_COMPARISON, aePaper, aeLocation, aeReadingContent } from '../src/aeJourney.js'

test('sample resolves current paper by source rather than database identity', () => {
  assert.equal(aePaper([{ id: 'old', status: 'archived', source_url: AE_PAPER_URL },
    { id: 'current', status: 'imported', source_url: AE_PAPER_URL }]).id, 'current')
  assert.equal(aePaper([{ id: 'unrelated', status: 'imported', source_url: 'https://example.com/paper.pdf' }]), null)
})

test('missing paper gives a preparation page instead of a broken document link', () => {
  assert.equal(aeLocation('paper', null).path, '/learn/autoencoder')
  assert.equal(aeLocation('paper', { id: 'fresh-id' }).path, '/documents/fresh-id')
})

test('reading route is explicit and comparison asks to distinguish original work from teaching code', () => {
  assert.deepEqual(aeLocation('model'), { path: '/models/autoencoder', query: { section: 'io', flow: 'ae' } })
  assert.equal(aeLocation('compare').query.question, AE_COMPARISON)
  assert.match(AE_COMPARISON, /区分论文设定与教学代码/)
  assert.match(AE_COMPARISON, /不要补造/)
  assert.equal(aeLocation('compare').query.document, undefined)
})

test('inline chapter links preserve only the opted-in reading route', () => {
  const source = '[代码](/models/autoencoder?section=code) [外链](https://example.com/)'
  assert.equal(aeReadingContent(source, '', 'autoencoder'), source)
  assert.equal(aeReadingContent(source, 'ae', 'clip'), source)
  assert.match(aeReadingContent(source, 'ae', 'autoencoder'), /section=code&flow=ae/)
  assert.match(aeReadingContent(source, 'ae', 'autoencoder'), /https:\/\/example.com\//)
})
