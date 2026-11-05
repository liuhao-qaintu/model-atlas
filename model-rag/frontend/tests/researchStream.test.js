import test from 'node:test'
import assert from 'node:assert/strict'
import { readEventStream } from '../src/stream.js'

function response(body) { return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } }) }
const valid = d => typeof d?.id === 'string' && typeof d?.answer === 'string' && d.generated === true

test('research events accept an independent report without a fake chat session', async () => {
  const result = { id: 'research-test', answer: '解释[1]', generated: true }
  const stages = []
  const actual = await readEventStream(response('event: stage\ndata: {"label":"正在查找"}\n\nevent: done\ndata: ' + JSON.stringify(result) + '\n\n'),
    (name, data) => stages.push({ name, data }), undefined, valid)
  assert.deepEqual(actual, result)
  assert.equal(stages[0].data.label, '正在查找')
})

test('research validation does not weaken the original chat completion contract', async () => {
  await assert.rejects(readEventStream(response('event: done\ndata: {"id":"research-test","answer":"正文","generated":true}\n\n')), /完整回答格式/)
})

test('unfinished research is rejected and not shown as a completed report', async () => {
  await assert.rejects(readEventStream(response('event: done\ndata: {"id":"research-test","answer":"半段","generated":false}\n\n'), () => {}, undefined, valid), /完整回答格式/)
})
