import test from 'node:test'
import assert from 'node:assert/strict'
import { readEventStream } from '../src/stream.js'

const encode = new TextEncoder()
const done = { session_id: 'test-session', answer: '完整回答' }
const frame = (event, data, ending = '\n') => `event: ${event}${ending}data: ${JSON.stringify(data)}${ending}${ending}`

function response(text, chunkSize = 1) {
  const bytes = encode.encode(text)
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}

test('UTF-8 中文与 CRLF 分散在不同网络块仍正确解析', async () => {
  const seen = []
  const text = ': heartbeat\r\n\r\n' + frame('stage', { label: '检索资料' }, '\r\n') + frame('delta', { text: '维度 [B,C,L]' }, '\r\n') + frame('done', done, '\r\n')
  const result = await readEventStream(response(text), (name, data) => seen.push([name, data]))
  assert.deepEqual(result, done)
  assert.deepEqual(seen.map(([name]) => name), ['stage', 'delta', 'done'])
  assert.equal(seen[1][1].text, '维度 [B,C,L]')
})

test('没有 done 的部分答案不能作为完整回答', async () => {
  await assert.rejects(readEventStream(response(frame('delta', { text: '未完成' }))), /连接提前结束/)
})

test('服务错误与无效正文明确报错', async () => {
  await assert.rejects(readEventStream(response(frame('error', { detail: '检索失败' }))), /检索失败/)
  await assert.rejects(readEventStream(response(frame('delta', { text: { invalid: true } }))), /内容无效/)
  await assert.rejects(readEventStream(response('event: delta\ndata: not-json\n\n')), /格式无效/)
  await assert.rejects(readEventStream(response(frame('done', { answer: '缺少会话' }))), /完整回答格式无效/)
})

test('停止会关闭读取，不把中断当作成功', async () => {
  const controller = new AbortController()
  let closed = false
  const stream = new ReadableStream({ cancel() { closed = true } })
  const pending = readEventStream(new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } }), () => {}, controller.signal)
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
  assert.equal(closed, true)
})

test('解析错误也关闭连接', async () => {
  let closed = false
  const stream = new ReadableStream({
    start(controller) { controller.enqueue(encode.encode('event: delta\ndata: invalid\n\n')) },
    cancel() { closed = true },
  })
  await assert.rejects(readEventStream(new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } })), /格式无效/)
  assert.equal(closed, true)
})
