// fetch 可用 POST 和 AbortSignal；逐帧解析 SSE，完整 done 才作为最终答案。
export async function readEventStream(response, onEvent = () => {}, signal, validateDone = data => typeof data?.answer === 'string' && typeof data?.session_id === 'string') {
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('后端没有返回回答事件流，请检查服务版本。')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const checkAbort = () => {
    if (signal?.aborted) throw new DOMException('请求已停止', 'AbortError')
  }
  const cancel = () => { reader.cancel().catch(() => {}) }
  signal?.addEventListener('abort', cancel, { once: true })
  try {
    checkAbort()
    while (true) {
      const chunk = await reader.read()
      checkAbort()
      buffer += decoder.decode(chunk.value, { stream: !chunk.done })
      if (buffer.length > 1048576) throw new Error('回答事件过长，请重试。')
      let boundary
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        let name = 'message'
        const lines = []
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith('event:')) name = line.slice(6).trim()
          else if (line.startsWith('data:')) lines.push(line.slice(5).replace(/^ /, ''))
        }
        if (!lines.length) continue // 心跳注释不是完成事件。
        let data
        try { data = JSON.parse(lines.join('\n')) } catch { throw new Error('回答事件格式无效，请重试。') }
        if (name === 'error') throw new Error(typeof data?.detail === 'string' ? data.detail : '本轮回答未完成。')
        if ((name === 'delta' && typeof data?.text !== 'string') || (name === 'stage' && typeof data?.label !== 'string')) {
          throw new Error('回答事件内容无效，请重试。')
        }
        if (name === 'done' && !validateDone(data)) {
          throw new Error('完整回答格式无效，请重试。')
        }
        await onEvent(name, data)
        if (name === 'done') {
          await reader.cancel()
          return data
        }
      }
      if (chunk.done) throw new Error('连接提前结束，本轮回答尚未完成，请重试。')
    }
  } finally {
    signal?.removeEventListener('abort', cancel)
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

// 归类任务的事件流只有 data: JSON，没有 event: 名称；按 type 判断结束。
export async function readJsonEventStream(response, onEvent = () => {}, signal) {
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('后端没有返回归类事件流，请检查服务版本。')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const cancel = () => { reader.cancel().catch(() => {}) }
  signal?.addEventListener('abort', cancel, { once: true })
  try {
    while (true) {
      const chunk = await reader.read()
      if (signal?.aborted) throw new DOMException('请求已停止', 'AbortError')
      buffer += decoder.decode(chunk.value, { stream: !chunk.done })
      let boundary
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        const lines = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).replace(/^ /, ''))
        if (!lines.length) continue
        let data
        try { data = JSON.parse(lines.join('\n')) } catch { throw new Error('归类事件格式无效，请重试。') }
        await onEvent(data)
        if (['job_done', 'job_failed', 'job_cancelled'].includes(data?.type)) {
          await reader.cancel()
          return data
        }
      }
      if (chunk.done) throw new Error('连接提前结束，归类任务尚未完成。')
    }
  } finally {
    signal?.removeEventListener('abort', cancel)
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
