export function conversationContext(messages, models) {
  const metadata = [...messages].reverse().find(message => message.role === 'assistant')?.metadata || {}
  const modelId = models.some(model => model.id === metadata.model_id) ? metadata.model_id : ''
  return { modelId, sectionId: modelId ? metadata.section_id || '' : '',
    mode: metadata.mode === 'research' ? 'research' : 'learning' }
}

export function mergeMessages(older, current) {
  const messages = new Map(), pending = []
  for (const message of [...older, ...current]) {
    if (message.id) messages.set(message.id, message)
    else pending.push(message)
  }
  return [...messages.values()].sort((a, b) => (a.sequence ?? Number.MAX_SAFE_INTEGER) - (b.sequence ?? Number.MAX_SAFE_INTEGER)).concat(pending)
}

export function sessionListPath(query = '', cursor = null) {
  const parameters = new URLSearchParams({ limit: '20' })
  if (query.trim()) parameters.set('query', query.trim())
  if (cursor) parameters.set('before', String(cursor))
  return '/sessions?' + parameters.toString()
}
