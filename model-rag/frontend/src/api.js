import { csrfHeaders,sessionExpired } from './auth.js'
import { readEventStream, readJsonEventStream } from './stream'

export async function api(path, body, options = {}) {
  const isFile = body instanceof FormData
  const method = options.method || (body === undefined ? 'GET' : 'POST')
  const response = await fetch(`/api${path}`, {
    method,
    credentials:'same-origin',
    headers: { ...(body === undefined || isFile ? {} : { 'Content-Type': 'application/json' }),...csrfHeaders(method) },
    body: body === undefined ? undefined : isFile ? body : JSON.stringify(body),
    signal: options.signal,
  })
  const result = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = result?.detail
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map(item => item?.msg || JSON.stringify(item)).join('；')
        : `请求失败（${response.status}）`
    if (response.status===401 && !path.startsWith('/auth/')) sessionExpired()
    const error = new Error(message)
    error.status = response.status
    throw error
  }
  return result
}

export async function chatStream(body, onEvent, signal) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream',...csrfHeaders('POST') },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    if (response.status===401) sessionExpired()
    const result = await response.json().catch(() => null)
    throw new Error(typeof result?.detail === 'string' ? result.detail : `请求失败（${response.status}）`)
  }
  return readEventStream(response, onEvent, signal)
}

export async function researchStream(body, onEvent, signal) {
  const response = await fetch('/api/research/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream',...csrfHeaders('POST') },
    body: JSON.stringify(body), signal,
  })
  if (!response.ok) {
    if (response.status===401) sessionExpired()
    const result = await response.json().catch(() => null)
    throw new Error(typeof result?.detail === 'string' ? result.detail : `请求失败（${response.status}）`)
  }
  return readEventStream(response, onEvent, signal, data =>
    typeof data?.id === 'string' && data.id.startsWith('research-') && data.protocol === 'agentic-evidence-gap-v1'
      && typeof data.answer === 'string' && data.generated === true && Array.isArray(data.evidence))
}

export function startIngest(documentId) {
  return api(`/ingest/documents/${encodeURIComponent(documentId)}`, {})
}

export function ingestJob(jobId) {
  return api(`/ingest/jobs/${encodeURIComponent(jobId)}`)
}

export function cancelIngest(jobId) {
  return api(`/ingest/jobs/${encodeURIComponent(jobId)}/cancel`, {})
}

export async function ingestStream(jobId, onEvent, signal) {
  const response = await fetch(`/api/ingest/jobs/${encodeURIComponent(jobId)}/events`, {
    headers: { Accept: 'text/event-stream' },
    signal,
  })
  if (!response.ok) {
    if (response.status===401) sessionExpired()
    const result = await response.json().catch(() => null)
    throw new Error(typeof result?.detail === 'string' ? result.detail : `请求失败（${response.status}）`)
  }
  return readJsonEventStream(response, onEvent, signal)
}

export function listReviewItems(params = {}) {
  const query = new URLSearchParams(params).toString()
  return api(`/review/items${query ? `?${query}` : ''}`)
}

export function approveReview(itemId) {
  return api(`/review/items/${encodeURIComponent(itemId)}/approve`, {})
}

export function rejectReview(itemId) {
  return api(`/review/items/${encodeURIComponent(itemId)}/reject`, {})
}

export function editReview(itemId, proposed) {
  return api(`/review/items/${encodeURIComponent(itemId)}/edit`, { proposed })
}

export function refreshReview() {
  return api('/review/refresh', {})
}

export function deleteDocument(documentId) {
  return api(`/documents/${encodeURIComponent(documentId)}`, { confirm: documentId }, { method: 'DELETE' })
}
