import { sourceTitle } from './sourcePresentation.js'

export function evidenceOrigin(metadata = {}) {
  const evidence = metadata.evidence || []
  const documents = new Set(evidence.filter(item => item.document_id).map(item => item.document_id)).size
  const cards = new Set(evidence.filter(item => !item.document_id).map(item => item.model_id).filter(Boolean)).size
  return [documents ? `${documents} 份导入资料` : '', cards ? `${cards} 张整理模型卡` : ''].filter(Boolean).join('、')
}

export function rewrittenQuery(plan) {
  const original = (plan?.original_query || '').trim()
  const normalized = (plan?.normalized_query || '').trim()
  return normalized && normalized !== original ? normalized : ''
}

export function evidenceExcerpt(item, limit = 320) {
  const text = item.generation_excerpt || item.retrieval_excerpt || item.content || ''
  return text.length > limit ? `${text.slice(0, limit)}…` : text
}
export function citationGroups(evidence = []) {
  const groups = new Map()
  evidence.forEach((item, index) => {
    // Distinct documents keep their identities even if their titles happen to match.
    const key = item.document_id ? `document:${item.document_id}` : `card:${item.model_id || item.id || index}`
    const documentSource = (item.sources || []).find(source => source.kind === 'document')
    const title = item.document_id ? documentSource?.title || item.title : item.title
    const parts = sourceTitle(title || '').split(/\s+·\s+/)
    if (!documentSource && item.locator && title?.endsWith(` · ${item.locator}`)) parts.splice(parts.length - item.locator.split(/\s+·\s+/).length)
    const label = item.document_id ? parts.join(' · ') : `${parts[0]} · 整理模型卡`
    if (!groups.has(key)) groups.set(key, { key, title: label, references: [] })
    groups.get(key).references.push({ number: index + 1, title: sourceTitle(item.title || '') })
  })
  return [...groups.values()]
}
