export function sourceType(source) {
  if (source.kind === 'internal') return '教学示例'
  if (source.kind === 'document') return '入库原文'
  if (/论文|arxiv|paper/i.test(`${source.title} ${source.url}`)) return '研究论文'
  if (/d2l|deeplearningbook/.test(source.url || '')) return '教材'
  return /pytorch|github\.com|openreview/i.test(source.url || '') ? '官方资料' : '参考资料'
}

export function sourceTitle(title = '') {
  return title.replace(/^(?:[SED]\d+|A\d+)\s*[·:：]\s*/, '')
}

export function groupSources(sources) {
  const groups = new Map()
  for (const source of sources) {
    const key = (source.original_url || source.url || source.id).split('#')[0]
    if (!groups.has(key)) groups.set(key, { ...source, related_topics: [], scopes: [], originals: [] })
    const group = groups.get(key)
    if (source.kind === 'document') Object.assign(group, { kind: 'document', url: source.url, title: source.title, original_url: source.original_url })
    group.related_topics.push(source.topic)
    group.scopes.push(source.scope)
    group.originals.push(source)
  }
  return [...groups.values()].map(group => ({ ...group, related_topics: [...new Set(group.related_topics.filter(Boolean))], scope: [...new Set(group.scopes.filter(Boolean))].join('；') }))
}

export function documentGroupTitle(section) {
  const suffix = section.locator ? ` · ${section.locator}` : ''
  return suffix && section.title.endsWith(suffix) ? section.title.slice(0, -suffix.length) : section.title
}
