// 只组织已有章节，不补造内容，也不改变原有章节链接。
const labels = {
  overview: '能做什么', mechanism: '工作原理', io: '输入与输出', formula: '公式与解读', practice: '推导与排错',
  masks: '掩码与接口', code: '代码与用法', relations: '模型联系', figures: '论文图解',
}
const order = ['overview', 'io', 'mechanism', 'code', 'formula', 'practice', 'masks', 'relations', 'figures']
const primary = new Set(['overview', 'io', 'mechanism', 'code'])

export function readingSections(detail) {
  return (detail?.sections || []).filter(section => !section.document_id && labels[section.slug] &&
    (section.slug !== 'figures' || detail.figures?.length)).map(section => ({ ...section, label: labels[section.slug], primary: primary.has(section.slug) }))
    .sort((a, b) => order.indexOf(a.slug) - order.indexOf(b.slug))
}

export function supplementarySections(detail) {
  const main = new Set(readingSections(detail).map(section => section.id))
  return (detail?.sections || []).filter(section => !section.document_id && !main.has(section.id) && section.slug !== 'figures')
}

export function readingNeighbors(detail, currentId) {
  const sections = readingSections(detail)
  const index = sections.findIndex(section => section.id === currentId)
  return index < 0 ? { previous: null, next: null } : { previous: sections[index - 1] || null, next: sections[index + 1] || null }
}
