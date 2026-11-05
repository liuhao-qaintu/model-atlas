const defaults = { summary: '文章概述', contributions: '资料要点', methods: '方法与数据流',
  formula_explanations: '公式与条件', limitations: '适用范围与待核实内容' }
const roles = { baseline: '基础方法', proposed: '本文方法', background: '相关工作', reviewed: '综述对象',
  teaching: '教学示例', interface: '接口说明', document: '文章整体' }
export function reportFields(report) {
  const legacy = report?.document_type === '综述' ? { contributions: '综述组织与归纳价值', methods: '方法类别与比较' } : {}
  const labels = { ...defaults, ...legacy, ...report?.report_profile?.fields }
  return Object.keys(defaults).map(key => [key, labels[key]])
}
export function claimContext(claim) {
  const role = roles[claim?.role]
  const subject = claim?.subject?.trim() || ''
  const generic = !subject || subject === role || ['本文方法/改动', '文章整体', '基础方法'].includes(subject)
  return role ? generic ? role : `${role} · ${subject}` : subject
}
export function reportScopeNote(report) {
  if (report?.coverage?.scope !== 'selected_pages') return ''
  const label = report.coverage.selection_label || `PDF 第 ${(report.coverage.selected_pages || []).join('、') || '选定'} 页`
  return `本报告仅分析${label}，不代表全文分析。`
}
