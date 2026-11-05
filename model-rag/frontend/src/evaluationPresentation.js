export const relevanceOptions = [
  { value: 'unjudged', label: '待判断' }, { value: 'direct', label: '直接相关' },
  { value: 'supporting', label: '辅助相关' }, { value: 'irrelevant', label: '无关' },
]
export const answerDimensions = [
  { key: 'correctness', label: '事实正确', help: '结论与参考资料一致' },
  { key: 'completeness', label: '回答完整', help: '覆盖问题的各个要求' },
  { key: 'faithfulness', label: '引用支持', help: '关键结论由实际引用正文支持' },
  { key: 'conditions', label: '条件明确', help: '版本、布局、参数及边界说明正确' },
]
export function metricText(value, unavailable = '待标注') {
  return value == null ? unavailable : `${(value * 100).toFixed(1)}%`
}
export function judgmentFor(row, sectionId) {
  return row.judgments?.[sectionId] || { label: 'unjudged', origin: null, note: '' }
}
export function filteredCases(cases, split) {
  return split === 'all' ? cases : cases.filter(item => (item.split || 'development') === split)
}
export function precisionStatus(row) {
  if (!row.metrics) return row.error ? '执行异常' : '未评测'
  if (!row.metrics.retrieved) return '未返回证据'
  return row.metrics.unjudged ? `待判断 ${row.metrics.unjudged} 条` : `实际精确率 ${metricText(row.metrics.returned_precision)}`
}
