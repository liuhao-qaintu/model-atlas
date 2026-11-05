export function comparisonPair(component) {
  const first = component.implementations[0]
  const second = component.implementations.find(item => item.model_name !== first?.model_name) || component.implementations[1]
  return [first?.edge_id || '', second?.edge_id || '']
}

export function comparisonRoute(component, pair) {
  const selected = pair.map(id => component.implementations.find(item => item.edge_id === id))
  if (selected.some(item => !item) || selected[0].edge_id === selected[1].edge_id) return null
  const objects = selected.map(item => item.model_name === item.entity ? item.entity : `${item.model_name} 中的 ${item.entity}`).join(' 与 ')
  return { path: '/chat', query: { model: '', question: `请比较 ${objects} 的${component.name}：代码、输入输出有什么相同与不同？请给出依据。` } }
}
