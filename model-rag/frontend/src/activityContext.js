import { userStorageKey } from './auth.js'
const key = 'model-atlas-reading-context'
export function recentModels() {
  try { return JSON.parse(localStorage.getItem(userStorageKey(key)) || '[]').filter(item => typeof item.model_id === 'string' && ['model', 'graph'].includes(item.action)).slice(0, 3) }
  catch { return [] }
}
export function rememberModel(modelId, action = 'model') {
  if (!modelId) return
  try { localStorage.setItem(userStorageKey(key), JSON.stringify([{ model_id: modelId, action }, ...recentModels().filter(item => item.model_id !== modelId)].slice(0, 3))) } catch {}
}
