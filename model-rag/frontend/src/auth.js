import { ref } from 'vue'

export const account = ref(null)

export function csrfHeaders(method='GET',cookies=typeof document==='undefined'?'':document.cookie){
  if (['GET','HEAD','OPTIONS'].includes(method.toUpperCase())) return {}
  const item = cookies.split(';').map(value=>value.trim()).find(value=>value.startsWith('atlas_csrf='))
  return item ? {'X-CSRF-Token':decodeURIComponent(item.slice('atlas_csrf='.length))} : {}
}

export function userStorageKey(name){
  return `${name}:${account.value?.id || 'anonymous'}`
}

export function safeNext(value,origin=typeof window==='undefined'?'http://localhost':window.location.origin){
  if (typeof value!=='string') return '/'
  try {
    const url = new URL(value,origin)
    return url.origin===origin && !['/login','/register'].includes(url.pathname) ? url.pathname+url.search+url.hash : '/'
  } catch { return '/' }
}

export function sessionExpired(){
  account.value = null
  if (typeof window!=='undefined') window.dispatchEvent(new CustomEvent('atlas-unauthorized'))
}
