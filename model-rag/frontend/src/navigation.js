export function scrollPosition(to, from, savedPosition) {
  if (savedPosition) return savedPosition
  if (to.path === from.path) return false
  return { top: 0 }
}
