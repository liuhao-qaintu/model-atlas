// 只修正自动识别的裸网址；显式 Markdown 链接保留作者指定的地址。
export function renderBareLink(token) {
  if (token.raw !== token.text || !/^https?:\/\//.test(token.href)) return false
  const boundary = token.href.search(/[。，；！？]/u)
  if (boundary < 0) return false
  const escape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;')
  const url = token.href.slice(0, boundary)
  return `<a href="${escape(url)}">${escape(url)}</a>${escape(token.text.slice(boundary))}`
}
