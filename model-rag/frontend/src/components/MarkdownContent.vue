<script setup>
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import { renderMarkdown } from '../markdownRendering.js'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github.css'

const props = defineProps({ content: { type: String, default: '' }, citationTargets: { type: Array, default: () => [] } })
const emit = defineEmits(['citation'])
function openCitation(event) {
  const link = event.target.closest?.('a[data-citation]')
  if (!link || !event.currentTarget.contains(link)) return
  event.preventDefault()
  emit('citation', Number(link.dataset.citation))
}
const html = computed(() => {
  const clean = DOMPurify.sanitize(renderMarkdown(props.content, props.citationTargets), { USE_PROFILES: { html: true, mathMl: true, svg: true } })
  const template = document.createElement('template')
  template.innerHTML = clean
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT)
  let node
  while ((node = walker.nextNode())) {
    if (!node.parentElement?.closest('pre,code,.katex')) node.textContent = node.textContent.replace(/(?:^|\n)(?:S\d+|E\d+|A\d+)\s*[:：]\s*/g, '')
  }
  template.content.querySelectorAll('a').forEach(link => {
    const href = link.getAttribute('href') || ''
    if (/^https?:\/\//.test(href)) { link.target = '_blank'; link.rel = 'noopener noreferrer' }
    else if (/\.md(?:#.*)?$/.test(href)) { link.removeAttribute('href'); link.title = '请通过本节来源或回答引用查看对应资料' }
  })
  return template.innerHTML
})
</script>

<template><div class="markdown" @click="openCitation" v-html="html" /></template>
