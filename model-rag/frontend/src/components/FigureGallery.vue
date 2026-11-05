<script setup>
import { nextTick, ref } from 'vue'
defineProps({ figures: { type: Array, default: () => [] } })
const active = ref(null), dialog = ref(null)
let returnFocus = null
async function show(figure) {
  returnFocus = document.activeElement
  active.value = figure
  await nextTick()
  dialog.value.showModal()
}
function close() {
  active.value = null
  returnFocus?.focus()
}
</script>

<template>
  <div v-if="figures.length" class="figure-gallery">
    <figure v-for="figure in figures" :key="figure.id" class="paper-figure">
      <button class="figure-preview" :aria-label="`放大：${figure.title}`" @click="show(figure)"><img :src="figure.image_url" :alt="figure.alt" loading="lazy" /></button>
      <figcaption><h3>{{ figure.title }}</h3><p>{{ figure.guide }}</p><small>{{ figure.source_version }} · {{ figure.figure_label }} · 原PDF第{{ figure.physical_page }}页</small><a :href="figure.source_url" target="_blank" rel="noopener noreferrer">阅读原论文 ↗</a><small>{{ figure.credit }}</small></figcaption>
    </figure>
  </div>
  <dialog ref="dialog" class="figure-dialog" :aria-label="active?.title || '原论文插图'" @close="close">
    <template v-if="active"><div class="page-top"><h2>{{ active.title }}</h2><button class="button" @click="dialog.close()">关闭</button></div><img :src="active.image_url" :alt="active.alt" /><p>{{ active.guide }}</p><a :href="active.source_url" target="_blank" rel="noopener noreferrer">{{ active.figure_label }} · 原PDF第{{ active.physical_page }}页 ↗</a></template>
  </dialog>
</template>

<style scoped>
.figure-gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr)); gap:18px; margin:20px 0; }
.paper-figure { margin:0; padding:18px; background:#fff; border:1px solid var(--line); border-radius:12px; }
.figure-preview { display:block; width:100%; padding:0; background:white; cursor:zoom-in; border-radius:6px; }
.figure-preview img { display:block; width:100%; height:300px; object-fit:contain; }
figcaption h3 { font-size:16px; margin:16px 0 8px; }
figcaption p { color:#475569; font-size:13px; line-height:1.8; }
figcaption small { display:block; margin:8px 0; color:#64748b; font-size:11px; overflow-wrap:anywhere; }
figcaption a { font-size:12px; color:#1d4ed8; }
.figure-dialog { max-width:min(1100px,94vw); width:max-content; padding:24px; border:1px solid var(--line); border-radius:14px; }
.figure-dialog::backdrop { background:rgb(15 23 42 / .5); }
.figure-dialog img { max-width:100%; max-height:70vh; display:block; margin:16px auto; object-fit:contain; }
.figure-dialog p { max-width:800px; font-size:14px; line-height:1.8; }
.figure-dialog a { color:#1d4ed8; font-size:13px; }
</style>
