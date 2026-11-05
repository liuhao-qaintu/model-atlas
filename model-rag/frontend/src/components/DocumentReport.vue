<script setup>
import { computed, inject, ref, watch } from 'vue'
import { api } from '../api'
import { claimContext, reportFields, reportScopeNote } from '../reportPresentation'

const props = defineProps({ report: Object, document: Object })
const emit = defineEmits(['confirmed'])
const developerMode = inject('developerMode', ref(false))
const dialog = ref(null), selected = ref([]), busy = ref(false), error = ref('')
const fields = computed(() => reportFields(props.report))
const approved = computed(() => props.report?.status === 'approved')
const visibleTags = computed(() => approved.value ? (props.report.confirmed_tags || []) : props.report.tags)
watch(() => props.report?.id, () => { selected.value = props.report?.selected_tag_ids || (props.report?.tags || []).map(t => t.id); error.value = '' }, { immediate: true })
function fileUrl(proof) { return `/api/documents/${props.document.id}/file?inline=true${proof.page ? `#page=${proof.page}` : ''}` }
function locations(claim) {
  const seen = new Set()
  return (claim.proofs || []).filter(proof => {
    const key = `${proof.chapter || ''}:${fileUrl(proof)}`
    if (seen.has(key)) return false
    seen.add(key); return true
  })
}
function proofLabel(proof) {
  const chapter = /^\s*(\d+(?:\.\d+)*)\.?\s/.exec(proof.chapter || proof.locator || '')
  const position = chapter ? `§${chapter[1]}` : '前置信息'
  const pages = proof.pages?.length ? proof.pages : proof.page ? [proof.page] : []
  const page = pages.length ? `PDF 第 ${pages[0]}${pages.at(-1) !== pages[0] ? `–${pages.at(-1)}` : ''} 页` : '页码未识别'
  const paragraph = proof.paragraph_start && proof.paragraph_end - proof.paragraph_start <= 2 ? ` · 解析第 ${proof.paragraph_start}${proof.paragraph_end !== proof.paragraph_start ? `–${proof.paragraph_end}` : ''} 段` : ''
  return `${position} · ${page}${paragraph}`
}
function openTags() { error.value = ''; dialog.value.showModal() }
async function confirm() {
  if (busy.value) return
  busy.value = true; error.value = ''
  try {
    const value = await api(`/ingest/reports/${props.report.id}/approve`, { tag_ids: selected.value })
    dialog.value.close(); emit('confirmed', value)
  } catch (e) { error.value = e.message }
  finally { busy.value = false }
}
</script>

<template>
  <article v-if="report" class="paper-report">
    <header class="report-heading"><div><span class="eyebrow">文献分析</span><h2>{{ report.document_type }}阅读报告</h2></div><span class="report-state">{{ approved ? '标签已确认' : report.content_review ? 'AI 草稿与来源校订 · 待确认标签' : 'AI 分析 · 待确认标签' }}</span></header>
    <p class="report-description">依据已解析正文整理，结论附原文定位。{{ report.document_type === '综述' ? '本文对已有研究进行归纳，所述模型并非均由本文作者提出。' : '' }}</p>
    <p v-if="reportScopeNote(report)" class="notice compact" role="status">{{ reportScopeNote(report) }}</p>
    <div class="report-tags"><span v-for="tag in visibleTags.slice(0, 8)" :key="tag.id"><small>{{ tag.kind }}</small>{{ tag.name }}</span><button class="button" @click="openTags">{{ approved ? '查看已确认标签' : `检查全部 ${report.tags.length} 个标签与关系` }}</button></div>
    <div class="report-sections"><section v-for="[field, label] in fields" :key="field" :class="['report-section', `report-${field}`]"><h3>{{ label }}</h3>
      <p v-if="!report.content[field]?.length" class="report-empty">解析正文未提供足够依据，暂不作结论。</p>
      <article v-for="(claim, index) in report.content[field]" :key="index" class="report-claim"><span v-if="claimContext(claim)" class="claim-context">{{ claimContext(claim) }}</span><p>{{ claim.text }}</p><div class="report-citations"><a v-for="(proof, p) in locations(claim)" :key="p" :href="fileUrl(proof)" target="_blank" rel="noopener noreferrer" :title="proof.quote">{{ proofLabel(proof) }} ↗</a></div><details v-if="claim.proofs?.length" class="claim-original"><summary>核对支撑原句</summary><blockquote v-for="(proof, p) in claim.proofs" :key="p">{{ proof.quote }}</blockquote></details></article>
    </section></div>
    <details class="report-notes"><summary>分析范围与说明</summary><p v-if="report.content_review">{{ report.content_review.note }}</p><p v-for="limit in report.limits" :key="limit">{{ limit }}</p><p v-if="developerMode">{{ report.generated_by }} · {{ report.coverage?.llm_calls }} 次模型调用 · {{ report.policy }}</p></details>
    <dialog ref="dialog" class="tag-dialog" aria-labelledby="tag-dialog-title"><header><div><span class="eyebrow">集中管理</span><h2 id="tag-dialog-title">文献标签与图谱关系</h2></div><button class="button" @click="dialog.close()">关闭</button></header>
      <p>一行对应一个有原文依据的标签，无需逐个确认后台切片。已有模型仅按明确别名关联，其余建立主题标签。</p>
      <div v-if="!approved" class="tag-toolbar"><button class="button" @click="selected = report.tags.map(t => t.id)">全选</button><button class="button" @click="selected = []">清空选择</button><small>已选择 {{ selected.length }} / {{ report.tags.length }}</small></div>
      <div class="tag-table-scroll"><table><thead><tr><th>{{ approved ? "状态" : "选择" }}</th><th>类别</th><th>标签</th><th>图谱关联</th><th>依据位置</th></tr></thead><tbody><tr v-for="tag in report.tags" :key="tag.id"><td v-if="approved">{{ report.selected_tag_ids?.includes(tag.id) ? "已加入" : "未选择" }}</td><td v-else><input v-model="selected" type="checkbox" :value="tag.id" :aria-label="`选择标签 ${tag.name}`" /></td><td>{{ tag.kind }}</td><td><strong>{{ tag.name }}</strong></td><td>{{ tag.target_name ? `关联已有：${tag.target_name}` : '文献 → 主题标签' }}</td><td><details><summary>{{ proofLabel(tag.proofs[0]) }}</summary><p class="tag-quote">{{ tag.proofs[0]?.quote }}</p><a :href="fileUrl(tag.proofs[0])" target="_blank" rel="noopener noreferrer">打开原文件 ↗</a></details></td></tr></tbody></table></div>
      <p v-if="error" class="notice error">{{ error }}</p><footer><small>仅确认标签及其有依据的文献关系，不将“提及”升级为“提出”或“采用”。</small><button v-if="!approved" class="button primary" :disabled="busy" @click="confirm">{{ busy ? '保存中…' : '确认所选标签并加入图谱' }}</button></footer>
    </dialog>
  </article>
</template>

<style scoped>
.claim-context{display:inline-block;font-size:10px;color:#5c70a4;background:#eef3fb;border-radius:4px;padding:4px 7px;margin-bottom:7px;line-height:1.6}.claim-original{font-size:11px;color:var(--muted);margin-top:9px}.claim-original summary{cursor:pointer}.claim-original blockquote{font-size:12px;line-height:1.8;margin:10px 0 14px;padding-left:12px;border-left:2px solid #d7e1f0;white-space:pre-wrap;overflow-wrap:anywhere}
.paper-report{margin-top:22px;border:1px solid var(--line);border-radius:16px;padding:28px;background:var(--paper)}.report-heading,.tag-dialog header{display:flex;justify-content:space-between;align-items:center;gap:14px}.report-heading h2,.tag-dialog h2{margin:8px 0;font-size:23px}.report-state{font-size:12px;padding:7px 10px;border-radius:20px;background:#f5f7fc;color:var(--green)}.report-description,.report-notes,.tag-dialog>p{font-size:13px;color:var(--muted);line-height:1.9}.report-tags{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin:20px 0 28px}.report-tags>span{border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:13px}.report-tags small{margin-right:8px;color:var(--muted);font-size:10px}.report-tags .button{margin-left:auto}.report-sections{display:grid;grid-template-columns:1fr 1fr;gap:20px}.report-section{min-width:0;border-top:1px solid var(--line);padding-top:17px}.report-summary,.report-methods{grid-column:1/-1}.report-section h3{font-size:16px;margin:0 0 15px}.report-claim{margin:0 0 18px}.report-claim>p{font-size:14px;line-height:1.95;margin:0 0 6px;white-space:pre-line}.report-citations{display:flex;flex-wrap:wrap;gap:5px 12px}.report-citations a{font-size:11px;color:var(--muted);line-height:1.7;overflow-wrap:anywhere}.report-empty{font-size:13px;color:var(--muted)}.report-notes{border-top:1px solid var(--line);padding-top:15px}.report-notes summary{cursor:pointer}.tag-dialog{width:min(1120px,94vw);max-height:85vh;box-sizing:border-box;border:1px solid var(--line);border-radius:16px;padding:24px;color:#26354f;background:var(--paper)}.tag-dialog::backdrop{background:rgb(20 33 60 / .45)}.tag-toolbar{display:flex;gap:10px;align-items:center;margin:14px 0}.tag-table-scroll{max-height:48vh;overflow:auto;border:1px solid var(--line);border-radius:8px}.tag-dialog table{width:100%;border-collapse:collapse;font-size:12px;min-width:620px}.tag-dialog th,.tag-dialog td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}.tag-dialog th{background:#f5f7fc;position:sticky;top:0}.tag-dialog input[type=checkbox]{width:16px;height:16px}.tag-dialog td:nth-last-child(1){width:42%}.tag-quote{max-width:430px;white-space:normal;line-height:1.8;overflow-wrap:anywhere;color:var(--muted)}.tag-dialog summary{cursor:pointer;line-height:1.7}.tag-dialog footer{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-top:18px}.tag-dialog footer small{color:var(--muted);max-width:500px;line-height:1.7}@media(max-width:720px){.paper-report{padding:18px}.report-sections{grid-template-columns:1fr}.report-section{grid-column:auto}.report-heading{align-items:flex-start;flex-direction:column}.report-tags .button{margin-left:0}.tag-dialog{padding:16px}.tag-dialog footer{align-items:stretch;flex-direction:column}}

.tag-dialog[open]{display:flex;flex-direction:column;height:min(760px,90vh);max-height:90vh;overflow:hidden}.tag-dialog>header,.tag-dialog>p,.tag-toolbar,.tag-dialog>footer{flex-shrink:0}.tag-dialog .tag-table-scroll{flex:1;min-height:70px;max-height:none}.tag-dialog>p{margin:8px 0}.tag-dialog>footer{margin-top:12px}
</style>
