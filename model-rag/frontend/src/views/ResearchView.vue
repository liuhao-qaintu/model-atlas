<script setup>
import { computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, researchStream } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'
import DocumentPicker from '../components/DocumentPicker.vue'
import { shouldSendOnEnter } from '../composerKeys'
import LearningTrail from '../components/LearningTrail.vue'

const route = useRoute(), router = useRouter()
const developerMode = inject('developerMode', ref(false))
const question = ref(''), documentIds = ref([]), documents = ref([]), result = ref(null)
async function refreshDocuments() { documents.value = await api('/documents') }
function researchKeydown(event) { if (shouldSendOnEnter(event)) { event.preventDefault(); if (!restoring.value) submit() } }
const busy = ref(false), restoring = ref(false), error = ref(''), stages = ref([]), evidencePanel = ref(null)
const continuing = ref(false)
const flowQuery = () => route.query.flow === 'ae' ? { flow: 'ae' } : {}
function applyDraft() {
  if (!busy.value && !route.query.run && typeof route.query.question === 'string') {
    question.value = route.query.question.slice(0, 1500); result.value = null
    const requested = Array.isArray(route.query.document) ? route.query.document : [route.query.document]
    documentIds.value = [...new Set(requested)].filter(id => documents.value.some(d => d.id === id && d.cloud_allowed)).slice(0, 3)
  }
}
async function continueChat() {
  if (!result.value || continuing.value) return
  continuing.value = true; error.value = ''
  try {
    const value = await api(`/research/runs/${encodeURIComponent(result.value.id)}/conversation`, {})
    await router.push({ path: '/chat', query: { ...flowQuery(), session: value.session_id } })
  } catch (e) { error.value = e.message }
  finally { continuing.value = false }
}
let controller = null
const examples = [
  'CLIP 和 LLaVA 的输入、训练目标、输出及用途有什么区别？',
  'ResNet-18 与 ViT 如何处理图像？比较输入表示、核心结构和输出。',
  'Transformer、ViT 和 DETR 分别怎样处理位置信息？请区分原理与具体实现。',
]
const reason = computed(() => ({ evidence_sufficient: '本轮资料已支持各项查证内容', evidence_insufficient: '部分内容仍缺少足够依据',
  budget_reached: '已达到本轮查找上限，保留未覆盖内容', no_new_evidence: '补查没有找到新的依据', repeated_actions: '未继续重复查找' })[result.value?.stop_reason] || '')
const gaps = computed(() => (result.value?.checks || []).filter(c => c.status !== 'supported'))
const targets = computed(() => (result.value?.evidence || []).map((_, i) => `#research-evidence-${i + 1}`))
async function cite(number) {
  if (!evidencePanel.value) return
  evidencePanel.value.open = true
  await nextTick()
  const target = document.getElementById(`research-evidence-${number}`)
  target?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  target?.focus({ preventScroll: true })
}
async function submit() {
  if (busy.value || !question.value.trim()) return
  busy.value = true; result.value = null; error.value = ''; stages.value = []
  controller = new AbortController()
  await router.replace({ path: '/research', query: flowQuery() })
  try {
    const data = await researchStream({ question: question.value.trim(), document_ids: documentIds.value }, (name, data) => {
      if (name === 'stage') stages.value.push(data)
    }, controller.signal)
    result.value = data
    await router.replace({ path: '/research', query: { ...flowQuery(), run: data.id } })
  } catch (e) {
    error.value = e.name === 'AbortError' ? '已停止本轮研究，没有把未完成内容保存成成功报告。' : e.message
  } finally { busy.value = false; controller = null }
}
function download() {
  if (!result.value) return
  const sourceLines = result.value.evidence.map((e, i) => `[${i + 1}] ${e.title}\n${e.locator || ''}\n${e.excerpt}`).join('\n\n')
  const blob = new Blob([`# ${result.value.question}\n\n${result.value.answer}\n\n## 本轮依据\n\n${sourceLines}\n\n记录：${result.value.id}\n`], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = `${result.value.id}.md`; link.click(); URL.revokeObjectURL(url)
}
onMounted(async () => {
  try {
    await refreshDocuments()
    applyDraft()
    if (typeof route.query.run === 'string') {
      restoring.value = true
      result.value = await api(`/research/runs/${encodeURIComponent(route.query.run)}`)
      question.value = result.value.question
      documentIds.value = result.value.document_ids || []
    }
  } catch (e) { error.value = e.message }
  finally { restoring.value = false }
})
onBeforeUnmount(() => controller?.abort())
watch(() => route.query.question, applyDraft)
</script>

<template>
  <section class="research-page">
    <LearningTrail v-if="route.query.flow === 'ae'" stage="compare" />
    <header class="research-heading"><div><span class="eyebrow">把分散的资料，整理成有依据的认识</span><h1>比较研究</h1><p>对照模型与文献，分清差异、联系和仍需核实的内容。</p></div><span class="research-tag">查找 · 核对 · 整理</span></header>
    <form class="research-form" @submit.prevent="submit">
      <label for="research-question">你想弄清楚什么？</label>
      <textarea id="research-question" v-model="question" rows="3" maxlength="1500" :disabled="busy || restoring" placeholder="例如：CLIP 和 LLaVA 的输入、训练目标、输出及用途有什么区别？" @keydown="researchKeydown" />
      <div class="research-options"><DocumentPicker v-model="documentIds" :documents="documents" :disabled="busy || restoring" @refresh="refreshDocuments" /><button v-if="busy" type="button" class="button" @click="controller?.abort()">停止</button><button v-else class="button primary" :disabled="!question.trim() || restoring">开始研究 →</button></div>
      <p class="scope-hint">{{ documentIds.length ? `本轮查找所选 ${documentIds.length} 篇文献` : '本轮查找全部允许模型处理的知识库资料' }} · Enter 开始 · Shift + Enter 换行</p>
      <small>问题与本轮证据会发送到已配置的模型。仅使用允许云端处理的资料；本地资料可在“文档导入与阅读”查看。</small>
    </form>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <div v-if="busy" class="research-progress" role="status"><span class="spinner" /><strong>{{ stages.at(-1)?.label || '正在开始…' }}</strong><small>可随时停止；完成后可通过引用核对原文。</small></div>
    <div v-if="!result && !busy && !restoring" class="research-examples"><span class="eyebrow">试一个比较问题</span><button v-for="(example, i) in examples" :key="example" @click="question = example"><small>0{{ i + 1 }}</small><span>{{ example }}</span><b aria-hidden="true">↗</b></button></div>
    <p v-if="restoring" class="notice">正在打开已保存的研究结果…</p>
    <article v-if="result" class="research-result">
      <header><div><span class="eyebrow">研究整理</span><h2>{{ result.question }}</h2></div><button class="button" @click="download">导出 Markdown</button></header>
      <div class="research-continue"><div><strong>还有没看明白的地方？</strong><p>带着本次比较结果和引用继续问，随时回到原文核对。</p></div><button class="button primary" :disabled="continuing" @click="continueChat">{{ continuing ? '正在接入对话…' : '继续追问 →' }}</button></div>
      <p class="research-outcome">{{ reason }}<small>证据判断由模型辅助完成，原文可核对，不代表独立事实认证。</small></p>
      <MarkdownContent :content="result.answer" :citation-targets="targets" @citation="cite" />
      <details v-if="gaps.length" class="research-gaps" open><summary>仍需核实的内容 · {{ gaps.length }} 项</summary><p v-for="gap in gaps" :key="gap.unit_id"><b>{{ result.units.find(u => u.id === gap.unit_id)?.label }}</b> · {{ result.outline.model_ids.includes(gap.unit_id.split('|')[0]) ? gap.unit_id.split('|')[0] : '' }}：{{ gap.note }}</p></details>
      <details class="research-outline"><summary>本次查证范围</summary><p>{{ result.outline.dimensions.map(d => d.label).join(' · ') }}</p><p>共 {{ result.units.length }} 项内容；没有依据的项目会保留缺口，不自动改成确定结论。</p></details>
      <details ref="evidencePanel" class="research-sources"><summary>查看引用原文 · {{ result.evidence.length }} 个片段</summary><article v-for="(evidence, i) in result.evidence" :id="`research-evidence-${i + 1}`" :key="evidence.section_id" tabindex="-1"><RouterLink :to="evidence.reader_url">[{{ i + 1 }}] {{ evidence.title }} →</RouterLink><small>{{ evidence.source_type }}{{ evidence.locator ? ` · ${evidence.locator}` : '' }}</small><pre class="research-excerpt">{{ evidence.excerpt }}</pre></article></details>
      <details v-if="developerMode" class="research-debug"><summary>开发者记录</summary><p>{{ result.protocol }} · {{ result.rounds }} 轮资料查找 · {{ result.tool_calls }} 次工具调用 · {{ result.model_calls }} 次研究模型调用</p><p>停止原因：{{ result.stop_reason }}；模型辅助判断，不是评测准确率。</p><pre>{{ JSON.stringify({ outline: result.outline, checks: result.checks, trace: result.trace, diagnostics: result.diagnostics }, null, 2) }}</pre></details>
    </article>
  </section>
</template>

<style scoped>
.research-continue{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:18px 20px;background:#f2f6fc;border:1px solid #e0e7f4;border-radius:10px;margin:20px 0}.research-continue strong{font-size:14px}.research-continue p{font-size:12px;line-height:1.8;color:var(--muted);margin:6px 0 0}.research-continue .button{flex:none}@media(max-width:650px){.research-continue{align-items:start;flex-direction:column}}
.research-page { max-width:1100px; margin:auto; }
.research-heading { display:flex; justify-content:space-between; gap:24px; align-items:center; margin:8px 0 28px; }
.research-heading .eyebrow { color:#6c7eaa; font-size:11px; }.research-heading h1 { font-size:39px; margin:13px 0; }.research-heading p { color:var(--muted); font-size:13px; margin:0; line-height:1.9; }
.research-tag { flex:none; padding:8px 12px; border:1px solid #dce4f1; border-radius:7px; color:#6e80a0; font-size:11px; }
.research-form { background:#fff; border:1px solid #dce4f0; border-radius:16px; padding:24px; box-shadow:0 6px 25px #2e497b06; }
.research-form>label { display:block; color:#405679; font-weight:600; font-size:14px; margin-bottom:14px; }
.research-form textarea { width:100%; border:0; resize:vertical; min-height:90px; background:transparent; font:inherit; color:#344969; line-height:1.9; }
.research-options { display:flex; gap:16px; justify-content:space-between; align-items:end; padding-top:16px; border-top:1px solid var(--line); margin:12px 0; }
.research-options label { display:grid; grid-template-columns:minmax(0,1fr); gap:6px; color:var(--muted); font-size:11px; min-width:0; max-width:70%; }.research-options select { width:100%; min-width:0; max-width:100%; font:inherit; padding:8px 12px; border:1px solid var(--line); border-radius:7px; color:#556c93; background:#f8faff; }
.research-form>small { font-size:10px; color:#77859d; display:block; line-height:1.9; }.research-page>.notice { margin-top:18px; }
.research-examples { display:grid; gap:12px; margin:28px 0; }.research-examples button { display:flex; align-items:center; gap:16px; padding:18px 22px; border:1px solid var(--line); border-radius:11px; background:#fff; color:#4d6080; text-align:left; font-size:13px; }.research-examples button:hover { border-color:#a7b9dc; }.research-examples small { color:#879cbe; font-family:var(--font-display); }.research-examples b { margin-left:auto; font-weight:400; }
.research-progress { display:flex; align-items:center; flex-wrap:wrap; gap:12px; padding:20px; margin:22px 0; background:#edf3fc; border-radius:12px; color:#57759d; font-size:13px; }.research-progress small { width:100%; margin-left:25px; color:#6e829f; font-size:11px; }
.research-result { padding:28px; background:white; border:1px solid var(--line); border-radius:15px; margin-top:28px; min-width:0; }.research-result>header { display:flex; justify-content:space-between; gap:20px; align-items:center; }.research-result h2 { font-size:23px; margin:8px 0 18px; line-height:1.6; }.research-result>header .button { flex:none; }
.research-outcome { padding:13px 16px; background:#f0f6f5; border-left:3px solid #6da795; color:#4c776d; font-size:13px; margin:10px 0 24px; }.research-outcome small { display:block; color:#6e8582; font-size:10px; margin-top:5px; }
.research-result details { margin-top:22px; padding-top:16px; border-top:1px solid var(--line); font-size:12px; color:var(--muted); }.research-result summary { cursor:pointer; color:#546f9b; }.research-result details>p { line-height:1.9; margin:12px 0; }.research-gaps { background:#fffaf0; padding:16px; border-radius:8px; border:1px solid #efe3c7; }.research-sources>article { padding:20px 0; border-bottom:1px solid var(--line); scroll-margin-top:20px; }.research-sources>article>a { font-weight:600; font-size:13px; color:#4566a1; }.research-sources>article>small { display:block; margin:9px 0 16px; color:#8694aa; }.research-debug pre { max-height:420px; overflow:auto; font-size:11px; white-space:pre-wrap; overflow-wrap:anywhere; background:#f5f7fb; padding:14px; }
.research-excerpt { white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.85 var(--font-mono, monospace); max-height:360px; overflow:auto; background:#f7f9fc; padding:16px; border-radius:8px; color:#53647e; }
@media(max-width:760px) { .research-heading { display:block; }.research-tag { display:none; }.research-heading h1 { font-size:32px; }.research-form,.research-result { padding:18px; }.research-options { gap:10px; }.research-options label { max-width:65%; }.research-result>header { display:block; }.research-result>header .button { margin-bottom:10px; }.research-result h2 { font-size:20px; }.research-examples button { padding:16px; font-size:12px; } }
</style>
