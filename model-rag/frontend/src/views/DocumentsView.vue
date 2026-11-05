<script setup>
import { computed, inject, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, deleteDocument, ingestStream, startIngest } from '../api'
import DocumentReport from '../components/DocumentReport.vue'
import IngestProgress from '../components/IngestProgress.vue'
import LearningTrail from '../components/LearningTrail.vue'
import { AE_PAPER_URL, aeLocation } from '../aeJourney'

const route = useRoute(), router = useRouter()
const developerMode = inject('developerMode', ref(false)), updateHealth = inject('updateHealth', () => {})
const models = ref([]), documents = ref([]), detail = ref(null), health = ref(null), report = ref(null)
const documentQuery = ref('')
const filteredDocuments = computed(() => {
  const query = documentQuery.value.trim().toLowerCase()
  return documents.value.filter(document => [document.title, document.filename, ...(document.analysis_tags || []).map(tag => tag.name)].join(' ').toLowerCase().includes(query))
})
const file = ref(null), modelId = ref('auto'), preview = ref(null), pageRange = ref('')
const title = ref(''), sourceUrl = ref(''), cloudAllowed = ref(false), documentNodeId = ref('')
const busy = ref(''), error = ref(''), success = ref(''), ingestJob = ref(null), ingestEvents = ref([])
let ingestController = null, pollTimer = null, unmounted = false
const modelsForSelect = computed(() => models.value.filter(m => !['文档', '知识标签'].includes(m.kind)))
const currentSection = computed(() => detail.value?.sections?.find(s => s.slug === route.query.part))
const chapters = computed(() => [...new Set((preview.value?.parts || []).map(p => p.chapter || p.title))])
const missingVectors = computed(() => Math.max(0, (health.value?.index?.eligible_chunks || 0) - (health.value?.index?.indexed_chunks || 0)))
const processing = computed(() => ['pending', 'running'].includes(ingestJob.value?.status))
async function loadList() {
  ;[models.value, documents.value, health.value] = await Promise.all([api('/models'), api('/documents'), api('/health')])
  updateHealth(health.value)
}
async function loadReport(id) {
  const value = await api(`/ingest/documents/${encodeURIComponent(id)}/report`)
  if (route.params.documentId !== id) return
  report.value = value.report
  if (!processing.value && value.job?.pipeline === 'document-report-v1') ingestJob.value = value.job
}
async function loadDetail() {
  detail.value = null; report.value = null; ingestJob.value = null; ingestEvents.value = []
  if (pollTimer) clearTimeout(pollTimer)
  if (!route.params.documentId) return
  const id = route.params.documentId, value = await api(`/documents/${encodeURIComponent(id)}`)
  if (route.params.documentId !== id) return
  detail.value = value; await loadReport(id)
  if (processing.value) pollJob(id)
}
async function pollJob(id) {
  if (unmounted || !processing.value || route.params.documentId !== id) return
  try {
    ingestJob.value = await api(`/ingest/jobs/${ingestJob.value.id}`)
    if (!processing.value) await loadReport(id)
    else pollTimer = setTimeout(() => pollJob(id), 2000)
  } catch { error.value = '原分析任务已不在运行，可重新生成报告。'; ingestJob.value = null }
}
function fileChanged(event) {
  file.value = event.target.files[0] || null; pageRange.value = ''; preview.value = null
  error.value = ''; success.value = ''; cloudAllowed.value = false; documentNodeId.value = ''
}
async function ensureDocumentNode() {
  if (modelId.value !== 'auto') return modelId.value
  if (documentNodeId.value) return documentNodeId.value
  const prefix = `文档 · ${file.value.name}`.slice(0, 60)
  const existing = models.value.find(m => m.kind === '文档' && m.name.startsWith(prefix))
  if (existing) { documentNodeId.value = existing.id; return existing.id }
  const value = await api('/models', { name: `${prefix} · ${Math.random().toString(36).slice(2, 8)}`, category: '文档', description: '独立文献，保留原文件及检索定位。', aliases: [], tags: [] })
  documentNodeId.value = value.id; return value.id
}
function showPreview(value) { preview.value = value; title.value = value.title; sourceUrl.value = value.source_url || ''; cloudAllowed.value = Boolean(value.cloud_allowed) }
async function parseFile() {
  if (!file.value || busy.value) return
  if (file.value.size > 20 * 1024 * 1024) { error.value = '文件大小上限为 20 MB。'; return }
  busy.value = 'parse'; error.value = ''; success.value = ''; preview.value = null
  try {
    const target = await ensureDocumentNode(), form = new FormData()
    form.append('file', file.value); form.append('model_id', target); form.append('pages', pageRange.value)
    showPreview(await api('/documents/preview', form))
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function importFile() {
  if (!preview.value || busy.value) return
  busy.value = 'import'; error.value = ''; success.value = ''
  const id = preview.value.id
  try {
    await api(`/documents/${id}/import`, { title: title.value, source_url: sourceUrl.value, cloud_allowed: cloudAllowed.value })
    const analyze = cloudAllowed.value
    preview.value = null; await loadList(); await router.push(`/documents/${id}`)
    success.value = '文献已入库。检索问答与标签确认独立，可先提问，再查看分析报告。'
    if (analyze) { startClassification(id); prepareIndex() }
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function reprocess(document) {
  if (busy.value) return
  busy.value = 'reprocess'; error.value = ''; success.value = ''
  try { const value = await api(`/documents/${document.id}/reprocess/preview`, {}); await router.push({ path: '/documents', query: { preview: value.id } }); showPreview(value) }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function changePriority(document, value) {
  if (busy.value) return
  busy.value = 'priority'; error.value = ''
  try { await api(`/documents/${document.id}/priority`, { priority: Number(value) }, { method: 'PATCH' }); await loadList(); success.value = '检索优先级已保存。' }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function setConsent(document, allowed) {
  if (busy.value) return
  if (allowed && !window.confirm('允许已配置的云端模型处理这篇文献的解析文字，用于分析报告、嵌入、重排和回答生成？')) return
  busy.value = 'consent'; error.value = ''
  try {
    await api(`/documents/${document.id}/cloud-consent`, { cloud_allowed: allowed }, { method: 'PATCH' })
    await loadList(); if (detail.value?.id === document.id) detail.value = { ...detail.value, cloud_allowed: allowed }
    success.value = allowed ? '已允许这篇文献用于云端分析与问答。' : '已改为仅本地处理，后续不发送云端模型。'
    if (allowed) prepareIndex()
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function startClassification(id) {
  if (processing.value) return
  ingestEvents.value = []; error.value = ''
  if (ingestController) ingestController.abort()
  try {
    ingestJob.value = await startIngest(id); ingestController = new AbortController()
    await ingestStream(ingestJob.value.id, event => {
      ingestEvents.value.push(event)
      if (event.type === 'job_started') ingestJob.value = { ...ingestJob.value, status: 'running' }
      if (event.type === 'progress') ingestJob.value = { ...ingestJob.value, progress: event.progress, total: event.total, stage: event.stage }
      if (['job_done', 'job_failed', 'job_cancelled', 'job_skipped'].includes(event.type)) {
        ingestJob.value = { ...ingestJob.value, status: event.status, error: event.error }
        loadReport(id).catch(e => { error.value = e.message })
      }
    }, ingestController.signal)
  } catch (e) { if (!unmounted && e.name !== 'AbortError') error.value = e.message }
}
function reportConfirmed(value) { report.value = value; success.value = '所选标签及文献关系已保存到图谱。'; loadList() }
async function removeDocument(document) {
  if (busy.value || !window.confirm(`删除《${document.title}》及其检索数据和关系？此操作不可恢复。`)) return
  busy.value = 'delete'; error.value = ''
  try { await deleteDocument(document.id); if (route.params.documentId === document.id) await router.push('/documents'); await loadList() }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function prepareIndex() {
  try { await api('/documents/index', {}); await loadList() }
  catch { success.value = '文献已保存，向量准备未完成；仍可使用关键词检索。可在管理中重试准备索引。' }
}
async function refreshCards() { busy.value = 'refresh'; try { await api('/knowledge/refresh', {}); await loadList() } catch (e) { error.value = e.message } finally { busy.value = '' } }
onMounted(async () => {
  try {
    await loadList(); await loadDetail()
    if (typeof route.query.preview === 'string' && !route.params.documentId) { const value = await api(`/documents/${encodeURIComponent(route.query.preview)}`); showPreview({ ...value, duplicate: value.status === 'imported' }) }
  } catch (e) { error.value = e.message }
})
watch(() => route.params.documentId, async () => { try { await loadDetail() } catch (e) { error.value = e.message } })
onBeforeUnmount(() => { unmounted = true; if (ingestController) ingestController.abort(); if (pollTimer) clearTimeout(pollTimer) })
</script>

<template>
  <section>
    <LearningTrail v-if="route.query.flow === 'ae'" stage="paper" />
    <div class="page-top"><span class="eyebrow">个人文献工作台</span><RouterLink v-if="detail" to="/documents" class="back-link">← 我的文献</RouterLink><span v-else class="soft-label">{{ documents.length }} 份文献</span></div>
    <h1>{{ route.params.documentId ? (detail?.title || '读取文献…') : '我的文献' }}</h1>
    <p v-if="!route.params.documentId" class="page-intro">管理你上传或导入的资料，阅读分析报告，围绕原文继续提问。</p>
    <p v-if="error" class="notice error">{{ error }}</p><p v-if="success" class="notice">{{ success }}</p>
    <template v-if="!route.params.documentId">
      <details class="document-upload" :open="route.query.upload === '1' || !!file || !!preview"><summary>导入新文献 <small>PDF / Markdown / TXT</small></summary>
      <form class="document-import" @submit.prevent="parseFile"><div class="document-form-grid"><label>选择文献<input type="file" accept=".pdf,.md,.txt" :disabled="!!busy" aria-label="上传资料文件" @change="fileChanged" /></label><label>收录方式<select v-model="modelId" :disabled="!!busy" aria-label="资料关联模型" @change="preview = null"><option value="auto">独立文献（推荐）</option><option v-for="model in modelsForSelect" :key="model.id" :value="model.id">关联：{{ model.name }}</option></select></label></div>
        <label v-if="file?.name.toLowerCase().endsWith('.pdf')" class="subheading">分析页码（可选）<input v-model="pageRange" :disabled="!!busy" maxlength="500" placeholder="例如 3-6,8；留空收录全文" aria-label="导入 PDF 页码" @input="preview = null" /></label><div class="document-actions"><small>PDF / MD / UTF-8 TXT · 最多 20 MB、200 页 · 原文件保存在本机</small><button class="button primary" :disabled="!file || !!busy">{{ busy === 'parse' ? '读取中…' : '读取文献' }}</button></div>
      </form></details>
      <article v-if="preview" class="document-preview">
        <div class="page-top"><h2>{{ preview.revision_of ? '更新文献解析' : '确认收录信息' }}</h2><span class="soft-label">{{ preview.page_count ? `${preview.page_count} 页` : preview.format.toUpperCase() }}</span></div>
        <p v-if="preview.revision_of" class="notice compact">确认后启用新版解析，旧版本保留供历史引用；新的分析报告单独生成。</p><p v-if="preview.duplicate" class="notice">这份文献已收录，无需重复上传。</p>
        <div v-if="!preview.duplicate" class="document-form-grid"><label>文献标题<input v-model="title" maxlength="180" aria-label="资料标题" /></label><label>原始来源网址（可选）<input v-model="sourceUrl" type="url" maxlength="2000" placeholder="论文主页 / DOI 链接" aria-label="原始来源网址" /></label></div>
        <details class="document-info"><summary>已识别的章节与解析说明</summary><ul><li v-for="chapter in chapters" :key="chapter">{{ chapter }}</li></ul><p v-for="warning in preview.warnings" :key="warning">{{ warning }}</p><p v-if="developerMode">后台检索单元：{{ preview.parts.length }}；{{ preview.char_count }} 字符。</p></details>
        <label v-if="!preview.duplicate" class="document-consent"><input v-model="cloudAllowed" type="checkbox" />允许已配置的云端模型处理本文的解析文字，用于分析报告、嵌入、重排和回答生成。</label><p v-if="!preview.duplicate && !cloudAllowed" class="document-hint">仅本地收录时不调用大模型，也不生成 AI 分析报告。以后可在文献管理中修改许可。</p>
        <div class="document-actions"><RouterLink v-if="preview.duplicate" :to="`/documents/${preview.id}`" class="button primary">打开文献 →</RouterLink><button v-else class="button primary" :disabled="!!busy || !title.trim()" @click="importFile">{{ busy === 'import' ? '保存中…' : cloudAllowed ? '收录并生成分析报告' : '仅在本地收录' }}</button><button class="button" @click="preview = null; router.replace('/documents')">收起</button></div>
      </article>
      <input v-if="documents.length" v-model="documentQuery" class="search-input wide" placeholder="搜索标题、文件名或标签" aria-label="搜索我的文献" />
      <p v-if="!documents.length" class="notice">还没有个人文献。上传一篇想阅读的论文，从这里开始整理。</p>
      <p v-else-if="!filteredDocuments.length" class="notice">没有找到匹配的文献，请调整搜索词。</p>
      <div class="sources-list"><article v-for="document in filteredDocuments" :key="document.id" class="source-card"><span class="source-id">{{ document.format.toUpperCase() }}</span><div class="document-summary"><h2>{{ document.title }}</h2><p>{{ document.page_count ? `${document.page_count} 页 · ` : '' }}第 {{ document.revision_number || 1 }} 版 · {{ document.cloud_allowed ? '可分析与问答' : '仅本地收录' }}</p><div class="document-tag-preview"><span v-for="tag in (document.analysis_tags || []).slice(0, 5)" :key="tag.id">{{ tag.name }}</span></div></div><div class="document-controls"><RouterLink :to="`/documents/${document.id}`" class="button primary">查看文献 →</RouterLink><details class="document-settings"><summary>管理</summary><label>检索优先级<select :value="document.retrieval_priority || 1" :disabled="!!busy" :aria-label="`${document.title}的检索优先级`" @change="changePriority(document, $event.target.value)"><option :value="1">标准</option><option :value="1.15">优先</option><option :value="1.25">重点</option></select></label><button class="button" :disabled="!!busy" @click="reprocess(document)">更新处理</button><button class="button" :disabled="!!busy" @click="setConsent(document, !document.cloud_allowed)">{{ document.cloud_allowed ? '改为仅本地' : '允许云端处理' }}</button><button class="button" :disabled="!!busy" @click="removeDocument(document)">删除</button></details></div></article></div>
      <details class="document-info"><summary>检索与模型卡维护</summary><p>新增文献在相关候选中获得短期优先。切片在后台完成，不需要逐条审核；仅本地资料不会用于云端回答。</p><button class="button" :disabled="!!busy || !missingVectors" @click="prepareIndex">准备检索索引</button><button v-if="developerMode" class="button" :disabled="!!busy" @click="refreshCards">同步模型卡</button></details>
    </template>
    <template v-else-if="detail">
      <p v-if="detail.status === 'archived'" class="notice">这是历史版本，已退出当前检索。<RouterLink :to="`/documents/${detail.superseded_by}`">查看当前版本 →</RouterLink></p>
      <div class="document-actions"><a :href="`/api/documents/${detail.id}/file?inline=true${currentSection?.page ? `#page=${currentSection.page}` : ''}`" target="_blank" rel="noopener noreferrer" class="button">打开原文件 ↗</a><a v-if="detail.source_url" :href="detail.source_url" target="_blank" rel="noopener noreferrer" class="button">原始来源 ↗</a><RouterLink v-if="detail.status === 'imported' && detail.cloud_allowed" :to="{ path: '/chat', query: { model: detail.model_id, document: detail.id, start: 'new' } }" class="button primary">围绕本文提问 →</RouterLink><button v-if="detail.status === 'imported' && detail.cloud_allowed" class="button" :disabled="processing" @click="startClassification(detail.id)">{{ processing ? '正在分析…' : report ? '重新分析' : '生成分析报告' }}</button></div>
      <div v-if="route.query.flow === 'ae' && detail.source_url === AE_PAPER_URL" class="paper-next"><div><strong>带着三个问题读本文</strong><p>训练时破坏了什么？重构目标是什么？与基础 AE 共享哪些结构？重点核对 2.2–2.4 节（原文件第 2–3 页）。</p></div><RouterLink :to="aeLocation('compare', detail)" class="button primary">比较普通 AE 与去噪 AE →</RouterLink></div>
      <p v-if="currentSection" class="citation-location">引用位置：{{ currentSection.locator }}{{ currentSection.page ? ` · 原文件第 ${currentSection.page} 页起` : '' }}。段落为解析编号，图表和公式请查看原文件。</p>
      <article v-if="!detail.cloud_allowed && detail.status === 'imported'" class="local-document-note"><h2>尚未调用大模型</h2><p>本文设置为仅本地收录，原文件和检索文字已保存。需要 AI 分析与问答时，请先允许云端处理；不会生成占位分类或要求逐片段确认。</p><button class="button" :disabled="!!busy" @click="setConsent(detail, true)">允许本文用于云端分析与问答</button></article>
      <IngestProgress v-if="ingestJob" :job="ingestJob" :events="ingestEvents" /><DocumentReport v-if="report" :report="report" :document="detail" @confirmed="reportConfirmed" />
      <p v-else-if="detail.cloud_allowed && !processing" class="notice">尚无阅读报告。可以直接围绕本文提问，也可以先生成报告，了解主题、方法和贡献。</p>
      <details class="document-info"><summary>文件与版本信息</summary><p>{{ detail.filename }} · {{ detail.page_count ? `${detail.page_count} 页 · ` : '' }}第 {{ detail.revision_number || 1 }} 版{{ detail.selected_pages?.length ? ` · 收录页码：${detail.selected_pages.join('、')}` : '' }}</p><RouterLink v-if="detail.revision_of" :to="`/documents/${detail.revision_of}`">查看上一版 →</RouterLink><p>前台提供报告及原文件定位；解析正文用于后台检索，不还原论文排版。</p><pre v-if="developerMode && currentSection" class="document-text">{{ currentSection.content }}</pre></details>
    </template>
  </section>
</template>

<style scoped>
.document-upload{border:1px solid var(--line);border-radius:10px;background:white;padding:16px 20px;margin:20px 0}.document-upload>summary{cursor:pointer;font-size:14px;font-weight:550}.document-upload>summary small{font-size:11px;font-weight:400;color:var(--muted);margin-left:12px}.document-upload .document-import{border:0;padding:18px 0 0;margin:0}.paper-next{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:22px 0;padding:20px 24px;border:1px solid #dee7f2;border-radius:10px;background:#f3f7fc}.paper-next strong{font-size:14px}.paper-next p{font-size:13px;line-height:1.9;color:var(--muted);margin:8px 0 0}.paper-next .button{flex:none}@media(max-width:760px){.paper-next{flex-direction:column;align-items:start}}
.document-info{margin:20px 0;color:var(--muted);font-size:12px;line-height:1.9}.document-info summary{cursor:pointer}.document-info .button{margin:8px 8px 0 0}.source-card>.document-summary{flex:1;min-width:0}.source-card>.document-controls{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;justify-content:flex-end;gap:8px;max-width:330px}.document-settings{font-size:12px;color:var(--muted)}.document-settings summary{cursor:pointer;padding:8px}.document-settings[open]{flex-basis:100%;padding:12px;border:1px solid var(--line);border-radius:8px}.document-settings label{display:flex;gap:10px;align-items:center;margin-bottom:10px}.document-settings select{font:inherit;padding:7px;max-width:120px;border:1px solid var(--line);border-radius:6px}.document-settings .button{margin:4px}.document-tag-preview{display:flex;flex-wrap:wrap;gap:6px}.document-tag-preview span{font-size:11px;color:var(--muted);padding:3px 7px;background:#f5f7fc;border-radius:4px}.document-hint{font-size:12px;line-height:1.8;color:var(--muted)}.local-document-note{margin:20px 0;padding:24px;border:1px solid var(--line);border-radius:12px;background:var(--paper)}.local-document-note h2{font-size:17px}.local-document-note p{font-size:14px;line-height:1.9;color:var(--muted)}.citation-location{font-size:13px;padding:15px 18px;border-left:3px solid var(--green);background:#f5f7fc;line-height:1.8}@media(max-width:760px){.source-card{flex-wrap:wrap;gap:12px}.source-card>.document-summary{flex-basis:calc(100% - 70px)}.source-card>.document-controls{max-width:none;flex-basis:100%;justify-content:flex-start}}
</style>
