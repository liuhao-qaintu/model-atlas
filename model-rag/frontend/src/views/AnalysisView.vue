<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'

const route = useRoute()
const documents = ref([]), codeOptions = ref([]), documentId = ref(''), left = ref(''), right = ref('')
const analysis = ref(null), comparison = ref(null), busy = ref(''), error = ref('')
const selectedDocument = computed(() => documents.value.find(d => d.id === documentId.value))
const counts = computed(() => ({ candidate: 0, accepted: 0, rejected: 0, ...(analysis.value?.candidates || []).reduce((result, c) => ({ ...result, [c.status]: (result[c.status] || 0) + 1 }), {}) }))
function applyDocument() {
  if (documents.value.some(d => d.id === route.query.document)) documentId.value = route.query.document
}
async function extract() {
  busy.value = 'extract'; error.value = ''; analysis.value = null
  try { analysis.value = await api(`/analysis/documents/${documentId.value}/relations`, {}) }
  catch (e) { error.value = e.message }
  finally { busy.value = '' }
}
async function review(candidate, decision) {
  busy.value = candidate.id; error.value = ''
  try { analysis.value = await api(`/analysis/relations/${analysis.value.id}/${candidate.id}/review`, { decision }) }
  catch (e) { error.value = e.message }
  finally { busy.value = '' }
}
async function compare() {
  busy.value = 'compare'; error.value = ''; comparison.value = null
  try { comparison.value = await api('/analysis/code/compare', { left: left.value, right: right.value }) }
  catch (e) { error.value = e.message }
  finally { busy.value = '' }
}
const codeText = code => code.blocks.map(b => '```python\n' + b.code + '\n```').join('\n\n')
onMounted(async () => {
  try {
    ;[documents.value, codeOptions.value] = await Promise.all([api('/documents'), api('/analysis/code')])
    documentId.value = documents.value[0]?.id || ''; applyDocument()
    left.value = codeOptions.value.find(c => c.section_id === 'cnn-1d:code')?.section_id || codeOptions.value[0]?.section_id || ''
    right.value = codeOptions.value.find(c => c.section_id === 'cnn-2d:code')?.section_id || codeOptions.value.find(c => c.section_id !== left.value)?.section_id || ''
  } catch (e) { error.value = e.message }
})
watch(() => route.query.document, applyDocument)
watch(documentId, () => { analysis.value = null })
</script>

<template>
  <section class="analysis-page">
    <div class="page-top"><span class="eyebrow">DOCUMENT ANALYSIS</span><RouterLink to="/graph" class="button">查看关系图谱 →</RouterLink></div>
    <h1>资料分析</h1><p class="page-intro">从原文发现模型联系，比较代码中的共同算子与结构差异。</p>
    <p v-if="error" class="notice error">{{ error }}</p>
    <section class="analysis-section">
      <h2>提取模型关系</h2><p>先查看候选的逐字原文依据，确认后进入关系图谱和检索；拒绝的候选保留记录。</p>
      <div class="analysis-controls"><label>选择已入库资料<select v-model="documentId" :disabled="!!busy" aria-label="关系提取资料"><option v-for="document in documents" :key="document.id" :value="document.id">{{ document.title }}</option></select></label><button class="button primary" :disabled="!!busy || !documentId" @click="extract">{{ busy === 'extract' ? '正在提取…' : '提取候选关系' }}</button></div>
      <small v-if="selectedDocument">{{ selectedDocument.cloud_allowed ? '该资料已允许云端处理：使用已配置模型提取，失败时采用本地规则。' : '该资料仅限本地处理：只采用本地规则，不发送云端。' }}</small>
      <template v-if="analysis">
        <div class="analysis-summary"><span>{{ analysis.method === 'llm' ? '模型提取' : '本地规则' }}{{ analysis.cached ? ' · 已保存结果' : '' }}</span><span>抽样 {{ analysis.processed_sections }} / {{ analysis.total_sections }} 个片段</span><span>{{ counts.candidate }} 待确认 · {{ counts.accepted }} 已确认 · {{ counts.rejected }} 已拒绝</span></div>
        <p v-for="warning in analysis.warnings" :key="warning" class="notice compact">{{ warning }}</p>
        <p v-if="!analysis.candidates.length" class="notice">本批正文没有产生可定位的候选关系。可尝试方法章节更明确的资料。</p>
        <article v-for="candidate in analysis.candidates" :key="candidate.id" class="relation-candidate">
          <div class="page-top"><h3>{{ analysis.owner_name }} <span>— {{ candidate.relation }} →</span> {{ candidate.target_name }}</h3><span class="soft-label">{{ { candidate: '待确认', accepted: '已确认', rejected: '已拒绝' }[candidate.status] }}</span></div>
          <blockquote>{{ candidate.quote }}</blockquote><small>{{ candidate.locator }}</small>
          <div class="candidate-actions"><RouterLink :to="candidate.reader_url">阅读这段原文 →</RouterLink><RouterLink v-if="candidate.status === 'accepted'" :to="{ path:'/graph', query:{ node:analysis.owner_model_id, edge:'auto-' + candidate.id } }">查看已加入关系 →</RouterLink><button class="button primary" :disabled="!!busy || candidate.status === 'accepted'" @click="review(candidate, 'accepted')">确认关系</button><button class="button" :disabled="!!busy || candidate.status === 'rejected'" @click="review(candidate, 'rejected')">拒绝关系</button></div>
        </article>
      </template>
    </section>
    <section class="analysis-section">
      <h2>比较代码结构</h2><p>分析Python语法树中的算子调用和节点形状；直接展示配置与原代码，便于核对相似之处。</p>
      <div class="code-controls"><label>代码片段 A<select v-model="left" :disabled="!!busy" aria-label="代码片段 A"><option v-for="option in codeOptions" :key="option.section_id" :value="option.section_id">{{ option.model_name }} · {{ option.title }}</option></select></label><label>代码片段 B<select v-model="right" :disabled="!!busy" aria-label="代码片段 B"><option v-for="option in codeOptions" :key="option.section_id" :value="option.section_id">{{ option.model_name }} · {{ option.title }}</option></select></label><button class="button primary" :disabled="!!busy || !left || !right || left === right" @click="compare">{{ busy === 'compare' ? '分析中…' : '比较代码' }}</button></div>
      <small>仅列出含可解析Python算子的片段；结构伪代码不参与比较，分析不执行资料代码。</small>
      <template v-if="comparison">
        <div class="analysis-summary"><span>算子调用重合度 {{ (comparison.operator_overlap * 100).toFixed(1) }}%</span><span>AST节点形状重合度 {{ (comparison.ast_shape_overlap * 100).toFixed(1) }}%</span></div><p class="notice compact">{{ comparison.scope }}</p>
        <div class="operator-summary"><div><h3>共同算子</h3><span v-for="(count, name) in comparison.shared_operators" :key="name" class="soft-label">{{ name }} × {{ count }}</span><p v-if="!Object.keys(comparison.shared_operators).length">没有共同的已识别算子。</p></div><div><h3>A 多出的调用</h3><span v-for="(count, name) in comparison.left_only" :key="name" class="soft-label">{{ name }} × {{ count }}</span></div><div><h3>B 多出的调用</h3><span v-for="(count, name) in comparison.right_only" :key="name" class="soft-label">{{ name }} × {{ count }}</span></div></div>
        <div class="code-comparison"><article v-for="(code, index) in [comparison.left, comparison.right]" :key="code.section_id"><h3>{{ index ? 'B' : 'A' }} · {{ code.title }}</h3><RouterLink :to="code.reader_url">阅读代码来源 →</RouterLink><details open><summary>静态调用与配置</summary><ul><li v-for="(call, n) in code.operators" :key="n"><code>{{ call.call }}</code><small>代码块{{ call.block }} · 第{{ call.line }}行</small></li></ul></details><details><summary>完整代码片段</summary><MarkdownContent :content="codeText(code)" /></details></article></div>
      </template>
    </section>
  </section>
</template>

<style scoped>
.analysis-section { margin:30px 0; padding:26px; background:white; border:1px solid var(--line); border-radius:14px; }
.analysis-section > p { font-size:14px; color:#475569; line-height:1.9; margin:10px 0 18px; }
.analysis-controls,.code-controls { display:flex; align-items:end; gap:14px; margin:20px 0 12px; }
label { flex:1; display:grid; gap:8px; min-width:0; font-size:13px; }
select { width:100%; min-width:0; padding:10px; border:1px solid var(--line); border-radius:8px; background:white; }
.analysis-summary { display:flex; gap:14px; flex-wrap:wrap; margin:24px 0 16px; font-size:12px; color:#334155; }
.relation-candidate { border-top:1px solid var(--line); padding:24px 0; }
.relation-candidate h3 { font-size:17px; }.relation-candidate h3 span { font-size:12px; color:#64748b; }
blockquote { margin:14px 0; padding:16px; white-space:pre-wrap; border-left:3px solid #93c5fd; background:#f8fafc; font-size:13px; line-height:1.8; }
.candidate-actions { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top:16px; font-size:12px; }
small { color:#64748b; font-size:12px; }
.operator-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0; }.operator-summary h3 { font-size:14px; margin-bottom:10px; }.operator-summary span { display:inline-block; margin:4px; }
.code-comparison { display:grid; grid-template-columns:1fr 1fr; gap:20px; }.code-comparison article { padding:18px; border:1px solid var(--line); border-radius:10px; min-width:0; }.code-comparison h3 { font-size:15px; }.code-comparison a { font-size:12px; }.code-comparison details { margin-top:18px; font-size:13px; }.code-comparison summary { cursor:pointer; }.code-comparison li { padding:10px 0; overflow-wrap:anywhere; }.code-comparison li small { display:block; margin-top:6px; }
@media(max-width:800px) { .analysis-controls,.code-controls { display:grid; }.code-comparison,.operator-summary { grid-template-columns:1fr; }.analysis-section { padding:18px; } }
</style>
