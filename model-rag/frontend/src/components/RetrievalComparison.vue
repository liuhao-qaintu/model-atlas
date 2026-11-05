<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'
import { metricText } from '../evaluationPresentation'
const props = defineProps({ cases: { type: Array, default: () => [] } })
const selected = ref([]), record = ref(null), history = ref([]), busy = ref(false), error = ref(''), drafts = ref({})
const variants = ['dense', 'hybrid', 'hybrid_graph']
const labels = { dense: '纯向量', hybrid: '混合检索', hybrid_graph: '混合＋图谱扩展' }
const recommendedIds = ['q01', 'q04', 'q12', 'q14', ...Array.from({ length: 8 }, (_, i) => `challenge-${String(i + 1).padStart(2, '0')}`)]
const recommended = computed(() => recommendedIds.map(id => props.cases.find(c => c.id === id)).filter(Boolean))
function selectRecommended() { selected.value = recommended.value.map(c => c.id) }
async function loadHistory() { history.value = await api('/evaluation/comparisons') }
async function open(id) {
  error.value = ''
  try { record.value = await api(`/evaluation/comparisons/${id}`); drafts.value = {} }
  catch (e) { error.value = e.message }
}
async function run() {
  busy.value = true; error.value = ''
  try { record.value = await api('/evaluation/comparisons', { case_ids: selected.value }); drafts.value = {}; await loadHistory() }
  catch (e) { error.value = e.message } finally { busy.value = false }
}
function evidenceUnion(row) {
  return [...new Map(variants.flatMap(name => row.results[name].evidence || []).map(e => [e.section_id, e])).values()]
}
function draft(row, evidence) {
  return drafts.value[`${row.case.id}/${evidence.section_id}`] ||= { label: row.judgments[evidence.section_id]?.label || 'unjudged', note: row.judgments[evidence.section_id]?.note || '' }
}
async function judge(row, evidence) {
  busy.value = true; error.value = ''
  try {
    record.value = await api(`/evaluation/comparisons/${record.value.id}/judgments/${row.case.id}`, { section_id: evidence.section_id, ...draft(row, evidence), origin: 'user' }, { method: 'PATCH' })
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(record.value, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a'); a.href = url; a.download = `${record.value.id}.json`; a.click(); URL.revokeObjectURL(url)
}
onMounted(async () => { try { await loadHistory(); if (history.value[0]) await open(history.value[0].id) } catch (e) { error.value = e.message } })
</script>

<template>
  <details class="surface-form retrieval-comparison">
    <summary><strong>检索方案对照</strong> · 固定资料和问题，检查图谱扩展带来的变化</summary>
    <p>三种方案共享规则规划、来源限制、查询向量和相关性标注。为分离召回效果，均关闭重排、实体章节召回、来源加权与定义补全；这不是线上完整配置，也不评判生成答案。</p>
    <p class="muted">固定规划可能使用已有模型关系；这里比较的是后续检索扩展，不是完全移除图知识的端到端消融。查询向量会复用，运行耗时不用于性能排名。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <div class="evaluation-controls">
      <button class="button" :disabled="busy" @click="selectRecommended">选择重点模型问题</button>
      <button class="button primary" :disabled="busy || !selected.length || selected.length > 12" @click="run">{{ busy ? '正在处理…' : `运行三组检索（${selected.length} 题）` }}</button>
      <button v-if="record" class="button" :disabled="busy" @click="download">导出对照记录</button>
    </div>
    <details><summary>选择问题 · 每轮最多 12 题</summary><label v-for="c in cases" :key="c.id" class="checkbox-label"><input v-model="selected" type="checkbox" :value="c.id" :disabled="busy" />{{ c.question }}</label></details>
    <div v-if="history.length" class="evaluation-controls"><label>历史对照<select :value="record?.id || ''" :disabled="busy" @change="open($event.target.value)"><option value="" disabled>选择记录</option><option v-for="item in history" :key="item.id" :value="item.id">{{ new Date(item.created_at).toLocaleString() }} · {{ item.case_count }} 题 · {{ item.complete ? '执行完成' : '含失败项' }}</option></select></label></div>
    <template v-if="record">
      <p class="muted">资料版本 {{ record.corpus_sha256.slice(0, 12) }} · 标注版本 {{ record.qrels_sha256.slice(0, 12) }}。助手标注仅供辅助，待独立复核；召回率只针对已标注相关集合。</p>
      <p v-if="!record.complete" class="notice error">存在失败项，本记录不支持完整的方案优劣结论；失败不会当作零命中计分。</p>
      <div class="comparison-table-scroll"><table class="comparison-table">
        <thead><tr><th>方案</th><th>实际精确率</th><th>已标注 Recall@5</th><th>MRR</th><th>标注覆盖</th><th>执行成功 / 题数</th></tr></thead>
        <tbody><tr v-for="name in variants" :key="name">
          <td>{{ labels[name] }}</td>
          <td>{{ metricText(record.summaries[name].returned_precision) }}<small> · {{ record.summaries[name].metric_counts.returned_precision }} 题</small></td>
          <td>{{ metricText(record.summaries[name].recall_at_5) }}<small> · {{ record.summaries[name].metric_counts.recall_at_5 }} 题</small></td>
          <td>{{ record.summaries[name].reciprocal_rank == null ? '待标注' : record.summaries[name].reciprocal_rank.toFixed(3) }}<small> · {{ record.summaries[name].metric_counts.reciprocal_rank }} 题</small></td>
          <td>{{ metricText(record.summaries[name].judgment_coverage) }}</td><td>{{ record.summaries[name].evaluated }} / {{ record.rows.length }}</td>
        </tr></tbody>
      </table></div>
      <p class="muted">各指标按适用题目单独计数；执行成功不代表回答正确。本批已有题用于开发复查，尚未经过独立人工复核。</p>
      <details v-for="row in record.rows" :key="row.case.id" class="comparison-question">
        <summary>{{ row.case.question }}</summary>
        <p>参考答案（待复核）：{{ row.case.reference_answer }}</p>
        <div class="comparison-result-grid"><article v-for="name in variants" :key="name"><h3>{{ labels[name] }}</h3><p v-if="row.results[name].error" class="notice error">{{ row.results[name].error }}</p><template v-else><p class="muted">{{ row.results[name].steps?.length || 0 }} 步 · {{ row.results[name].steps?.filter(s => s.kind === 'graph_hop').length || 0 }} 次图扩展 · {{ row.results[name].stop_reason }}</p><ol><li v-for="e in row.results[name].evidence" :key="e.section_id"><RouterLink :to="e.reader_url">{{ e.title }}</RouterLink></li></ol></template></article></div>
        <details><summary>共同证据池 · 一次判断同步用于三组结果</summary>
          <article v-for="e in evidenceUnion(row)" :key="e.section_id" class="evidence-judgment">
            <RouterLink :to="e.reader_url">{{ e.title }}</RouterLink><small> · {{ row.judgments[e.section_id]?.origin === 'user' ? '用户判断' : row.judgments[e.section_id] ? '助手辅助标注' : '待标注' }}</small>
            <details><summary>查看完整检索正文</summary><pre class="comparison-evidence">{{ e.content }}</pre></details>
            <label>相关性<select v-model="draft(row, e).label" :disabled="busy"><option value="direct">直接回答</option><option value="supporting">补充依据</option><option value="irrelevant">无关</option><option value="unjudged">待判断</option></select></label>
            <label>判断依据<textarea v-model="draft(row, e).note" rows="2" maxlength="2000" :disabled="busy" /></label><button class="button" :disabled="busy || !draft(row, e).note.trim()" @click="judge(row, e)">保存判断并同步三组</button>
          </article>
        </details>
      </details>
    </template>
  </details>
</template>

<style scoped>
.retrieval-comparison { margin: 1.5rem 0; }
.retrieval-comparison > summary { cursor: pointer; line-height: 1.8; }
.comparison-table-scroll { overflow-x: auto; }
.comparison-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.comparison-table th, .comparison-table td { text-align: left; padding: .85rem .65rem; border-bottom: 1px solid var(--border, #deded8); white-space: nowrap; }
.comparison-question { padding: 1rem 0; border-bottom: 1px solid var(--border, #deded8); }
.comparison-question > summary { cursor: pointer; line-height: 1.7; }
.comparison-result-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.comparison-result-grid h3 { font-size: .95rem; }
.comparison-result-grid ol { padding-left: 1.3rem; }
.comparison-result-grid li { margin-bottom: .7rem; overflow-wrap: anywhere; }
.comparison-evidence { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 28rem; overflow: auto; font: inherit; }
@media (max-width: 850px) { .comparison-result-grid { grid-template-columns: 1fr; } }
</style>
