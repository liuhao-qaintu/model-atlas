<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'
import RetrievalComparison from '../components/RetrievalComparison.vue'
import AnswerAssessment from '../components/AnswerAssessment.vue'
import { answerDimensions, filteredCases, judgmentFor, metricText, precisionStatus, relevanceOptions } from '../evaluationPresentation'
const cases = ref([]), runs = ref([]), run = ref(null), error = ref(''), busy = ref(''), mode = ref('rag')
const models = ref([]), sections = ref([]), adding = ref(false)
const selectionMode = ref('all'), selectedCases = ref([]), split = ref('development')
const drafts = ref({}), answerDrafts = ref({}), pools = ref({}), poolSearch = ref({})
const emptyForm = () => ({ question: '', model_id: '', reference_answer: '', relevant_section_ids: [], cloud_allowed: false, split: 'development' })
const form = ref(emptyForm())
const summary = computed(() => run.value?.summary || {})
const visibleCases = computed(() => filteredCases(cases.value, split.value))
const metrics = [
  { label: '实际返回精确率', key: 'returned_precision', help: '相关条数 ÷ 实际返回条数' },
  { label: '固定 Precision@5', key: 'precision_at_5', help: '相关条数 ÷ 5' },
  { label: '已标注集合 Recall@5', key: 'recall_at_5', help: '命中条数 ÷ 已标注相关条数' },
  { label: '证据标注覆盖率', key: 'judgment_coverage', help: '已判断条数 ÷ 实际返回条数' },
]
function modelName(id) { return models.value.find(m => m.id === id)?.name || '跨模型 / 自动识别' }
function evidenceTitle(e) { const name = modelName(e.model_id); return e.title.startsWith(`${name} ·`) ? e.title : `${name} · ${e.title}` }
function splitName(value) { return ({ development: '开发题', challenge: '保留挑战题' })[value] || value }
function excerptLabel(e) { return ({ full: '完整正文', python_units: '按完整函数节选', head: '正文开头节选' })[e.generation_selection?.strategy] || '旧记录保留正文' }
function answerStatus(row) {
  if (!row.case.answer_contract) return precisionStatus(row)
  const assessment = row.human_assessment || row.machine_assessment
  if (assessment) return `${assessment.origin === 'user' ? '用户复核' : assessment.origin === 'assistant' ? '助手辅助' : '机器建议'}：${({ correct:'符合标准', incorrect:'关键错误', partial:'部分符合', pending:'待判断' })[assessment.overall]}`
  return row.generated || row.answer_ready ? '待逐项评估' : '待获得完整回答'
}
function resetDrafts() { drafts.value = {}; answerDrafts.value = {}; pools.value = {}; poolSearch.value = {} }
async function load() { [cases.value, runs.value, models.value] = await Promise.all([api('/evaluation/cases'), api('/evaluation/runs'), api('/models')]) }
async function execute() {
  busy.value = 'run'; error.value = ''
  try {
    run.value = await api('/evaluation/runs', { mode: mode.value, split: split.value, ...(selectionMode.value === 'custom' ? { case_ids: selectedCases.value } : {}) })
    resetDrafts(); await load()
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function open(id) { error.value = ''; try { run.value = await api(`/evaluation/runs/${id}`); resetDrafts() } catch (e) { error.value = e.message } }
async function generate(row) {
  busy.value = row.case.id; error.value = ''
  try { run.value = await api(`/evaluation/runs/${run.value.id}/answers/${row.case.id}`, {}); delete answerDrafts.value[row.case.id]; await load() }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
function answerDraft(row) { return answerDrafts.value[row.case.id] ||= { dimensions: { ...(row.answer_rubric || {}) }, note: row.grade_note || '' } }
async function grade(row, value) {
  busy.value = row.case.id; error.value = ''
  try { run.value = await api(`/evaluation/runs/${run.value.id}/grades/${row.case.id}`, { grade: value, ...answerDraft(row) }, { method: 'PATCH' }); await load() }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
function draft(row, id) { return drafts.value[`${row.case.id}/${id}`] ||= { ...judgmentFor(row, id) } }
async function judge(row, id) {
  busy.value = `${row.case.id}/${id}`; error.value = ''
  try {
    const value = draft(row, id)
    run.value = await api(`/evaluation/runs/${run.value.id}/judgments/${row.case.id}`, { section_id: id, label: value.label, note: value.note, origin: 'user' }, { method: 'PATCH' })
    delete drafts.value[`${row.case.id}/${id}`]; await load()
  } catch (e) { error.value = e.message } finally { busy.value = '' }
}
async function loadPool(row) {
  busy.value = row.case.id; error.value = ''
  try { pools.value[row.case.id] = await api(`/evaluation/runs/${run.value.id}/pool/${row.case.id}`) }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
function poolEntries(row) {
  const term = (poolSearch.value[row.case.id] || '').toLowerCase()
  return (pools.value[row.case.id] || []).filter(e => !term || `${e.title} ${e.content}`.toLowerCase().includes(term))
}
async function chooseModel() {
  form.value.relevant_section_ids = []; sections.value = []
  if (form.value.model_id) sections.value = (await api(`/models/${form.value.model_id}`)).sections.filter(s => !['figures', 'sources', 'followups'].includes(s.slug))
}
async function add() {
  busy.value = 'add'; error.value = ''
  try { await api('/evaluation/cases', form.value); await load(); adding.value = false; form.value = emptyForm() }
  catch (e) { error.value = e.message } finally { busy.value = '' }
}
function exportRun() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(run.value, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a'); a.href = url; a.download = `${run.value.id}.json`; a.click(); URL.revokeObjectURL(url)
}
onMounted(async () => { try { await load(); if (runs.value[0]) await open(runs.value[0].id) } catch (e) { error.value = e.message } })
</script>

<template>
  <section>
    <div class="page-top"><span class="eyebrow">检索证据与回答质量</span><button class="button" :disabled="!!busy" @click="adding = !adding">新增评测题</button></div>
    <h1>质量评测</h1><p class="page-intro">检查回答错在哪里、遗漏了什么条件，以及本轮资料是否足够。检索指标和回答判断分别记录。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <details><summary>比较检索方案</summary><RetrievalComparison :cases="cases" /></details>
    <form v-if="adding" class="surface-form" @submit.prevent="add">
      <h2>新增标注问题</h2><label>问题<textarea v-model="form.question" required maxlength="2000" /></label>
      <label>模型<select v-model="form.model_id" required @change="chooseModel"><option value="">选择模型</option><option v-for="m in models" :key="m.id" :value="m.id">{{ m.name }}</option></select></label>
      <label>参考答案<textarea v-model="form.reference_answer" required maxlength="5000" rows="3" /></label>
      <label>题集用途<select v-model="form.split"><option value="development">开发题：日常调试</option><option value="challenge">保留题：阶段检查</option></select></label>
      <fieldset><legend>预标注相关片段（后续可补标原文）</legend><label v-for="s in sections" :key="s.id" class="checkbox-label"><input type="checkbox" v-model="form.relevant_section_ids" :value="s.id" />{{ s.title }}</label></fieldset>
      <label class="checkbox-label"><input type="checkbox" v-model="form.cloud_allowed" />允许已配置的模型 API 处理本题与对应资料</label>
      <button class="button primary" :disabled="!!busy || !form.relevant_section_ids.length">保存题目</button>
    </form>
    <div class="evaluation-controls">
      <div><strong>{{ visibleCases.length }} 道{{ split === 'challenge' ? '保留' : split === 'all' ? '可选' : '开发' }}问题</strong><small>共 {{ cases.length }} 题 · 规则规划，复用实际检索流程</small></div>
      <label>题集用途<select v-model="split" :disabled="!!busy" @change="selectedCases = []"><option value="development">开发题 · 日常调试</option><option value="challenge">保留挑战题 · 阶段检查</option><option value="all">全部题目 · 混合结果</option></select></label>
      <label>评测范围<select v-model="selectionMode" :disabled="!!busy"><option value="all">本题集全部问题</option><option value="custom">自选题目</option></select></label>
      <label>检索方式<select v-model="mode" :disabled="!!busy"><option value="rag">三路召回 + 重排（配置 API）</option><option value="local">本地关键词 + 实体章节</option></select></label>
      <button class="button primary" :disabled="!!busy || !visibleCases.length || (selectionMode === 'custom' && !selectedCases.length)" @click="execute">{{ busy === 'run' ? '正在检索评测…' : '运行检索评测' }}</button>
    </div>
    <p class="muted">自建题库由助手编写并预标注，参考答案待独立复核。保留题与开发题分开运行；同一助手出题并评分仍可能存在偏差。</p>
    <fieldset v-if="selectionMode === 'custom'" class="surface-form evaluation-case-picker"><legend>选择问题（已选 {{ selectedCases.length }} 题）</legend><label v-for="item in visibleCases" :key="item.id" class="checkbox-label"><input type="checkbox" v-model="selectedCases" :value="item.id" :disabled="!!busy" />{{ item.question }}</label></fieldset>
    <p v-if="run" class="muted">当前记录 {{ run.rows.length }} 题 · {{ run.mode === 'rag' ? '三路检索' : '本地检索' }} · {{ summary.dataset_splits?.map(splitName).join(' / ') }}；上方控件用于下一次评测。</p>
    <p v-if="run?.qrels_sha256" class="muted">标注版本 {{ run.qrels_sha256.slice(0, 12) }} · 判断会按题目和资料版本复用，历史记录保持各自快照。</p>
    <p v-if="run?.purpose === 'regression'" class="notice">本记录用于已知问题回归。题目已用于本轮修复，不属于独立盲测。</p>
    <h2>回答诊断</h2>
    <div class="evaluation-stats"><article><small>有逐项标准</small><strong>{{ summary.answer_assessment?.contract_cases || 0 }}</strong><p>事实、条件与必答点</p></article><article><small>辅助审阅</small><strong>{{ summary.answer_assessment?.machine_reviewed || 0 }}</strong><p>其中提示问题 {{ summary.answer_assessment?.machine_flagged || 0 }} 题，待复核</p></article><article><small>人工逐项复核</small><strong>{{ summary.answer_assessment?.human_reviewed || 0 }}</strong><p>机器建议不计为人工复核</p></article><article><small>人工评分中完全正确</small><strong>{{ metricText(summary.user_answer_accuracy, '待复核') }}</strong><p>计入 {{ summary.user_graded_answers || 0 }} 题；不代表独立盲测</p></article></div>
    <details><summary>检索指标与标注情况</summary>
    <div class="evaluation-stats"><article v-for="metric in metrics" :key="metric.key"><small>{{ metric.label }}</small><strong>{{ metricText(summary[metric.key], metric.key === 'judgment_coverage' ? '无结果' : '待标注') }}</strong><p>{{ metric.help }}</p><small v-if="summary.metric_counts && metric.key !== 'judgment_coverage'">计入 {{ summary.metric_counts[metric.key] || 0 }} / {{ summary.evaluated || 0 }} 题</small></article></div>
    <div class="evaluation-note">
      <p>已判断 {{ summary.judged_sections || 0 }} / {{ summary.retrieved_sections || 0 }} 个实际返回片段，待判断 {{ summary.unjudged_sections || 0 }} 个。精确率仅计入返回证据全部已判断的题；空结果的实际返回精确率不适用。</p>
      <p>已标注集合 Hit@5：{{ metricText(summary.hit_at_5, '不适用') }} · MRR：{{ metricText(summary.reciprocal_rank, '不适用') }}。召回率、命中率及 MRR 仅针对已标注相关集合，不代表全库相关证据全集。</p>
      <p>用户复核完全部返回证据：{{ summary.user_reviewed_rows || 0 }} 题；其实际精确率 {{ metricText(summary.user_reviewed_returned_precision, '待复核') }}。</p>
      <p>已评分回答中完全正确占比：<b>{{ metricText(summary.answer_accuracy, '未评分') }}</b> · 共 {{ summary.graded_answers || 0 }} 题，部分正确 {{ summary.partly_correct || 0 }} 题。助手辅助评分 {{ summary.assistant_graded_answers || 0 }} 题，用户评分 {{ summary.user_graded_answers || 0 }} 题。</p>
      <small>直接相关与辅助相关均计为相关；置信度尚未校准，不报告概率。题目、证据、标注来源、判断依据与修改记录可随结果导出。{{ run ? `成功检索 ${summary.evaluated} 题，异常 ${summary.failed} 题。` : '' }}</small>
    </div>
    <details v-if="run?.legacy_summary" class="evaluation-row"><summary>历史计算口径（保留原分数）</summary><div class="evaluation-row-body"><p>旧 Precision@5：{{ metricText(run.legacy_summary.precision_at_5, '未评测') }}。旧算法将未标注片段按未命中处理并固定除以 5。上方展示新口径，原始分数保存在导出记录中。</p></div></details>
    <div v-if="summary.graded_answers" class="evaluation-stats"><article v-for="dimension in answerDimensions" :key="dimension.key"><small>{{ dimension.label }}</small><strong>{{ metricText(summary.answer_dimensions?.[dimension.key]?.pass_rate, '未评分') }}</strong><p>{{ dimension.help }}</p><small>单独评分 {{ summary.answer_dimensions?.[dimension.key]?.count || 0 }} 题</small></article></div>
    </details>
    <div class="toolbar"><label v-if="runs.length">历史记录<select :value="run?.id" :disabled="!!busy" @change="open($event.target.value)"><option v-for="item in runs" :key="item.id" :value="item.id">{{ new Date(item.created_at).toLocaleString() }} · {{ item.mode === 'rag' ? '三路检索' : '本地检索' }}</option></select></label><button v-if="run" class="button" @click="exportRun">导出评测结果</button></div>
    <div class="evaluation-rows">
      <details v-for="row in (run?.rows || visibleCases.map(item => ({ case:item })))" :key="row.case.id" class="evaluation-row">
        <summary><span>{{ modelName(row.case.model_id) }}</span><strong>{{ row.case.question }}</strong><small>{{ answerStatus(row) }}</small></summary>
        <div class="evaluation-row-body">
          <p class="muted">{{ row.case.split === 'challenge' ? '保留挑战题' : '开发题' }} · {{ row.case.annotation_origin === 'user' ? '用户编写' : '助手编写，参考答案待复核' }}</p>
          <template v-if="row.case.prior_turns?.length"><h3>前序问答</h3><p v-for="(turn, i) in row.case.prior_turns" :key="i">{{ i + 1 }}. {{ turn.question }}</p><details v-for="(turn, i) in row.answer_prelude || []" :key="i"><summary>查看第 {{ i + 1 }} 轮实际回答</summary><MarkdownContent :content="turn.answer" /></details></template>
          <h3>参考答案</h3><p>{{ row.case.reference_answer }}</p>
          <details v-if="row.plan"><summary>查询规划与检索步骤</summary><p>检索用问题：{{ row.plan.normalized_query }}</p><p v-for="step in row.retrieval_steps || []" :key="step.id">{{ step.query }} · {{ step.kind }} · 保留依据 {{ step.references.length }} 条</p></details>
          <template v-if="row.metrics"><h3>本题指标</h3><p>实际返回 {{ row.metrics.retrieved }} 条，待判断 {{ row.metrics.unjudged }} 条。实际精确率 {{ metricText(row.metrics.returned_precision) }}；固定 Precision@5 {{ metricText(row.metrics.precision_at_5) }}。</p><p v-if="row.metrics.unjudged" class="muted">按现有标注，实际精确率范围 {{ metricText(row.metrics.returned_precision_lower_bound) }}–{{ metricText(row.metrics.returned_precision_upper_bound) }}；固定 Precision@5 范围 {{ metricText(row.metrics.precision_lower_bound) }}–{{ metricText(row.metrics.precision_upper_bound) }}。范围表示待判断的不确定性。</p></template>
          <h3>实际检索证据 · 逐条判断</h3><p v-if="!row.evidence?.length" class="muted">{{ row.evidence ? '本题未返回证据；澄清题可以只检查回答行为。' : '运行评测后查看证据。' }}</p>
          <article v-for="e in row.evidence || []" :key="e.section_id" class="evidence-judgment">
            <RouterLink :to="e.reader_url">{{ evidenceTitle(e) }} →</RouterLink>
            <small>{{ judgmentFor(row, e.section_id).origin === 'user' ? '用户判断' : judgmentFor(row, e.section_id).origin === 'assistant' ? '助手预标注，待复核' : '尚未判断' }}</small>
            <details><summary>查看本轮证据正文</summary><MarkdownContent v-if="e.content" :content="e.content" /><p v-else class="muted">旧记录未保存正文，可打开原文核对；资料版本变化后需重新评测。</p></details>
            <div class="judgment-edit"><label>相关性<select v-model="draft(row, e.section_id).label" :disabled="!!busy"><option v-for="option in relevanceOptions" :key="option.value" :value="option.value">{{ option.label }}</option></select></label><label>判断依据<input v-model="draft(row, e.section_id).note" maxlength="2000" placeholder="说明它支持哪个结论，或为何无关" :disabled="!!busy" /></label><button class="button" :disabled="!!busy || !draft(row, e.section_id).note.trim()" @click="judge(row, e.section_id)">保存判断</button></div>
          </article>
          <details v-if="run" class="evaluation-pool"><summary>补标其他原文 · 扩大相关证据池</summary><p class="muted">预标注不构成全集。可检查本次涉及模型的其他章节，包括未返回的论文与源码。补标相关章节会更新已标注集合的召回率。</p><button class="button" :disabled="!!busy" @click="loadPool(row)">载入候选原文</button><label v-if="pools[row.case.id]">搜索原文<input v-model="poolSearch[row.case.id]" placeholder="标题或正文关键词" /></label><div class="evaluation-pool-list"><details v-for="e in poolEntries(row)" :key="e.section_id"><summary>{{ evidenceTitle(e) }}</summary><RouterLink :to="e.reader_url">打开原文 →</RouterLink><MarkdownContent :content="e.content" /><div class="judgment-edit"><label>相关性<select v-model="draft(row, e.section_id).label" :disabled="!!busy"><option v-for="option in relevanceOptions" :key="option.value" :value="option.value">{{ option.label }}</option></select></label><label>判断依据<input v-model="draft(row, e.section_id).note" maxlength="2000" :disabled="!!busy" /></label><button class="button" :disabled="!!busy || !draft(row, e.section_id).note.trim()" @click="judge(row, e.section_id)">保存判断</button></div></details></div></details>
          <p v-if="row.error" class="notice error">{{ row.error }}</p>
          <template v-if="run">
            <div class="page-top"><h3>系统实际回答</h3><button v-if="!(row.generated || row.answer_ready)" class="button" :disabled="!!busy || !row.case.cloud_allowed" @click="generate(row)">{{ busy === row.case.id ? '执行真实问答中…' : row.answer ? '重新生成本题回答' : '生成本题回答' }}</button></div>
            <p v-if="row.answer && !(row.generated || row.answer_ready)" class="notice">本次未完成生成，以下为检索摘录；可以重试。</p>
            <p v-if="row.answer_plan?.source_scope" class="muted">本轮回答的来源范围：{{ row.answer_plan.source_scope.label }}。</p><MarkdownContent v-if="row.answer" :content="row.answer" /><p v-else class="muted">按需生成，再对照实际引用和参考答案核查。</p>
            <details v-if="row.answer_evidence_details?.length"><summary>回答实际使用的证据 · 可能与规则检索不同</summary><p class="muted">上方检索指标采用规则规划；生成回答复用真实聊天规划。引用支持需核对下面本次生成实际获得的正文。</p><details v-for="(e, i) in row.answer_evidence_details" :key="e.section_id"><summary>[{{ i + 1 }}] {{ e.title }}</summary><RouterLink :to="e.reader_url">打开原文 →</RouterLink><p class="muted">{{ e.source_type || '旧记录未标注类型' }} · {{ excerptLabel(e) }}</p><MarkdownContent :content="e.generation_excerpt || e.content" /><details v-if="e.generation_excerpt && e.generation_excerpt !== e.content"><summary>查看完整原文（不代表本轮全部发送给模型）</summary><MarkdownContent :content="e.content" /></details></details></details>
            <AnswerAssessment v-if="row.case.answer_contract" :row="row" :run-id="run.id" :disabled="!!busy" @update="run = $event" @busy="busy = $event ? row.case.id : ''" />
            <fieldset v-if="!row.case.answer_contract && (row.generated || row.answer_ready)" class="answer-rubric"><legend>回答质量 · 各维度独立判断</legend><label v-for="dimension in answerDimensions" :key="dimension.key">{{ dimension.label }}<small>{{ dimension.help }}</small><select v-model="answerDraft(row).dimensions[dimension.key]" :disabled="!!busy"><option :value="undefined">未判断</option><option value="correct">符合</option><option value="partial">部分符合</option><option value="incorrect">不符合</option><option value="not_applicable">不适用</option></select></label><label class="rubric-note">评分依据<textarea v-model="answerDraft(row).note" maxlength="2000" rows="2" /></label></fieldset>
            <div v-if="!row.case.answer_contract && (row.generated || row.answer_ready)" class="evaluation-grade"><span>{{ row.grade_origin === 'assistant' ? '助手辅助评分（待复核）：' : '用户综合评分：' }}</span><button v-for="item in [{value:'correct',label:'正确'}, {value:'partial',label:'部分正确'}, {value:'incorrect',label:'错误'}]" :key="item.value" :class="['button', {primary: row.grade === item.value}]" :disabled="!!busy" @click="grade(row, item.value)">{{ item.label }}并保存</button></div>
            <p v-if="row.grade_note" class="muted">已保存的核查说明：{{ row.grade_note }}</p>
          </template>
        </div>
      </details>
    </div>
  </section>
</template>
