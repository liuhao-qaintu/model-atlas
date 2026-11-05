<script setup>
import { computed, ref, watch } from 'vue'
import { api } from '../api'

const props = defineProps({ row: { type: Object, required: true }, runId: { type: String, required: true }, disabled: Boolean })
const emit = defineEmits(['update', 'busy'])
const busy = ref(false), error = ref(''), checks = ref([]), issues = ref([])
const sessionId = ref(''), turnId = ref('')
const contract = computed(() => props.row.case.answer_contract)
const machine = computed(() => props.row.machine_assessment)
const human = computed(() => props.row.human_assessment)
const ready = computed(() => props.row.generated || props.row.answer_ready)
const verdicts = { pass: '符合', partial: '部分符合', fail: '不符合 / 缺项', unjudged: '待判断' }
const overall = { correct: '符合全部标准', partial: '部分符合', incorrect: '存在关键错误', pending: '仍有待判断项' }
const evidenceLabels = { supported: '本轮证据足够', missing: '本轮证据不足', contradicted: '本轮证据与标准矛盾', not_applicable: '不适用', unjudged: '待判断' }
const diagnoses = { generation_error: '证据已提供，优先排查生成错误', generation_incomplete: '优先检查回答遗漏', evidence_gap: '先检查召回与资料覆盖', reference_conflict: '先复核参考标准与来源版本', pending: '仍待判断', ok: '本项未发现问题' }
function reset() {
  const previous = human.value || machine.value
  checks.value = contract.value.criteria.map(criterion => ({ criterion_id: criterion.id, verdict: 'unjudged', answer_quote: '', note: '', evidence_verdict: 'unjudged', evidence_quotes: [], ...JSON.parse(JSON.stringify(previous?.checks.find(c => c.criterion_id === criterion.id) || {})) }))
  issues.value = previous ? JSON.parse(JSON.stringify(previous.issues)) : []
  error.value = ''
}
watch(() => [props.runId, props.row.answer, machine.value?.created_at, human.value?.created_at], reset, { immediate: true })
async function request(path, body, method = 'POST') {
  busy.value = true; error.value = ''; emit('busy', true)
  try { emit('update', await api(`/evaluation/runs/${props.runId}/${path}/${props.row.case.id}`, body, { method })) }
  catch (e) { error.value = e.message }
  finally { busy.value = false; emit('busy', false) }
}
async function assist() {
  busy.value = true; error.value = ''; emit('busy', true)
  try { emit('update', await api(`/evaluation/runs/${props.runId}/assessments/${props.row.case.id}/machine`, {})) }
  catch (e) { error.value = e.message }
  finally { busy.value = false; emit('busy', false) }
}
function save() { request('assessments', { checks: checks.value, issues: issues.value, origin: 'user' }, 'PATCH') }
function capture() { request('capture', { session_id: sessionId.value.trim(), turn_id: turnId.value.trim() }) }
</script>

<template>
  <section class="item-assessment" aria-label="逐项回答评估">
    <div class="assessment-heading"><h3>逐项回答评估</h3><button v-if="ready && !machine" class="button" :disabled="disabled || busy" @click="assist">{{ busy ? '正在审阅…' : '机器辅助审阅' }}</button></div>
    <p class="muted">参考标准检查事实与必要条件；本轮生成证据检查引用支持。机器建议需要复核，关键错误不被其他通过项抵消。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <p v-if="row.capture" class="notice">这份回答来自已保存的真实会话，未重新生成。原会话未记录完整知识库版本，不能当成当前版本的新回答。</p>
    <p v-if="row.assessment_attempts?.at(-1)?.status === 'unavailable' && !machine" class="notice">机器审阅未完成或未通过结构核验，保留待复核；没有将失败计作零分或合格。</p>
    <p v-if="row.assessment_attempts?.at(-1)?.reason && !machine" class="muted">原因：{{ row.assessment_attempts.at(-1).reason }}</p>
    <p v-if="machine" class="assessment-result">{{ machine.origin === 'model' ? '机器建议' : '助手辅助审阅' }}：<strong>{{ overall[machine.overall] }}</strong> · {{ machine.passed }}/{{ machine.total }} 项符合，{{ machine.unjudged }} 项待判断。{{ machine.evaluator ? `审阅模型：${machine.evaluator}。` : '' }}该结果不计入人工正确率。</p>
    <p v-if="human" class="assessment-result">人工逐项复核：<strong>{{ overall[human.overall] }}</strong></p>
    <details v-if="row.assessment_history?.length" class="assessment-history"><summary>审阅记录与原始结果（{{ row.assessment_history.length }}）</summary><article v-for="(entry, i) in row.assessment_history" :key="i"><p>{{ ({ model:'机器建议', assistant:'助手辅助审阅', user:'用户逐项复核' })[entry.after.origin] }} · {{ new Date(entry.after.created_at).toLocaleString() }} · {{ overall[entry.after.overall] }}</p><p v-for="check in entry.after.checks.filter(c => c.verdict !== 'pass')" :key="check.criterion_id">{{ check.criterion_id }}：{{ check.note }}</p><details v-if="entry.after.diagnostics?.raw_content"><summary>查看原始机器 JSON</summary><pre>{{ entry.after.diagnostics.raw_content }}</pre></details></article></details>
    <details v-for="(criterion, i) in contract.criteria" :key="criterion.id" class="criterion-card" :open="ready && ((!human && !machine) || (human || machine).checks[i].verdict !== 'pass')">
      <summary><h4><span v-if="criterion.critical" class="critical-label">关键项</span>{{ criterion.text }}</h4></summary>
      <p v-if="(human || machine)?.checks?.[i]?.diagnosis" class="muted">排查提示：{{ diagnoses[(human || machine).checks[i].diagnosis] }}。该提示取决于本项审阅判断。</p>
      <p v-if="machine?.evidence_coverage?.[criterion.id]?.applicable" class="muted">预标注依据原句：{{ machine.evidence_coverage[criterion.id].annotated_span_present ? '已进入本轮生成节选' : '未在本轮生成节选中定位到' }}。此项只定位原句，不代表语义覆盖率。</p>
      <template v-if="ready">
        <div class="criterion-fields"><label>回答判断<select v-model="checks[i].verdict" :disabled="disabled || busy"><option v-for="(label, key) in verdicts" :key="key" :value="key">{{ label }}</option></select></label><label>本轮证据<select v-model="checks[i].evidence_verdict" :disabled="disabled || busy"><option v-for="(label, key) in evidenceLabels" :key="key" :value="key">{{ label }}</option></select></label></div>
        <label>回答原句<textarea v-model="checks[i].answer_quote" rows="2" maxlength="3000" :disabled="disabled || busy" placeholder="逐字摘录；回答缺少这一项时可留空" /></label>
        <label>判断依据<textarea v-model="checks[i].note" rows="2" maxlength="2000" :disabled="disabled || busy" placeholder="指出具体错误、遗漏条件或符合之处" /></label>
        <details><summary>本轮证据引文</summary><p class="muted">仅能引用下方“回答实际使用的证据”中的实际节选，参考答案不算检索证据。</p><div v-for="(quote, j) in checks[i].evidence_quotes" :key="j" class="quote-edit"><select v-model="quote.section_id" :disabled="disabled || busy" aria-label="证据章节"><option v-for="e in row.answer_evidence_details || []" :key="e.section_id" :value="e.section_id">{{ e.title }}</option></select><textarea v-model="quote.quote" rows="2" maxlength="3000" :disabled="disabled || busy" aria-label="证据原句" /><button class="button" :disabled="disabled || busy" @click="checks[i].evidence_quotes.splice(j, 1)">移除这条引文</button></div><button class="button" :disabled="disabled || busy" @click="checks[i].evidence_quotes.push({ section_id: row.answer_evidence_details?.[0]?.section_id || '', quote: '' })">添加证据引文</button></details>
      </template>
    </details>
    <details v-if="contract.sources?.length"><summary>参考标准的公开来源</summary><a v-for="url in contract.sources" :key="url" :href="url" target="_blank" rel="noopener noreferrer">{{ url }}</a></details>
    <template v-if="ready">
      <details class="additional-issues" :open="!!issues.length"><summary>回答额外引入的问题（{{ issues.length }}）</summary><p class="muted">用于记录标准之外的无依据结论、前后矛盾、答非所问或上下文丢失。</p><div v-for="(issue, i) in issues" :key="i" class="issue-edit"><label>问题类型<select v-model="issue.kind" :disabled="disabled || busy"><option value="unsupported">缺少证据</option><option value="contradiction">前后矛盾</option><option value="off_topic">答非所问</option><option value="context_loss">上下文丢失</option></select></label><label>回答原句<textarea v-model="issue.answer_quote" rows="2" :disabled="disabled || busy" /></label><label>原因<textarea v-model="issue.note" rows="2" :disabled="disabled || busy" /></label><label class="critical-check"><input v-model="issue.critical" type="checkbox" :disabled="disabled || busy" />影响核心结论</label><button class="button" :disabled="disabled || busy" @click="issues.splice(i, 1)">移除这项问题</button></div><button class="button" :disabled="disabled || busy" @click="issues.push({ kind:'unsupported', answer_quote:'', note:'', critical:false })">添加问题</button></details>
      <button class="button primary" :disabled="disabled || busy || checks.some(c => !c.note.trim())" @click="save">保存人工逐项复核</button>
    </template>
    <details v-else class="capture-answer"><summary>导入已保存的失败回答</summary><p class="muted">指定已有会话及回合，服务会核对问题与前序是否一致；保留原回答，不重新抽样。</p><label>会话 ID<input v-model="sessionId" :disabled="disabled || busy" /></label><label>回合 ID<input v-model="turnId" :disabled="disabled || busy" /></label><button class="button" :disabled="disabled || busy || !sessionId.trim() || !turnId.trim()" @click="capture">导入该回答</button></details>
  </section>
</template>

<style scoped>
.item-assessment { margin:20px 0; padding:20px; border:1px solid #dce4f3; border-radius:10px; background:#fbfcff; min-width:0; }
.assessment-heading,.criterion-fields { display:flex; gap:14px; justify-content:space-between; align-items:start; flex-wrap:wrap; }
.assessment-heading h3 { margin:0; }
.criterion-card { padding:15px 0; border-top:1px solid var(--line); }
.criterion-card h4 { margin:0 0 12px; line-height:1.8; font-weight:550; }
.criterion-card>summary h4 { display:inline; font-size:13px; color:#33405b; }
.critical-label { display:inline-block; margin-right:8px; padding:1px 6px; background:#edf1fb; color:#526b9b; border-radius:4px; font-size:11px; }
label { display:grid; gap:6px; margin:10px 0; font-size:12px; color:#55647c; min-width:0; }
.criterion-fields>label { flex:1; min-width:160px; }
textarea,select,input:not([type=checkbox]) { box-sizing:border-box; width:100%; min-width:0; font:inherit; color:#33405b; border:1px solid var(--line); border-radius:6px; padding:8px 10px; background:#fff; }
textarea { resize:vertical; line-height:1.7; }
details { margin:10px 0; font-size:12px; }
summary { cursor:pointer; color:#52688e; padding:5px 0; }
a { display:block; overflow-wrap:anywhere; margin:8px 0; }
.quote-edit,.issue-edit { padding:12px; margin:8px 0; border:1px solid var(--line); border-radius:6px; background:#fff; }
.assessment-history article { border-top:1px solid var(--line); padding:8px 0; }
.assessment-history pre { max-height:320px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; padding:12px; background:#f2f5fa; }
.critical-check { display:flex; align-items:center; gap:7px; }
.assessment-result { line-height:1.9; font-size:13px; }
@media(max-width:600px) { .item-assessment { padding:14px; }.criterion-fields { display:block; } }
</style>
