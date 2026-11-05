<script setup>
import { userStorageKey } from '../auth.js'
import { computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, chatStream } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'
import AnswerSupport from '../components/AnswerSupport.vue'
import { evidenceTargets } from '../citations.js'
import SessionHistory from '../components/SessionHistory.vue'
import { conversationContext, mergeMessages } from '../sessionHistory'
import QueryPlan from '../components/QueryPlan.vue'
import { recentModels } from '../activityContext'
import DocumentPicker from '../components/DocumentPicker.vue'
import { shouldSendOnEnter } from '../composerKeys'
import LearningTrail from '../components/LearningTrail.vue'
import { AE_FOLLOWUPS } from '../aeJourney'

const props = defineProps({ embedded: { type: Boolean, default: false }, readingContext: { type: Object, default: null } })
const emit = defineEmits(['busy'])
const route = useRoute(), router = useRouter()
const developerMode = inject('developerMode', ref(false))
const models = ref([]), sections = ref([]), modelId = ref(''), sectionId = ref(''), mode = ref('learning')
const documentId = ref('')
const documentIds = ref([]), documentPolicy = ref('auto'), documents = ref([])
async function refreshDocuments() { documents.value = await api('/documents') }
function restoreDocuments(metadata = {}) {
  documentIds.value = metadata.document_ids || (metadata.document_id ? [metadata.document_id] : [])
  documentId.value = documentIds.value.length === 1 ? documentIds.value[0] : ''
  documentPolicy.value = metadata.document_policy || 'auto'
}
async function selectDocuments(ids) {
  documentIds.value = ids; documentId.value = ids.length === 1 ? ids[0] : ''; sectionId.value = ''
  const selected = documents.value.filter(d => ids.includes(d.id))
  modelId.value = selected.length === 1 ? selected[0].model_id : ''
  await loadSections(modelId.value)
  if (!props.embedded) await router.replace({ path: '/chat', query: { ...route.query,
    model: selected.length === 1 ? selected[0].model_id : undefined,
    document: ids.length === 1 ? ids[0] : undefined, section: undefined } })
}
function composerKeydown(event) {
  if (!shouldSendOnEnter(event)) return
  event.preventDefault()
  if (!navigationDisabled.value) submit()
}
const question = ref(''), messages = ref([]), busy = ref(false), error = ref(''), end = ref(null)
const sessionStorageKey = userStorageKey(props.embedded ? `model-atlas-reader-session:${props.readingContext?.modelId || 'general'}` : 'model-atlas-session')
const sessionId = ref(localStorage.getItem(sessionStorageKey) || '')
const questionInputId = props.embedded ? 'reader-question' : 'question'
const draft = ref(''), stage = ref(''), stopping = ref(false)
const livePlan = ref(null), conversationBox = ref(null), useBrowsing = ref(false)
const browsing = ref(recentModels())
const browsingContext = computed(() => browsing.value.filter(a => models.value.some(m => m.id === a.model_id)))
const restoring = ref(true)
const selectedEvidence = ref('')
const historyOpen = ref(false), historyRefresh = ref(0), sessionTitle = ref(''), savedQuestions = ref(0)
const olderCursor = ref(null), hasEarlier = ref(false), loadingEarlier = ref(false)
const navigationDisabled = computed(() => busy.value || restoring.value || loadingEarlier.value)
watch(navigationDisabled, value => emit('busy', value), { immediate: true })
const currentSection = computed(() => sections.value.find(s => s.id === sectionId.value))
const questionCount = computed(() => Math.max(savedQuestions.value, messages.value.filter(m => m.role === 'user').length))
let controller = null, pageController = null
function applySavedPage(page, retainOlder = false) {
  messages.value = retainOlder ? mergeMessages(messages.value.filter(message => message.id), page.messages) : page.messages
  sessionTitle.value = page.session.title; savedQuestions.value = page.session.question_count
  hasEarlier.value = messages.value.length < savedQuestions.value * 2
  olderCursor.value = hasEarlier.value ? messages.value[0]?.sequence : null
}
async function readSavedSession(id, keepRoute = false) {
  pageController?.abort(); pageController = new AbortController()
  const signal = pageController.signal
  const page = await api(`/sessions/${encodeURIComponent(id)}/messages`, undefined, { signal })
  const context = conversationContext(page.messages, models.value)
  const useSavedContext = !keepRoute || !(props.embedded ? props.readingContext?.modelId : route.query.model)
  const detail = useSavedContext && context.modelId ? await api(`/models/${encodeURIComponent(context.modelId)}`, undefined, { signal }) : null
  if (signal.aborted) return false
  applySavedPage(page); selectedEvidence.value = ''
  if (useSavedContext) {
    restoreDocuments([...page.messages].reverse().find(m => m.role === 'assistant')?.metadata)
    modelId.value = context.modelId; mode.value = context.mode; sections.value = detail?.sections || []
    sectionId.value = sections.value.some(section => section.id === context.sectionId) ? context.sectionId : ''
  }
  sessionId.value = id; localStorage.setItem(sessionStorageKey, id)
  return true
}
async function openSession(item) {
  if (navigationDisabled.value) return
  restoring.value = true; error.value = ''
  try {
    if (await readSavedSession(item.id)) {
      question.value = ''; historyOpen.value = false
      await router.replace({ path: '/chat' }); await showLatestAnswer()
    }
  } catch (e) { if (e.name !== 'AbortError') error.value = `会话打开失败：${e.message}` }
  finally { restoring.value = false }
}
function renamedSession(result) { if (result.id === sessionId.value) sessionTitle.value = result.title }
async function loadEarlier() {
  if (navigationDisabled.value || !hasEarlier.value) return
  loadingEarlier.value = true; error.value = ''; pageController = new AbortController()
  try {
    const page = await api(`/sessions/${encodeURIComponent(sessionId.value)}/messages?before=${olderCursor.value}`, undefined, { signal: pageController.signal })
    if (!pageController.signal.aborted) {
      messages.value = mergeMessages(page.messages, messages.value)
      hasEarlier.value = page.has_more; olderCursor.value = page.next_cursor
    }
  } catch (e) { if (e.name !== 'AbortError') error.value = e.message }
  finally { loadingEarlier.value = false }
}
async function showEvidence(message, position, number) {
  const anchor = evidenceTargets(message, position)[number - 1]?.slice(1)
  if (!anchor) return
  const element = document.getElementById(anchor)
  const panel = element?.closest('details')
  if (!panel) return
  panel.open = true; selectedEvidence.value = anchor
  await nextTick()
  if (props.embedded && conversationBox.value) {
    const box = conversationBox.value
    box.scrollTo({ top: box.scrollTop + element.getBoundingClientRect().top - box.getBoundingClientRect().top - 16, behavior: 'smooth' })
  } else element.scrollIntoView({ behavior: 'smooth', block: 'center' })
  element.focus({ preventScroll: true })
}
const selectedModel = computed(() => models.value.find(m => m.id === modelId.value))
const suggestions = computed(() => selectedModel.value ? [
  { label: '理解模型', title: '解释当前阅读内容', question: `请解释 ${selectedModel.value.name}${currentSection.value ? ` 的${currentSection.value.title}` : ' 的输入、核心结构与输出'}，说明必要术语并给出资料依据。` },
  { label: '结构与代码', title: '把结构和代码对应起来', question: `${selectedModel.value.name} 的核心结构在已入库代码中是怎样实现的？请结合输入输出说明。` },
  { label: '模型关系', title: '寻找相关模型与模块', question: `${selectedModel.value.name} 与哪些已入库模型或模块有联系？请结合已有关系和资料说明。` },
] : [
  { label: '模型解释', title: '理解 CNN 的输入与输出', question: '1D CNN 的输入为什么是 [B, C, L]？输出的通道和长度分别由什么决定？' },
  { label: '比较分析', title: '比较 ResNet-18 与 ViT', question: '请简短回答：用表格比较 ResNet-18 与 ViT 的输入、核心结构和输出表示，不展开推导。' },
  { label: '模型关系', title: 'CLIP 和 ViT 有什么联系', question: 'CLIP 与 ViT 有什么联系？请沿视觉分支解释中间的模型关系，并给出每一步的资料依据。' },
])
async function chooseQuestion(value) {
  question.value = value
  await nextTick()
  document.getElementById(questionInputId)?.focus()
}
async function loadSections(id, desired = '') {
  sections.value = []; sectionId.value = ''
  if (id) {
    const detail = await api(`/models/${encodeURIComponent(id)}`)
    sections.value = detail.sections
    if (sections.value.some(s => s.id === desired)) sectionId.value = desired
  }
}
async function applyRoute() {
  if (props.embedded) {
    modelId.value = props.readingContext?.modelId || ''
    await loadSections(modelId.value, props.readingContext?.sectionId || '')
    return
  }
  if (!busy.value && typeof route.query.session === 'string' && route.query.session !== sessionId.value) {
    await readSavedSession(route.query.session)
  }
  if (Object.hasOwn(route.query, 'model')) {
    documentId.value = typeof route.query.document === 'string' ? route.query.document : ''
    documentIds.value = documentId.value ? [documentId.value] : []
    modelId.value = String(route.query.model || '')
    await loadSections(modelId.value, String(route.query.section || ''))
  }
  if (!busy.value && typeof route.query.question === 'string') question.value = route.query.question.slice(0, 2000)
}
async function changeModel() {
  documentId.value = ''
  documentIds.value = []; documentPolicy.value = 'auto'
  try { await loadSections(modelId.value) } catch (e) { error.value = e.message }
}
async function scrollEnd() { await nextTick(); conversationBox.value?.scrollTo({ top: conversationBox.value.scrollHeight, behavior: 'smooth' }) }
async function showLatestAnswer() {
  await nextTick()
  const box = conversationBox.value
  const answer = box?.querySelector('.message.assistant:last-child')
  if (answer) box.scrollTo({ top: box.scrollTop + answer.getBoundingClientRect().top - box.getBoundingClientRect().top - 8, behavior: 'smooth' })
}
async function newSession() {
  if (navigationDisabled.value) return
  sessionId.value = ''; localStorage.removeItem(sessionStorageKey); messages.value = []; error.value = ''
  modelId.value = ''; sectionId.value = ''; sections.value = []; question.value = ''
  documentId.value = ''
  documentIds.value = []; documentPolicy.value = 'auto'
  sessionTitle.value = ''; savedQuestions.value = 0; olderCursor.value = null; hasEarlier.value = false; selectedEvidence.value = ''
  if (props.embedded) {
    restoring.value = true
    try { await applyRoute() } catch (e) { error.value = e.message }
    finally { restoring.value = false }
  } else await router.replace({ path: '/chat' })
  await nextTick()
  conversationBox.value?.scrollTo({ top: 0 })
}
async function submit(text = question.value) {
  const value = text.trim()
  if (!value || navigationDisabled.value) return
  busy.value = true; stopping.value = false; error.value = ''; question.value = ''; draft.value = ''; livePlan.value = null; stage.value = '正在连接知识库'
  controller = new AbortController()
  // 请求前确定会话 ID；完整回答保存后即使页面断开，也能用同一 ID 恢复。
  sessionId.value ||= crypto.randomUUID()
  localStorage.setItem(sessionStorageKey, sessionId.value)
  const userIndex = messages.value.length
  const turnId = crypto.randomUUID()
  let completed = false
  messages.value.push({ role: 'user', content: value }); await scrollEnd()
  try {
    const result = await chatStream({ question: value, model_id: modelId.value || null, document_ids: documentIds.value, document_policy: documentPolicy.value, section_id: sectionId.value || null, mode: mode.value, session_id: sessionId.value, turn_id: turnId, activities: useBrowsing.value ? browsingContext.value : [] }, (name, data) => {
      if (name === 'stage') stage.value = data.stage === 'evidence_check' ? data.label : ({ history: '正在读取对话…', planning: '正在理解问题…', retrieving: '正在查找资料…', generating: '正在组织回答…', reviewing: '正在核对回答…', saving: '正在保存对话…' })[data.stage] || '正在处理…'
      if (name === 'delta') draft.value += data.text
      if (name === 'plan') livePlan.value = data
    }, controller.signal)
    completed = true
    sessionId.value = result.session_id; localStorage.setItem(sessionStorageKey, result.session_id)
    restoreDocuments(result)
    messages.value.push({ role: 'assistant', content: result.answer, metadata: result })
    if (modelId.value !== (result.model_id || '') || sectionId.value !== (result.section_id || '')) {
      modelId.value = result.model_id || ''
      await loadSections(modelId.value, result.section_id || '')
    }
    applySavedPage(await api(`/sessions/${sessionId.value}/messages`), true)
  } catch (e) {
    if (completed) {
      error.value = `回答已保存，页面更新失败：${e.message}`
    } else {
      // 断开时后端可能刚完成提交，先恢复完整会话，避免误删已保存的答案。
      const saved = await api(`/sessions/${sessionId.value}/messages`).catch(() => null)
      const last = saved?.messages?.at(-1)
      if (last?.role === 'assistant' && last.metadata?.turn_id === turnId) {
        applySavedPage(saved, true); completed = true
        restoreDocuments(last.metadata)
        modelId.value = last.metadata.model_id || ''
        await loadSections(modelId.value, last.metadata.section_id || '')
        error.value = '连接已结束，已恢复后端保存的完整回答。'
      } else {
        messages.value.splice(userIndex)
        question.value = value
        error.value = e.name === 'AbortError' ? '已停止本轮请求，可以修改问题后重新发送。' : e.message
      }
    }
  }
  finally { busy.value = false; stopping.value = false; controller = null; draft.value = ''; stage.value = ''; if (completed) { historyRefresh.value++; await showLatestAnswer() } else await scrollEnd() }
}
function stopAnswer() { stopping.value = true; controller?.abort() }
defineExpose({ showLatestAnswer })
onBeforeUnmount(() => { controller?.abort(); pageController?.abort() })
onMounted(async () => {
  try {
    models.value = (await api('/models')).filter(m => m.searchable || m.ready)
    await refreshDocuments()
    if (!props.embedded && typeof route.query.session === 'string') {
      await readSavedSession(route.query.session)
    } else if (!props.embedded && route.query.start === 'new') {
      sessionId.value = ''; localStorage.removeItem(sessionStorageKey)
      await router.replace({ path: '/chat', query: { ...route.query, start: undefined } })
    } else if (sessionId.value) {
      await readSavedSession(sessionId.value, true)
    }
    await applyRoute()
    if (messages.value.length) await showLatestAnswer()
  } catch (e) { error.value = e.message }
  finally { restoring.value = false }
})
watch(() => route.query, async () => { if (!props.embedded) { try { await applyRoute() } catch (e) { error.value = e.message } } })
watch(() => props.readingContext, async () => {
  if (!props.embedded || navigationDisabled.value) return
  restoring.value = true; error.value = ''
  try { await applyRoute() } catch (e) { error.value = e.message }
  finally { restoring.value = false }
})
</script>

<template>
  <section :class="['chat-page', { 'embedded-chat': embedded }]">
    <LearningTrail v-if="!embedded && route.query.flow === 'ae'" stage="followup" />
    <div class="page-top chat-heading"><div v-if="!embedded"><h1>智能问答</h1><p>理解模型、比较结构，沿着引用继续探索。</p></div><span v-else class="reader-chat-note">连续追问会保留本次对话。</span><div class="chat-actions"><button v-if="!embedded" class="button" :aria-expanded="historyOpen" aria-controls="session-history" @click="historyOpen = !historyOpen">历史会话</button><button class="button" :disabled="navigationDisabled" @click="newSession">新建会话</button></div></div>
    <div v-if="historyOpen" class="history-overlay"><div class="form-actions"><button class="button" @click="historyOpen = false">关闭历史列表</button></div><SessionHistory :current-id="sessionId" :disabled="navigationDisabled" :refresh="historyRefresh" :models="models" @select="openSession" @renamed="renamedSession" /></div>
    <div v-if="restoring || messages.length || selectedModel" class="session-context"><div class="session-summary"><strong :title="sessionTitle">{{ restoring ? '正在恢复会话…' : sessionTitle || '当前会话' }}</strong><small v-if="developerMode && questionCount">{{ questionCount }} 次提问</small><span v-if="documentIds.length" class="scope-label">文献范围 · {{ documentIds.length }} 篇</span><span v-else-if="selectedModel" class="scope-label">{{ selectedModel.name }}{{ currentSection ? ` / ${currentSection.title}` : '' }}</span></div><RouterLink v-if="(selectedModel || documentIds.length) && !embedded" :to="{ path: documentIds.length > 1 ? '/documents' : documentId ? `/documents/${documentId}` : `/models/${modelId}`, query: !documentId && currentSection ? { section: currentSection.slug } : {} }">查看当前资料 →</RouterLink></div>
    <details class="chat-settings"><summary>回答偏好 <span>{{ mode === 'learning' ? '入门讲解' : '深入分析' }}</span></summary><div class="chat-context"><div class="mode-switch"><button :class="{ active: mode === 'learning' }" :disabled="navigationDisabled" @click="mode = 'learning'">入门讲解</button><button :class="{ active: mode === 'research' }" :disabled="navigationDisabled" @click="mode = 'research'" title="复杂问题会额外核对，等待可能更长">深入分析</button></div><label>模型范围<select v-model="modelId" :disabled="navigationDisabled" @change="changeModel" aria-label="当前模型"><option value="">根据问题和对话判断</option><option v-for="model in models" :key="model.id" :value="model.id">{{ model.name }}</option></select></label><label>阅读章节<select v-model="sectionId" :disabled="navigationDisabled || !modelId" @change="selectedEvidence = ''" aria-label="当前章节"><option value="">全部章节</option><option v-for="section in sections" :key="section.id" :value="section.id">{{ section.title }}</option></select></label></div><label class="checkbox-label"><input v-model="useBrowsing" type="checkbox" :disabled="navigationDisabled" />参考最近浏览的模型</label><small v-if="useBrowsing">{{ browsingContext.map(a => models.find(m => m.id === a.model_id)?.name).join('、') || '尚无可用浏览记录' }} · 明确提问和会话对象优先</small></details>
    <details v-if="!embedded && route.query.flow === 'ae' && messages.some(m => m.metadata?.research_run_id)" class="followup-guide"><summary>围绕刚才的比较，可以接着问</summary><p>比较结果与引用已保留。点击下列问题会填入输入框，由你确认发送。</p><button v-for="item in AE_FOLLOWUPS" :key="item" type="button" :disabled="navigationDisabled" @click="chooseQuestion(item)">{{ item }}</button></details>
    <div ref="conversationBox" class="chat-body">
    <div v-if="!messages.length && !restoring" class="chat-welcome">
      <span v-if="!embedded" class="welcome-mark" aria-hidden="true">M</span><h2><template v-if="embedded">哪里还没看明白？</template><template v-else>把模型里的问题，<br /><em>一步步问明白。</em></template></h2><p>{{ embedded ? '可以要求举例、解释代码，也可以接着上一轮追问。' : '从原理、结构或代码出发，用引用核对答案，接着上一轮继续提问。' }}</p>
      <div v-if="!question.trim()" class="starter-grid"><button v-for="suggestion in embedded ? suggestions.slice(0, 1) : suggestions" :key="suggestion.label" type="button" class="starter-card" :disabled="navigationDisabled" @click="chooseQuestion(suggestion.question)"><small v-if="!embedded" class="starter-label">{{ suggestion.label }}</small><strong>{{ suggestion.title }}</strong><small v-if="!embedded" class="starter-preview">{{ suggestion.question }}</small><span aria-hidden="true">↗</span></button></div>
      <div v-else class="draft-ready">问题已准备好，点击下方“发送”开始。</div>
    </div>
    <div v-else class="conversation">
      <div v-if="hasEarlier" class="earlier-messages"><button class="button" :disabled="navigationDisabled" @click="loadEarlier">{{ loadingEarlier ? '正在加载…' : '加载更早问答' }}</button><small>已显示 {{ messages.filter(message => message.role === 'user').length }} / {{ savedQuestions }} 次已保存提问</small></div>
      <article v-for="(message, i) in messages" :key="message.id || i" :class="['message', message.role]">
        <span class="message-label">{{ message.role === 'user' ? '你' : 'Model Atlas' }}</span>
        <p v-if="message.role === 'user'">{{ message.content }}</p>
        <template v-else>
          <MarkdownContent :content="message.content" :citation-targets="evidenceTargets(message, i)" @citation="number => showEvidence(message, i, number)" />
          <AnswerSupport v-if="message.metadata" :metadata="message.metadata" :targets="evidenceTargets(message, i)" :selected-evidence="selectedEvidence" @evidence="number => showEvidence(message, i, number)" />
        </template>
      </article>
      <article v-if="busy" class="message assistant live-answer" aria-busy="true">
        <span class="message-label">MODEL ATLAS · {{ stopping ? '正在停止' : '处理中' }}</span>
        <div class="pending-answer" role="status"><span class="spinner" />{{ stage }}</div>
        <p v-if="developerMode && livePlan && !draft" class="live-query">本轮检索：{{ livePlan.normalized_query }}</p>
        <p v-if="draft" class="stream-draft">{{ draft }}</p>
        <QueryPlan v-if="developerMode && livePlan" :plan="livePlan" />
        <small v-if="draft">回答生成中，完成后可查看引用来源。</small>
      </article>
    </div>
    <p v-if="error" class="notice error">{{ error }}</p>
    </div>
    <form ref="end" class="composer" @submit.prevent="submit()"><div class="composer-documents"><DocumentPicker :model-value="documentIds" :documents="documents" :disabled="navigationDisabled" @update:model-value="selectDocuments" @refresh="refreshDocuments" /><select v-if="documentIds.length" v-model="documentPolicy" :disabled="navigationDisabled" aria-label="资料补查方式"><option value="auto">允许补充查找知识库</option><option value="strict">仅查所选资料</option></select></div><label :for="questionInputId">输入你的问题</label><textarea :id="questionInputId" v-model="question" :disabled="navigationDisabled" maxlength="2000" rows="1" placeholder="询问原理、公式或代码，也可以继续追问…" @keydown="composerKeydown" /><div><small>Enter 发送 · Shift + Enter 换行</small><button v-if="busy" type="button" class="button" :disabled="stopping" @click="stopAnswer">{{ stopping ? '正在停止…' : '停止回答' }}</button><button v-else class="button primary" :disabled="!question.trim() || navigationDisabled">发送 →</button></div></form>
  </section>
</template>

<style scoped>
.followup-guide{border:1px solid var(--line);border-radius:9px;padding:13px 16px;margin:0 0 12px;background:white;font-size:12px}.followup-guide summary{cursor:pointer;color:#4660a1}.followup-guide p{color:var(--muted);line-height:1.7}.followup-guide button{display:block;border:0;border-top:1px solid #edf0f6;background:none;text-align:left;width:100%;padding:10px 0;font:inherit;color:#546889;cursor:pointer;line-height:1.8}
.composer-documents{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;min-width:0}.composer-documents>select{font-size:11px;color:#526991;border:1px solid var(--line);border-radius:6px;padding:7px;max-width:100%;background:white}
.embedded-chat { width:100%; height:100%; min-width:0; }
.embedded-chat .chat-heading { justify-content:space-between; margin:0; }
.reader-chat-note { color:var(--muted); font-size:11px; }
.embedded-chat .session-context { padding:9px 0; }
.embedded-chat .session-summary strong { display:none; }
.embedded-chat .chat-welcome { margin:16px auto; }
.embedded-chat .chat-welcome h2 { font-size:18px; }
.embedded-chat .message { padding:15px; min-width:0; }
.embedded-chat .message.user { margin-left:8%; }
.embedded-chat .composer { padding:9px 12px; gap:4px 8px; }
.embedded-chat .composer .button { padding:7px 10px; }
.embedded-chat :deep(.markdown) { font-size:13px; overflow-wrap:anywhere; }
.chat-actions { display:flex; gap:10px; }.session-context strong { display:inline-block; max-width:440px; vertical-align:bottom; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.session-summary { display:flex; align-items:center; gap:10px; min-width:0; flex-wrap:wrap; }
.session-summary small { color:var(--muted); font-size:11px; }
.scope-label { font-size:11px; color:#536992; border:1px solid #e2e8f3; border-radius:5px; padding:2px 8px; background:#f5f7fc; }
.chat-welcome { max-width:850px; margin:10px auto 8px; padding:0; }
.welcome-mark { display:inline-grid; place-items:center; width:42px; height:42px; border:1px solid #dce5fb; border-radius:14px; background:linear-gradient(145deg,#e4edff,#f4effb); color:#526dc7; font-family:var(--font-display); font-size:22px; font-weight:650; }
.chat-welcome h2 { margin:14px 0 13px; color:#2c3c5a; font-size:30px; line-height:1.4; letter-spacing:-.8px; font-weight:700; }
.chat-welcome h2 em { font-style:normal; color:#6b71b3; }
.chat-welcome>p { font-size:12px; color:#69768e; margin-bottom:22px; }
.starter-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; text-align:left; }
.chat-welcome .starter-card { --starter-accent:#5279bb; display:flex; flex-direction:column; align-items:start; justify-content:start; position:relative; gap:11px; margin:0; padding:20px; min-height:145px; border:1px solid #e0e6f0; border-radius:13px; background:linear-gradient(150deg,#fff,#f5f8fe); }
.chat-welcome .starter-card:nth-child(2) { --starter-accent:#7b5c9f; background:linear-gradient(150deg,#fff,#f8f5fc); }
.chat-welcome .starter-card:nth-child(3) { --starter-accent:#247c76; background:linear-gradient(150deg,#fff,#f2f9f7); }
.starter-card:hover { border-color:#b0c2ea; }
.starter-card .starter-label { color:var(--starter-accent); font-size:10px; letter-spacing:1px; }
.starter-card strong { color:#3b4c6b; font-size:14px; line-height:1.7; font-weight:600; padding-right:10px; }
.starter-card .starter-preview { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; color:#69788f; font-size:11px; line-height:1.8; }
.starter-card>span { position:absolute; right:15px; top:18px; color:var(--starter-accent); font-size:14px; }
.chat-page:not(.embedded-chat) .chat-heading h1 { font-size:24px; font-weight:650; letter-spacing:-.5px; }
.chat-page:not(.embedded-chat) .chat-settings { margin-top:3px; padding:10px 15px; background:#eef2f87a; border:1px solid #e9edf5; border-radius:9px; }
.chat-page:not(.embedded-chat) .message.user { width:fit-content; max-width:86%; margin-left:auto; padding:17px 23px; color:#fff; background:#3c568f; border:0; border-radius:17px 17px 4px 17px; box-shadow:0 3px 9px #263d6b0b; }
.chat-page:not(.embedded-chat) .message.user .message-label { color:#d6e1f5; font-size:10px; }
.chat-page:not(.embedded-chat) .message.user p { font-size:14px; line-height:1.85; }
.chat-page:not(.embedded-chat) .message.assistant { border-radius:15px; padding:25px 28px; border-color:#e2e7f1; box-shadow:0 4px 16px #29406506; }
.chat-page:not(.embedded-chat) .message.assistant>.message-label { display:flex; align-items:center; gap:8px; font-family:var(--font-display); color:#45629c; font-size:13px; font-weight:600; }
.chat-page:not(.embedded-chat) .message.assistant>.message-label::before { content:""; width:5px; height:15px; background:#8aa7d7; border-radius:3px; }
.chat-page:not(.embedded-chat) .composer { margin-top:14px; padding:16px 19px; border-radius:17px; border-color:#cfd9ed; box-shadow:0 5px 25px #2e497b0b; }
.chat-page:not(.embedded-chat) .composer:focus-within { border-color:#7d96cd; box-shadow:0 0 0 3px #597acc0c; }
.chat-page:not(.embedded-chat) .composer textarea { min-height:48px; font-size:14px; }
.embedded-chat .starter-grid { grid-template-columns:1fr; }
.embedded-chat .starter-card { display:flex; flex-direction:row; align-items:center; padding:13px 16px; min-height:0; }
.embedded-chat .starter-card>span { position:static; margin-left:auto; }
.draft-ready { max-width:480px; margin:20px auto; padding:15px; border:1px dashed #d8e1f4; border-radius:10px; color:var(--muted); font-size:12px; }
.earlier-messages { display:flex; gap:14px; align-items:center; justify-content:center; padding:16px 0; }.earlier-messages small { color:var(--muted); font-size:12px; }
@media(max-width:760px) { .chat-actions { gap:6px; }.session-context strong { max-width:230px; }.earlier-messages { flex-direction:column; gap:8px; }.starter-grid { grid-template-columns:1fr; gap:9px; }.chat-welcome .starter-card { min-height:0; gap:7px; padding:16px 18px; }.starter-card .starter-preview { -webkit-line-clamp:1; }.chat-welcome { margin-top:14px; }.chat-welcome h2 { font-size:27px; }.chat-welcome>p { font-size:11px; margin-bottom:22px; }.chat-page:not(.embedded-chat) .message.assistant { padding:18px; }.chat-page:not(.embedded-chat) .composer { margin-top:8px; padding:12px 14px; }.chat-page:not(.embedded-chat) .composer textarea { min-height:42px; } }
</style>
