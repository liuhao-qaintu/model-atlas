<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { sessionListPath } from '../sessionHistory'

const props = defineProps({ currentId: String, disabled: Boolean, refresh: Number, models: Array })
const emit = defineEmits(['select', 'renamed'])
const items = ref([]), search = ref(''), activeQuery = ref(''), cursor = ref(null), hasMore = ref(false)
const loading = ref(false), error = ref(''), notice = ref(''), editing = ref(''), title = ref(''), saving = ref(false)
const modelNames = computed(() => Object.fromEntries((props.models || []).map(model => [model.id, model.name])))
let controller = null, requestNumber = 0
function dateLabel(value) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}
async function load(more = false) {
  controller?.abort(); controller = new AbortController()
  const request = ++requestNumber
  loading.value = true; error.value = ''
  try {
    const result = await api(sessionListPath(activeQuery.value, more ? cursor.value : null), undefined, { signal: controller.signal })
    if (request !== requestNumber) return
    items.value = [...new Map([...(more ? items.value : []), ...result.items].map(item => [item.id, item])).values()]
    cursor.value = result.next_cursor; hasMore.value = result.has_more
  } catch (e) {
    if (request === requestNumber && e.name !== 'AbortError') error.value = e.message
  } finally { if (request === requestNumber) loading.value = false }
}
function searchSessions() { activeQuery.value = search.value; notice.value = ''; load() }
function edit(item) { editing.value = item.id; title.value = item.title; notice.value = ''; error.value = '' }
async function saveTitle() {
  if (saving.value || props.disabled || !title.value.trim()) return
  const id = editing.value
  saving.value = true; error.value = ''
  try {
    const result = await api(`/sessions/${encodeURIComponent(id)}`, { title: title.value.trim() }, { method: 'PATCH' })
    items.value = items.value.map(item => item.id === id ? { ...item, title: result.title } : item)
    editing.value = ''; notice.value = '会话标题已更新'; emit('renamed', result)
    if (activeQuery.value) await load()
  } catch (e) { error.value = e.message }
  finally { saving.value = false }
}
onMounted(() => load())
watch(() => props.refresh, () => load())
onBeforeUnmount(() => { requestNumber++; controller?.abort() })
</script>

<template>
  <section id="session-history" class="history-panel" aria-label="历史会话">
    <div class="history-heading"><h2>历史会话</h2><small>按最近一次问答排序</small></div>
    <form class="history-search" @submit.prevent="searchSessions"><label for="history-query">搜索标题或最后一次提问</label><div><input id="history-query" v-model="search" maxlength="80" placeholder="输入关键词" :disabled="saving" /><button class="button" :disabled="saving">搜索</button><button type="button" class="button" :disabled="loading || saving" @click="load()">刷新列表</button></div></form>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="notice" class="history-notice" role="status">{{ notice }}</p>
    <p v-if="loading && !items.length" class="history-empty" role="status">正在读取已保存会话…</p>
    <p v-else-if="!items.length" class="history-empty">{{ activeQuery ? '没有匹配的会话，可以换一个关键词。' : '暂无已保存会话，完成第一次问答后会自动出现在这里。' }}</p>
    <div v-if="items.length" :class="['history-list', { single: items.length === 1 }]" :aria-busy="loading">
      <article v-for="item in items" :key="item.id" :class="['history-item', { selected: item.id === currentId }]">
        <form v-if="editing === item.id" class="history-edit" @submit.prevent="saveTitle"><label :for="`title-${item.id}`">会话标题</label><input :id="`title-${item.id}`" v-model="title" maxlength="120" :disabled="saving || disabled" /><div><button class="button primary" :disabled="saving || disabled || !title.trim()">{{ saving ? '保存中…' : '保存标题' }}</button><button type="button" class="button" :disabled="saving" @click="editing = ''">取消</button></div></form>
        <template v-else><button class="history-open" :disabled="disabled || saving || loading" :aria-current="item.id === currentId ? 'true' : undefined" @click="emit('select', item)"><span class="history-title" :title="item.title">{{ item.title }}</span><span class="history-preview">{{ item.last_question }}</span><span class="history-meta">{{ item.question_count }} 次提问 · {{ modelNames[item.model_id] || '自动识别' }} · {{ dateLabel(item.updated_at) }}</span></button><div class="history-footer"><span>{{ item.id === currentId ? '当前会话' : '打开后继续追问' }}</span><button type="button" :disabled="disabled || saving || loading" :aria-label="`重命名会话：${item.title}`" @click="edit(item)">重命名</button></div></template>
      </article>
    </div>
    <button v-if="hasMore" class="button history-more" :disabled="loading || saving" @click="load(true)">{{ loading ? '加载中…' : '加载更多会话' }}</button>
  </section>
</template>

<style scoped>
.history-panel { margin:22px 0; padding:22px; background:white; border:1px solid var(--line); border-radius:12px; }
.history-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; }.history-heading h2 { margin:0; font-size:18px; }.history-heading small { color:var(--muted); font-size:12px; }
.history-search { margin:18px 0; }.history-search label,.history-edit label { display:block; font-size:12px; color:var(--muted); margin-bottom:7px; }.history-search > div { display:flex; gap:8px; }
input { min-width:0; width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:white; }.history-search input { flex:1; }.history-search .button { flex-shrink:0; }
.history-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; max-height:370px; overflow:auto; padding:2px; }
.history-list.single { grid-template-columns:1fr; }
.history-item { border:1px solid var(--line); border-radius:10px; overflow:hidden; min-width:0; }.history-item.selected { border-color:#2563eb; background:#f4f8ff; }
.history-open { display:grid; gap:8px; text-align:left; width:100%; padding:15px 16px 10px; background:transparent; }.history-title { font-weight:600; white-space:nowrap; text-overflow:ellipsis; overflow:hidden; }.history-preview { color:#64748b; font-size:12px; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow-wrap:anywhere; line-height:1.7; }.history-meta { font-size:11px; color:var(--muted); }
.history-footer { display:flex; justify-content:space-between; gap:8px; align-items:center; padding:0 16px 13px; font-size:11px; color:#64748b; }.history-footer button { padding:4px 0 4px 8px; color:#2563eb; background:transparent; font-size:12px; }
.history-edit { padding:16px; }.history-edit > div { display:flex; gap:8px; margin-top:10px; }.history-more { margin-top:14px; }.history-empty { color:var(--muted); margin-bottom:0; font-size:13px; }.history-notice { color:#2563eb; font-size:12px; }
@media(max-width:760px) { .history-list { grid-template-columns:1fr; }.history-search > div { flex-wrap:wrap; }.history-search input { flex-basis:100%; }.history-panel { padding:16px; }.history-heading { align-items:start; flex-direction:column; } }
</style>
