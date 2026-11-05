<script setup>
import { computed, defineAsyncComponent, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api'
import MarkdownContent from '../components/MarkdownContent.vue'
import RelationPaths from '../components/RelationPaths.vue'
import FigureGallery from '../components/FigureGallery.vue'
import { rememberModel } from '../activityContext'
import { documentGroupTitle } from '../sourcePresentation'
import { readingSections, supplementarySections, readingNeighbors } from '../readingPresentation'
import LearningTrail from '../components/LearningTrail.vue'
import { aeReadingContent } from '../aeJourney'

const route = useRoute(), router = useRouter(), readingCard = ref(null)
const ReadingChat = defineAsyncComponent(() => import('./ChatView.vue'))
const readerOpen = ref(false), readerStarted = ref(false), readerBusy = ref(false), readerContext = ref(null)
const assistantPanel = ref(null), readerChat = ref(null)
let assistantTrigger = null, contextRevision = 0
const models = ref([]), detail = ref(null), error = ref(''), loading = ref(false)
const filter = ref('全部'), query = ref(typeof route.query.search === 'string' ? route.query.search : '')
const categories = computed(() => ['全部', ...new Set(models.value.map(m => m.category))])
const filtered = computed(() => models.value.filter(m => (filter.value === '全部' || m.category === filter.value) && `${m.name} ${m.tags.join(' ')} ${m.description}`.toLowerCase().includes(query.value.trim().toLowerCase() === 'ae' ? 'autoencoder' : query.value.toLowerCase())))
const cardSections = computed(() => readingSections(detail.value))
const primarySections = computed(() => cardSections.value.filter(section => section.primary))
const deeperSections = computed(() => cardSections.value.filter(section => !section.primary))
const extraSections = computed(() => supplementarySections(detail.value))
const documentGroups = computed(() => {
  const groups = new Map()
  for (const section of detail.value?.sections || []) {
    if (!section.document_id) continue
    if (!groups.has(section.document_id)) groups.set(section.document_id, { id: section.document_id, title: documentGroupTitle(section) })
  }
  return [...groups.values()]
})
const current = computed(() => detail.value?.sections.find(s => s.slug === route.query.section) ||
  (route.query.section === 'figures' && detail.value?.figures?.length ? { slug: 'figures', title: '架构图与论文插图', content: '', source_ids: [] } : detail.value?.sections[0]))
const neighbors = computed(() => readingNeighbors(detail.value, current.value?.id))
const invalidSection = computed(() => route.query.section && detail.value &&
  !detail.value.sections.some(s => s.slug === route.query.section) && !(route.query.section === 'figures' && detail.value.figures?.length))
function chapterQuery(slug) { return { section: slug, ...(route.query.flow === 'ae' ? { flow: 'ae' } : {}) } }
function chapterHref(section) { return router.resolve({ path: `/models/${detail.value.id}`, query: chapterQuery(section.slug) }).href }
async function showAssistant(event, useCurrentChapter = false) {
  if (useCurrentChapter && readerBusy.value) return
  assistantTrigger = event.currentTarget
  if (!readerStarted.value || useCurrentChapter) {
    readerContext.value = { modelId: detail.value.id, sectionId: current.value?.id || '', revision: ++contextRevision }
  }
  readerStarted.value = true; readerOpen.value = true
  await nextTick()
  assistantPanel.value?.focus({ preventScroll: true })
  assistantPanel.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  readerChat.value?.showLatestAnswer()
}
function hideAssistant() {
  readerOpen.value = false
  if (assistantTrigger?.isConnected) assistantTrigger.focus({ preventScroll: true })
}
function selectChapter(event) { router.push({ path: `/models/${detail.value.id}`, query: chapterQuery(event.target.value) }) }
async function readChapter(event, section) {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return
  event.preventDefault()
  await router.push(chapterHref(section))
  await nextTick()
  readingCard.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

async function loadDetail() {
  detail.value = null; error.value = ''
  readerOpen.value = false; readerStarted.value = false; readerBusy.value = false; readerContext.value = null
  if (!route.params.modelId) return
  loading.value = true
  try { detail.value = await api(`/models/${encodeURIComponent(route.params.modelId)}`); rememberModel(detail.value.id) }
  catch (e) { error.value = e.message }
  finally { loading.value = false }
}
onMounted(async () => {
  try { models.value = (await api('/models')).filter(model => !['文档', '知识标签'].includes(model.kind)); await loadDetail() }
  catch (e) { error.value = e.message }
})
watch(() => route.params.modelId, loadDetail)
</script>

<template>
  <section v-if="!route.params.modelId" class="model-index">
    <header class="catalog-header"><div><span class="eyebrow">按机制认识模型</span><h1>模型库<span>原理 · 结构 · 实现</span></h1><p class="page-intro">从卷积、注意力到多模态，选择一个主题开始阅读。</p></div><div v-if="models.length" class="catalog-meta"><strong>{{ models.filter(model => model.searchable).length }}</strong><span>可检索主题</span></div></header>
    <div class="toolbar"><div class="filter-tabs"><button v-for="category in categories" :key="category" :class="{ active: filter === category }" @click="filter = category">{{ category }}</button></div><input v-model="query" class="search-input" placeholder="搜索模型、机制或任务" aria-label="搜索模型" /></div>
    <p v-if="error" class="notice error">{{ error }}</p>
    <p v-if="!error && models.length && !filtered.length" class="notice">没有匹配的模型，请调整分类或关键词。</p><p v-if="filtered.length" class="catalog-result" role="status">{{ filter === '全部' ? '全部主题' : filter }} <span>{{ filtered.length }} 项{{ query.trim() ? '匹配结果' : '内容' }}</span></p><div class="model-grid">
      <RouterLink v-for="model in filtered" :key="model.id" :to="`/models/${model.id}`" class="model-card" :data-category="model.category">
        <div class="card-top"><span class="model-category"><i aria-hidden="true" />{{ model.category }}</span><span v-if="!model.searchable" class="badge pending">待补充资料</span></div>
        <h2>{{ model.name }}</h2><p>{{ model.description }}</p>
        <small class="model-kind">{{ model.kind }}</small>
        <div class="tag-row"><span v-for="tag in model.tags" :key="tag">{{ tag }}</span></div>
        <div class="card-bottom"><span>{{ model.document_count ? `${model.document_count} 份参考资料` : model.ready ? '模型学习笔记' : '自定义主题' }}</span><span>阅读模型 →</span></div>
      </RouterLink>
    </div>
  </section>
  <section v-else class="model-reader" :data-category="detail?.category">
    <LearningTrail v-if="route.query.flow === 'ae'" stage="model" />
    <RouterLink to="/models" class="back-link">← 返回模型库</RouterLink>
    <p v-if="loading" class="notice">正在读取模型资料…</p><p v-if="error" class="notice error">{{ error }}</p>
    <template v-if="detail">
      <header class="model-masthead"><div class="model-masthead-copy"><div class="page-top"><span class="eyebrow">{{ detail.category }} <span> / {{ detail.kind }}</span></span><span v-if="!detail.searchable" class="badge pending">待补充资料</span></div><h1>{{ detail.name }}</h1><p class="page-intro">{{ detail.description }}</p><RouterLink :to="{ path: '/graph', query: { node: detail.id } }" class="model-related-link">探索模型关系 <span aria-hidden="true">↗</span></RouterLink></div><div class="reading-meta"><span><b>{{ cardSections.length }}</b> 个阅读章节</span><RouterLink v-if="detail.figures?.length" :to="{ path: `/models/${detail.id}`, query: chapterQuery('figures') }"><b>{{ detail.figures.length }}</b> 幅论文图解 <span aria-hidden="true">↗</span></RouterLink></div></header>
      <div v-if="!detail.searchable && !detail.ready" class="empty-state"><h2>这张模型卡还在准备中</h2><p>可以先查看原始来源，或导入这个模型的资料。</p><RouterLink to="/documents" class="button primary">导入资料 →</RouterLink></div>
      <div v-else :class="['reader-workspace', { 'assistant-open': readerOpen }]">
      <div class="reader-layout">
        <nav class="section-nav" aria-label="模型章节">
          <span class="eyebrow">阅读目录</span>
          <RouterLink v-for="section in primarySections" :key="section.id" :class="{ selected: current?.id === section.id }" :aria-current="current?.id === section.id ? 'page' : undefined" :to="{ path: `/models/${detail.id}`, query: chapterQuery(section.slug) }">{{ section.label }}</RouterLink>
          <details v-if="deeperSections.length || detail.figures?.length" :open="deeperSections.some(s => s.id === current?.id) || current?.slug === 'figures'">
            <summary>推导与扩展</summary>
            <RouterLink v-for="section in deeperSections" :key="section.id" :class="{ selected: current?.id === section.id }" :aria-current="current?.id === section.id ? 'page' : undefined" :to="{ path: `/models/${detail.id}`, query: chapterQuery(section.slug) }">{{ section.label }}</RouterLink>
            <RouterLink v-if="detail.figures?.length && !cardSections.some(s => s.slug === 'figures')" :to="{ path:`/models/${detail.id}`, query:chapterQuery('figures') }" :class="{ selected:current?.slug === 'figures' }" :aria-current="current?.slug === 'figures' ? 'page' : undefined">论文图解</RouterLink>
          </details>
          <details v-if="documentGroups.length || extraSections.length" class="reader-references" :open="!!current?.document_id || extraSections.some(s => s.id === current?.id)">
            <summary>原始资料与补充</summary>
            <RouterLink v-for="document in documentGroups" :key="document.id" :to="`/documents/${document.id}`">{{ document.title }}<small>阅读原文 →</small></RouterLink>
            <RouterLink v-for="section in extraSections" :key="section.id" :class="{ selected: current?.id === section.id }" :to="{ path: `/models/${detail.id}`, query: chapterQuery(section.slug) }">{{ section.title }}</RouterLink>
          </details>
        </nav>
        <article v-if="current" ref="readingCard" class="reading-card"><nav class="reader-location" aria-label="阅读位置"><RouterLink to="/models">模型库</RouterLink><span>/</span><strong>{{ detail.name }}</strong><span>/</span><span>{{ current.title }}</span><span class="content-type">{{ current.document_id ? '资料原文' : '学习笔记' }}</span></nav>
          <p v-if="invalidSection" class="notice" role="status">请求的章节不存在，当前显示首个可用章节。请从目录选择或打开原始资料。</p>
          <div v-if="readerOpen" class="compact-chapters"><label for="reading-chapter">阅读章节</label><select id="reading-chapter" :value="current.slug" @change="selectChapter"><option v-for="section in [...cardSections, ...extraSections]" :key="section.id" :value="section.slug">{{ section.label || section.title }}</option></select></div>
          <div class="reading-heading"><h2>{{ current.title }}</h2><div class="reading-ask-actions"><button v-if="readerStarted && !readerOpen" class="button" aria-controls="reader-assistant" @click="showAssistant($event)">继续问答</button><button class="button" :disabled="readerBusy" :aria-expanded="readerOpen" aria-controls="reader-assistant" @click="showAssistant($event, true)">问这一节</button></div></div>
          <MarkdownContent v-if="current.slug !== 'figures' || !detail.figures?.length" :content="aeReadingContent(current.content, route.query.flow, detail.id)" />
          <template v-if="current.slug === 'overview' && detail.figures?.length"><div class="figure-section-heading"><h3>从论文图解理解模型</h3><RouterLink :to="{ path: `/models/${detail.id}`, query: chapterQuery('figures') }">全部图解 →</RouterLink></div><FigureGallery :figures="detail.figures.slice(0, 1)" /></template>
          <FigureGallery v-if="current.slug === 'figures'" :figures="detail.figures || []" />
          <MarkdownContent v-if="current.slug === 'figures' && detail.figures?.length && current.title === '架构图与论文插图' && current.content" class="figure-reading-guide" :content="current.content" />
          <details v-if="current.slug === 'relations' && detail.paths?.length" class="model-relation-evidence"><summary>查看关联模型与依据</summary><RelationPaths :paths="detail.paths" /></details>
          <div class="reading-footer">{{ current.document_id ? '这是导入的原文文字，解析结果尚未自动核验。' : '资料为整理后的模型卡；原始来源与版本可在“来源与定位”章节查看。' }}<RouterLink v-if="current.reader_url" :to="current.reader_url"> 查看原文位置 →</RouterLink></div>
          <nav v-if="neighbors.previous || neighbors.next" class="reading-pagination" aria-label="相邻章节">
            <a v-if="neighbors.previous" :href="chapterHref(neighbors.previous)" @click="readChapter($event, neighbors.previous)">← {{ neighbors.previous.label }}</a><span v-else />
            <a v-if="neighbors.next" :href="chapterHref(neighbors.next)" @click="readChapter($event, neighbors.next)">接着读：{{ neighbors.next.label }} →</a>
          </nav>
        </article>
      </div>
      <aside v-if="readerStarted" v-show="readerOpen" id="reader-assistant" ref="assistantPanel" class="reader-assistant" tabindex="-1" aria-labelledby="reader-assistant-title" @keydown.esc.stop="hideAssistant">
        <header class="reader-assistant-heading"><h2 id="reader-assistant-title">阅读问答</h2><button class="button" aria-label="收起阅读问答" @click="hideAssistant">收起</button></header>
        <ReadingChat ref="readerChat" :key="detail.id" embedded :reading-context="readerContext" @busy="readerBusy = $event" />
      </aside>
      </div>
    </template>
  </section>
</template>

<style scoped>
.model-index,.model-reader { --model-accent:#4c66b9; --model-tint:#f0f4fc; }
[data-category="卷积网络"] { --model-accent:#4874b9; --model-tint:#f0f5fc; }
[data-category="表示学习"],[data-category="注意力"] { --model-accent:#8260b3; --model-tint:#f6f2fc; }
[data-category="序列建模"],[data-category="状态空间模型"] { --model-accent:#ad7442; --model-tint:#fbf5ed; }
[data-category="视觉与多模态"] { --model-accent:#26827e; --model-tint:#edf8f5; }
[data-category="应用案例"] { --model-accent:#566bc5; --model-tint:#f1f3fd; }
.catalog-header { display:flex; justify-content:space-between; gap:28px; align-items:center; margin-bottom:26px; padding:12px 0 27px; border-bottom:1px solid var(--line); }
.catalog-header h1 { font-size:40px; margin:11px 0 12px; letter-spacing:-1px; }
.catalog-header h1>span { display:inline-block; vertical-align:middle; margin-left:20px; color:#8791a7; font-size:12px; letter-spacing:2px; font-weight:400; }
.catalog-header .page-intro { margin:0; font-size:13px; }
.catalog-meta { display:grid; flex:none; gap:3px; text-align:right; padding-left:26px; border-left:1px solid #dfe5f0; }
.catalog-meta strong { font-family:var(--font-display); font-size:42px; line-height:1.15; letter-spacing:-2px; color:#6075ac; font-weight:500; }
.catalog-meta>span { font-size:10px; color:#68758d; letter-spacing:1px; }
.model-index .toolbar { gap:14px; padding:15px 18px; border:1px solid var(--line); border-radius:12px; background:#fff; margin-bottom:18px; }
.model-index .filter-tabs { gap:4px; }
.model-index .filter-tabs button { color:#6d7890; padding:7px 10px; }
.model-index .filter-tabs button.active { background:#edf1fc; box-shadow:none; color:#405bc0; }
.model-index .search-input { background:#f7f9fc; border-color:#e9edf4; min-width:180px; font-size:12px; }
.catalog-result { font-size:12px; color:#58667e; margin:0 0 15px; font-weight:550; }
.catalog-result>span { margin-left:10px; color:#6d7b94; font-size:10px; font-weight:400; }
.model-index .model-card { position:relative; padding:24px; overflow:hidden; min-height:280px; background:linear-gradient(155deg,#fff 65%,var(--model-tint)); }
.model-index .model-card::before { content:""; position:absolute; left:24px; top:0; height:3px; width:34px; border-radius:0 0 3px 3px; background:var(--model-accent); }
.model-index .model-card:hover { border-color:var(--model-accent); }
.model-category { display:flex; gap:7px; align-items:center; font-size:10px; color:var(--model-accent); letter-spacing:.5px; }
.model-category i { width:5px; height:5px; border-radius:50%; background:currentColor; }
.model-index .card-top { margin-bottom:15px; }
.model-index .model-card h2 { font-family:var(--font-display); font-size:25px; font-weight:650; letter-spacing:-.6px; line-height:1.2; margin:0 0 13px; overflow-wrap:anywhere; }
.model-index .model-card p { font-size:13px; line-height:1.9; margin-bottom:15px; }
.model-kind { color:#68778f; font-size:10px; margin-bottom:12px; }
.model-index .tag-row { margin-bottom:18px; gap:6px; }
.model-index .tag-row span { font-size:10px; color:#6c7a91; background:#f2f4f8; padding:2px 7px; }
.model-index .card-bottom { font-size:10px; border-color:#e8ecf3; padding-top:13px; gap:10px; }
.model-index .card-bottom span:last-child { color:var(--model-accent); flex:none; }
.model-masthead { display:flex; gap:24px; justify-content:space-between; align-items:center; margin-bottom:30px; padding:20px 28px; border-left:3px solid var(--model-accent); border-radius:0 14px 14px 0; background:linear-gradient(110deg,var(--model-tint),#f8f9fc); }
.model-masthead-copy { min-width:0; }
.model-masthead .eyebrow { color:var(--model-accent); font-size:11px; }
.model-masthead .eyebrow>span { color:#6b7891; margin-left:8px; }
.model-masthead .page-top { min-height:15px; }
.model-masthead h1 { font-family:var(--font-display); font-size:40px; letter-spacing:-1px; margin:9px 0; overflow-wrap:anywhere; }
.model-masthead .page-intro { font-size:13px; margin:0 0 10px; max-width:640px; }
.model-related-link { font-size:11px; color:var(--model-accent); display:inline-flex; gap:16px; }
.reading-meta { display:grid; gap:12px; flex:none; padding-left:24px; border-left:1px solid #dfe5ee; color:#6e7b92; font-size:11px; }
.reading-meta b { font-size:20px; font-family:var(--font-display); color:#68799c; font-weight:500; margin-right:3px; }
.reading-meta>a:hover { color:var(--model-accent); }
.model-reader>.back-link { margin-bottom:16px; font-size:11px; }
.reading-card { border-radius:14px; border-color:#e1e6ee; box-shadow:0 3px 18px #243c6510; }
.reading-card .reader-location { font-size:11px; border-bottom:0; padding-bottom:0; margin-bottom:24px; color:#6d7b92; }
.reading-card .content-type { margin-left:auto; font-size:10px; border-radius:5px; background:var(--model-tint); color:var(--model-accent); }
.reading-heading { padding-bottom:22px; margin-bottom:26px; }
.reading-heading h2 { font-size:25px; font-weight:700; letter-spacing:-.5px; }
.reading-card :deep(.markdown) { font-size:15px; line-height:2.05; color:#465269; }
.reading-card :deep(.markdown h2) { font-size:23px; color:#243553; margin:34px 0 16px; letter-spacing:-.3px; }
.reading-card :deep(.markdown h3) { font-size:18px; color:#314969; margin:30px 0 14px; }
.reading-card :deep(.markdown h4) { font-size:15px; color:#36506f; margin:24px 0 12px; }
.reading-card :deep(.markdown>p:first-child) { font-size:16px; color:#314666; line-height:1.95; }
.reading-card :deep(.markdown pre) { background:#f5f7fb; border:1px solid #e2e8f1; border-left:3px solid #bdcce3; padding:20px; font-size:12px; line-height:1.85; border-radius:8px; }
.reading-card :deep(.markdown code) { font-family:var(--font-code); }
.reading-card :deep(.markdown table) { width:100%; font-size:12px; line-height:1.85; }
.reading-card :deep(.markdown th) { background:#edf2f8; color:#39516e; }
.reading-card :deep(.markdown tr:nth-child(even) td) { background:#fafbfd; }
.reading-card :deep(.markdown blockquote) { background:var(--model-tint); border-left-color:var(--model-accent); padding:16px 20px; color:#536582; }
.reading-card :deep(.markdown strong) { color:#2b3e5c; font-weight:650; }
.section-nav { border:0; background:transparent; padding:10px 8px 10px 0; }
.section-nav>.eyebrow { color:#8e98ad; font-size:10px; letter-spacing:1.5px; }
.section-nav a.selected { color:var(--model-accent); background:var(--model-tint); }
.section-nav a[aria-current=page] { border-left-color:var(--model-accent); }
.reader-workspace { min-width:0; }
.reader-workspace.assistant-open { display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,.95fr); gap:20px; align-items:start; }
.assistant-open .reader-layout { grid-template-columns:minmax(0,1fr); }
.assistant-open .section-nav { display:none; }
.reader-assistant { position:sticky; top:16px; scroll-margin-top:16px; height:calc(100dvh - 32px); max-height:900px; min-height:400px; min-width:0; display:flex; flex-direction:column; padding:16px; border:1px solid #dce3f1; border-radius:12px; background:#fff; box-shadow:0 5px 24px #233c6b0a; }
.reader-assistant-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }
.reader-assistant-heading h2 { font-size:17px; margin:0; }
.reader-assistant-heading .button { padding:5px 10px; font-size:12px; }
.reading-ask-actions { display:flex; flex-wrap:wrap; gap:7px; }
.compact-chapters { display:flex; gap:10px; align-items:center; margin-bottom:16px; font-size:12px; color:var(--muted); }
.compact-chapters select { min-width:0; flex:1; background:#fff; border:1px solid var(--line); border-radius:6px; padding:7px; color:#394763; font:inherit; }
@media(max-width:1100px) {
  .reader-workspace.assistant-open { display:block; }
  .reader-assistant { position:fixed; right:16px; bottom:16px; top:auto; z-index:45; width:min(520px,calc(100vw - 32px)); height:min(76dvh,800px); min-height:0; box-sizing:border-box; box-shadow:0 10px 45px #233c6b30; }
}
@media(max-width:600px) {
  .reader-assistant { right:0; bottom:0; width:100%; height:80dvh; border-radius:14px 14px 0 0; padding:12px; }
  .reading-ask-actions .button { padding:6px 9px; font-size:12px; }
}
.section-nav { min-width:0; }
.section-nav a { display:block; overflow-wrap:anywhere; }
.section-nav a small { display:block; margin-top:3px; }
.section-nav details { border-top:1px solid var(--line); padding:12px 0; font-size:12px; }
.section-nav summary { padding:0 10px; cursor:pointer; color:#475569; line-height:1.8; overflow-wrap:anywhere; }
.section-nav details a { margin-top:5px; }
.model-relation-evidence { margin-top:20px; border-top:1px solid var(--line); padding-top:16px; font-size:13px; }
.model-relation-evidence>summary { color:#52688e; cursor:pointer; }
.figure-reading-guide { margin-top:24px; border-top:1px solid var(--line); padding-top:8px; }
@media(max-width:1100px) { .model-masthead h1 { font-size:36px; }.reading-meta { padding-left:15px; }.model-masthead { padding:22px; }.catalog-header h1>span { display:block; margin:12px 0 0; }.model-index .toolbar { align-items:stretch; } }
@media(max-width:760px) { .catalog-header { gap:15px; padding-top:5px; margin-bottom:20px; }.catalog-header h1 { font-size:34px; }.catalog-meta { padding-left:15px; }.catalog-meta strong { font-size:34px; }.model-index .toolbar { padding:13px; }.model-index .search-input { width:100%; }.model-masthead { display:block; padding:20px; margin-bottom:20px; }.model-masthead h1 { font-size:35px; }.reading-meta { display:flex; flex-wrap:wrap; gap:20px; padding:15px 0 0; margin-top:18px; border-left:0; border-top:1px solid #dfe5ee; }.reading-meta b { font-size:17px; }.reading-heading h2 { font-size:22px; }.reading-card :deep(.markdown) { font-size:14px; }.reading-card :deep(.markdown>p:first-child) { font-size:15px; }.reading-card :deep(.markdown h2) { font-size:21px; }.reading-card :deep(.markdown pre) { padding:15px; }.reading-card .reader-location { margin-bottom:18px; }.section-nav { padding:0; }.reading-card .content-type { margin-left:0; } }
</style>
