<script setup>
import { computed, inject, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api'
import SourceLinks from '../components/SourceLinks.vue'
import { comparisonPair, comparisonRoute } from '../graphComparison'
import { rememberModel } from '../activityContext'

const route = useRoute(), router = useRouter()
const developerMode = inject('developerMode', ref(false))
const graph = ref(null), loading = ref(false), error = ref('')
const pairs = ref({})
const edgePanel = ref(null)
let revision = 0
const root = computed(() => typeof route.query.node === 'string' ? route.query.node : 'resnet-18')
const hops = computed(() => route.query.hops === '1' ? 1 : 2)
const kind = computed(() => typeof route.query.kind === 'string' ? route.query.kind : '')
const current = computed(() => graph.value?.edges.find(edge => edge.id === route.query.edge) || graph.value?.edges[0])
const rootName = computed(() => graph.value?.catalog.find(node => node.id === root.value)?.name || '全部关系')
const catalog = computed(() => [...(graph.value?.catalog || [])].sort((a, b) => Number(Boolean(b.model_id)) - Number(Boolean(a.model_id)) || a.name.localeCompare(b.name)))
const nodeNames = computed(() => Object.fromEntries((graph.value?.catalog || []).map(node => [node.id, node.name])))
const missingEdge = computed(() => route.query.edge && graph.value && !graph.value.edges.some(edge => edge.id === route.query.edge))

async function load() {
  const turn = ++revision
  loading.value = true; error.value = ''
  const params = new URLSearchParams({ hops: String(hops.value) })
  if (root.value !== 'all') params.set('root', root.value)
  if (kind.value) params.set('kind', kind.value)
  try {
    const result = await api(`/graph?${params}`)
    if (turn === revision) {
      graph.value = result
      if (result.catalog.some(node => node.id === root.value && node.model_id)) rememberModel(root.value, 'graph')
      pairs.value = Object.fromEntries(result.shared.map(component => [component.node_id, comparisonPair(component)]))
    }
  } catch (e) { if (turn === revision) error.value = e.message }
  finally { if (turn === revision) loading.value = false }
}
function focus(node, depth = hops.value, relationKind = kind.value) {
  router.push({ path: '/graph', query: { node, hops: String(depth), ...(relationKind ? { kind: relationKind } : {}) } })
}
async function chooseEdge(edge) {
  if (!edge) return
  await router.replace({ path: '/graph', query: { ...route.query, edge: edge.id } })
  await nextTick()
  if (edgePanel.value) edgePanel.value.scrollTop = 0
}
function ask(edge) {
  const subject = edge.owner_name === edge.source_name ? edge.source_name : `${edge.owner_name} 中的 ${edge.source_name}`
  return { path: '/chat', query: { model: edge.owner_model_id, section: edge.evidence[0].section_id,
    question: `${subject} ${edge.relation} ${edge.target_name}，具体如何实现？请解释输入输出和这条关系的依据。` } }
}
function compare(component) {
  return comparisonRoute(component, pairs.value[component.node_id] || [])
}
watch(() => [root.value, hops.value, kind.value], load, { immediate: true })
onBeforeUnmount(() => { revision++ })
</script>

<template>
  <section class="graph-page">
    <span class="eyebrow">模型、模块与结构的联系</span>
    <h1>模型关系</h1>
    <p class="page-intro">选择模型，了解它与哪些结构有关。点击一条联系，可查看说明或继续提问。</p>
    <div class="graph-controls">
      <label class="graph-start">从哪个模型开始？<select :value="root" @change="focus($event.target.value)"><option value="all">全部关系</option><option v-for="node in catalog" :key="node.id" :value="node.id">{{ node.name }}</option></select></label>
      <details class="graph-filters"><summary>筛选 <small>{{ hops === 1 ? '直接相关' : '更多关联' }}{{ kind ? ` · ${kind}` : '' }}</small></summary><div>
        <label>关联范围<select :value="hops" @change="focus(root, Number($event.target.value))"><option :value="1">直接相关</option><option :value="2">更多关联（包含相关模型的联系）</option></select></label>
        <label>关系类型<select :value="kind" @change="focus(root, hops, $event.target.value)"><option value="">全部类型</option><option v-for="type in graph?.kinds || []" :key="type">{{ type }}</option></select></label>
      </div></details>
    </div>
    <p v-if="loading" class="notice" role="status">正在读取关系与依据…</p>
    <p v-if="error" class="notice error" role="alert">{{ error }} <button class="button" @click="focus('all', 2, '')">查看全部关系</button></p>
    <template v-if="graph">
      <div class="graph-summary"><span><b>{{ rootName }}</b><small>{{ graph.edges.length }} 条联系</small></span></div>
      <details v-if="developerMode" class="graph-node-list"><summary>图谱节点与整理说明</summary><p class="graph-explanation">{{ graph.curation }} 箭头表示关系原始方向，邻接探索可沿反方向寻找联系。已整理 {{ graph.stats.edges }} 条有摘录的关系{{ graph.stats.excluded_edges ? `，${graph.stats.excluded_edges} 条缺少完整依据暂不展示` : '' }}。</p><div class="node-cloud" aria-label="当前关系节点">
        <button v-for="node in graph.nodes" :key="node.id" :class="['node-chip', { focused: node.id === graph.root }]" :aria-pressed="node.id === graph.root" @click="focus(node.id)"><strong>{{ node.name }}</strong><small>{{ node.kind }}{{ !node.ready ? ' · 待整理' : '' }}{{ node.distance !== null ? ` · ${node.distance} 跳` : '' }}</small></button>
      </div></details>
      <p v-if="missingEdge" class="notice">链接中的关系不在当前范围内；请调整起点或类型后查看。</p>
      <div v-if="graph.edges.length" class="graph-layout">
        <div class="relation-browser">
          <h2 class="graph-heading">模型关系 <small>选择一条查看依据</small></h2>
          <div class="graph-edges" aria-label="模型关系列表">
            <button v-for="edge in graph.edges" :key="edge.id" :class="['graph-edge', { selected: current?.id === edge.id }]" :aria-pressed="current?.id === edge.id" @click="chooseEdge(edge)">
              <span class="edge-end"><strong>{{ edge.source_name }}</strong></span>
              <span class="edge-arrow"><small>{{ edge.relation }}</small><span aria-hidden="true">→</span></span>
              <span class="edge-end"><strong>{{ edge.target_name }}</strong></span>
            </button>
          </div>
          <details v-if="developerMode && graph.paths.length" class="graph-paths"><summary>查看从起点出发的路径（{{ graph.paths.length }} 条）</summary><p>每个节点展示一条最短邻接路径；“逆向”仅表示浏览方向。</p><div v-for="path in graph.paths" :key="path.target"><b>{{ path.hops }} 跳 · {{ nodeNames[path.target] }}</b><ol><li v-for="step in path.steps" :key="step.edge_id"><button @click="chooseEdge(graph.edges.find(e => e.id === step.edge_id))">{{ nodeNames[step.from] }} → {{ nodeNames[step.to] }} · {{ step.reverse ? '逆向邻接' : '顺向邻接' }} · 查看依据</button></li></ol></div></details>
        </div>
        <aside v-if="current" ref="edgePanel" class="edge-detail" aria-label="关系依据">
          <span class="eyebrow">{{ current.owner_name }} · 关系依据</span><h2>{{ current.source_name }}<span class="detail-verb">{{ current.relation }} →</span>{{ current.target_name }}</h2>
          <div class="detail-actions"><RouterLink :to="ask(current)" class="button primary">问问这条联系 ↗</RouterLink><button class="button" :title="`以 ${current.target_name} 为起点`" :aria-label="`以 ${current.target_name} 为起点`" @click="focus(current.target)">查看 {{ current.target_name }} 的关联 →</button></div>
          <p v-if="current.implementation" class="implementation">{{ current.implementation }}</p>
          <details :key="`${current.id}-proof`" class="proof-details"><summary>查看资料依据</summary><div v-for="proof in current.evidence" :key="proof.section_id" class="proof"><span class="eyebrow">{{ current.owner_name }} · {{ proof.title }}</span><blockquote>{{ proof.quote }}</blockquote><RouterLink :to="proof.reader_url" class="back-link">阅读这段内容 →</RouterLink></div></details>
          <details :key="current.id" class="relation-context"><summary>来源与适用范围</summary><p class="detail-locator">{{ current.locator }}</p><SourceLinks :sources="current.sources" /><p class="detail-scope">{{ current.scope }} {{ current.origin === 'user' ? '用户创建的证据关系。' : current.origin === 'llm' ? '自动提取后已确认，原文可复核。' : '摘录由人工整理，可通过来源核查。' }}</p></details>
        </aside>
      </div>
      <div v-else class="empty-state"><h2>这个范围还没有已整理的关系</h2><p>可先选择其他已整理主题，或调整起点与类型；更多资料可逐步补充。</p><RouterLink v-if="graph.nodes[0]?.reader_url" :to="graph.nodes[0].reader_url" class="button">查看模型资料 →</RouterLink><button class="button" @click="focus('all', 2, '')">查看全部关系</button></div>
      <details v-if="graph.shared.length" class="shared-section">
        <summary>比较共享结构 <small>{{ graph.shared.length }} 类结构</small></summary>
        <p>选择两个实现，比较代码与输入输出。此处包含模型库中的其他已知实现。</p>
        <div class="shared-grid"><article v-for="component in graph.shared" :key="component.node_id" class="shared-card">
          <h3>{{ component.name }}</h3>
          <div class="comparison-selectors"><label v-for="(label, index) in ['对象 A', '对象 B']" :key="label">{{ label }}<select v-model="pairs[component.node_id][index]" :aria-label="`${component.name} ${label}`"><option v-for="item in component.implementations" :key="item.edge_id" :value="item.edge_id">{{ item.model_name }} · {{ item.entity }}</option></select></label></div>
          <RouterLink v-if="compare(component)" :to="compare(component)" class="button">比较选中实现 →</RouterLink><span v-else class="notice compact">请选择两个不同的实现。</span>
          <details class="implementation-notes"><summary>查看实现说明</summary><div v-for="implementation in component.implementations" :key="implementation.edge_id"><b>{{ implementation.entity }}</b><p>{{ implementation.text }}</p><RouterLink :to="implementation.reader_url" class="back-link">查看实现依据 →</RouterLink></div></details>
        </article></div>
        <p class="graph-explanation">共享模块不代表权重、任务或整体架构相同；教学实例的组成不能推广到同一家族的所有模型。</p>
      </details>
    </template>
  </section>
</template>

<style scoped>
.graph-page { --green:#23827f; }
.graph-page>h1 { font-size:38px; letter-spacing:-1px; margin:12px 0; }
.graph-page>.eyebrow { color:#4c9290; font-size:11px; letter-spacing:1px; }
.graph-page>.page-intro { font-size:13px; max-width:650px; margin-bottom:26px; }
.graph-controls { display: flex; flex-wrap: wrap; gap: 22px; padding: 21px 25px; border: 1px solid #d9e7e6; border-left:3px solid #64a8a0; border-radius: 0 14px 14px 0; background: linear-gradient(100deg,#f0f8f6,#fbfdfc); }
.graph-controls label { display: grid; gap: 7px; color: var(--muted); font-size: 12px; }
.graph-controls select { padding: 9px 12px; max-width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; color: #1e293b; }
.graph-start { min-width:min(300px,100%); flex:1; }
.graph-filters { flex:1; min-width:0; padding-top:26px; font-size:12px; }
.graph-filters>summary { cursor:pointer; color:#52688e; }
.graph-filters>summary small { color:var(--muted); margin-left:9px; font-size:11px; }
.graph-filters>div { display:grid; gap:12px; padding-top:14px; }
.proof-details { margin:20px 0 15px; font-size:12px; }
.proof-details>summary { color:#52688e; cursor:pointer; }
.graph-summary { margin:25px 0 10px; }
.graph-summary b { font-family:var(--font-display); font-size:20px; font-weight:600; letter-spacing:-.4px; color:#335d65; }
.graph-summary small { padding:3px 8px; border-radius:5px; background:#eaf3f0; }
.graph-summary>span { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.graph-summary small, .graph-explanation, .graph-heading small { color: var(--muted); font-size: 12px; }
.graph-explanation { line-height: 1.8; }
.node-cloud { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 26px; }
.node-chip { display: grid; gap: 2px; background: #ffffff; padding: 10px 14px; border: 1px solid transparent; border-radius: 9px; text-align: left; color: #334155; }
.node-chip small { color: var(--muted); font-size: 12px; }
.node-chip.focused { background: #eaf1ff; border-color: #2563eb; }
.graph-layout { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(300px, 1fr); gap: 18px; align-items: start; }
.relation-browser { padding:21px; background:white; border:1px solid var(--line); border-radius:14px; min-width:0; }
.graph-heading { font-size: 15px; margin:0 0 14px; display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.graph-edges { display: grid; gap: 8px; max-height: min(560px, 62vh); overflow:auto; padding:2px 6px 2px 2px; scrollbar-gutter:stable; }
.graph-edge { display: grid; grid-template-columns: minmax(0, 1fr) 78px minmax(0, 1fr); align-items: center; gap: 8px; width: 100%; padding: 17px 14px; border: 1px solid #e4ebed; border-radius: 10px; text-align: left; background: linear-gradient(110deg,#f9fbfd,#fff); color: #1e293b; }
.graph-edge:hover { border-color:#9cc6c0; }
.edge-end strong { font-size:13px; font-weight:550; }
.graph-edge.selected { border-color: #76b4a8; background: #eef7f3; box-shadow:inset 3px 0 #4a9e91; }
.edge-end { display: grid; gap: 5px; overflow-wrap: anywhere; }
.edge-end small { font-size: 10px; color: var(--muted); }
.edge-arrow { display: grid; text-align: center; color: var(--green); }
.edge-arrow small { font-size: 11px; }
.edge-arrow > span { font-size: 23px; line-height: 1; }
.edge-detail { padding: 26px; border: 1px solid #d9e7e3; border-top:3px solid #77b6aa; border-radius: 14px; background: linear-gradient(155deg,#fff 55%,#f3f9f6); position: sticky; top: 24px; overflow-wrap: anywhere; max-height:calc(100vh - 90px); overflow:auto; scrollbar-gutter:stable; box-shadow:0 4px 20px #335e5d06; }
.edge-detail>.eyebrow { color:#517b74; font-size:10px; }
.edge-detail h2 { font-family:var(--font-display); font-size: 25px; line-height:1.45; letter-spacing:-.5px; margin: 16px 0 22px; color:#284f58; }
.detail-verb { display: block; font-size: 12px; color: var(--green); margin: 6px 0; font-weight: 400; }
.detail-locator { font-size: 12px; color: var(--muted); }
.implementation { padding: 16px; background: #eaf4f0; color:#476a64; border-radius: 8px; font-size: 13px; line-height:1.95; }
.proof { margin: 23px 0; }
.proof blockquote { margin: 10px 0 14px; padding: 12px 14px; border-left: 3px solid #b4c4ef; background: #f8faff; white-space: pre-wrap; font-family: inherit; font-size: 13px; }
.edge-detail h3 { font-size: 14px; margin-bottom: 8px; }
.detail-scope { margin-top: 18px; font-size: 12px; color: var(--muted); line-height: 1.8; }
.detail-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 18px; }
.detail-actions .button { font-size:12px; padding:8px 11px; }
.relation-context { border-top:1px solid var(--line); padding-top:12px; font-size:12px; }
.relation-context>summary,.implementation-notes>summary { color:#52688e; cursor:pointer; }
.graph-paths { margin-top: 20px; padding: 16px; background: #ffffff; border-radius: 10px; font-size: 12px; }
.graph-paths summary { cursor: pointer; }
.graph-paths p { color: var(--muted); margin-top: 12px; }
.graph-paths button { text-align: left; background: transparent; color: var(--green); padding: 4px 0; }
.shared-section { margin-top:24px; border:1px solid var(--line); border-radius:12px; padding:18px 20px; background:white; }
.shared-section>summary { cursor:pointer; font-size:15px; font-weight:550; }
.shared-section>summary small { color:var(--muted); font-size:11px; font-weight:400; margin-left:10px; }
.shared-section > p { color: var(--muted); }
.shared-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.shared-card { padding: 23px; border: 1px solid var(--line); border-radius: 13px; background: var(--paper); }
.shared-card h3 { margin-top: 8px; }
.shared-card > div { padding: 15px 0; border-top: 1px solid var(--line); }
.shared-card p { font-size: 13px; margin: 6px 0; }
.shared-card > .button { margin-top: 12px; }
.implementation-notes { margin-top:16px; border-top:1px solid var(--line); padding-top:12px; font-size:12px; }
.implementation-notes>div { padding:13px 0; border-bottom:1px solid #edf0f5; }
@media (max-width: 980px) { .graph-layout { grid-template-columns: 1fr; } .edge-detail { position: static; max-height:none; } }
@media (max-width: 650px) { .graph-controls { display: grid; } .graph-filters { padding-top:0; } .graph-edge { grid-template-columns: minmax(0, 1fr) 80px minmax(0, 1fr); padding: 14px 10px; gap: 6px; } .shared-grid { grid-template-columns: 1fr; } .edge-detail { padding: 20px; } }
</style>
