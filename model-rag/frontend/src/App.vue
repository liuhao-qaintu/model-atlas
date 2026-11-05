<script setup>
import { computed, onMounted, provide, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api } from './api'
import { account } from './auth.js'

const status = ref(null)
const error = ref('')
const route = useRoute()
const isAuthPage = computed(()=>['/login','/register'].includes(route.path))
async function logout(){
  try { await api('/auth/logout',{}) } finally { account.value=null; window.location.assign('/login') }
}
const pageName = computed(() => route.meta.title || '模型库')
const toolPages = ['/graph', '/manage', '/analysis', '/evaluation']
function inArea(path) { return route.path === path || route.path.startsWith(`${path}/`) }
const isToolPage = computed(() => toolPages.some(inArea) && !route.params.documentId)
const developerPages = ['/analysis', '/evaluation']
const developerMode = ref(false)
provide('developerMode', developerMode)
const toolsOpen = ref(isToolPage.value)
watch(() => route.path, () => {
  if (isToolPage.value) toolsOpen.value = true
  if (account.value?.is_maintainer && developerPages.some(inArea)) developerMode.value = true
})
watch(account,()=>{ if (!account.value?.is_maintainer) developerMode.value=false; refresh() })
async function refresh() {
  if (!account.value || isAuthPage.value) return
  try { status.value = await api('/health'); error.value = '' }
  catch { error.value = '后端暂未连接' }
}
onMounted(refresh)
watch(() => route.path, refresh)
provide('updateHealth', value => { status.value = value; error.value = '' })
</script>

<template>
  <RouterView v-if="isAuthPage" />
  <div v-else :class="['app-shell', { 'chat-shell': route.path === '/chat' }]">
    <aside class="sidebar">
      <RouterLink class="brand" to="/"><span class="brand-mark">M</span><span>Model Atlas<small>深度学习模型知识库</small></span></RouterLink>
      <div class="sidebar-label">学习与查证</div>
      <nav aria-label="主要导航">
        <RouterLink to="/" exact-active-class="active-home"><span>⌂</span>首页</RouterLink>
        <RouterLink to="/models" :class="{ 'router-link-active': inArea('/models') }" :aria-current="inArea('/models') ? 'location' : undefined"><span>▦</span>学模型</RouterLink>
        <RouterLink to="/documents" :class="{ 'router-link-active': inArea('/documents') }" :aria-current="inArea('/documents') ? 'location' : undefined"><span>▤</span>我的文献</RouterLink>
        <RouterLink to="/research"><span>⌕</span>比较研究</RouterLink>
      </nav>
      <nav aria-label="学习助手" class="assistant-nav">
        <RouterLink to="/chat"><span>◌</span>智能问答</RouterLink>
      </nav>
      <details class="workspace-tools" :open="toolsOpen" @toggle="toolsOpen = $event.target.open">
        <summary :class="{ 'tool-page-active': isToolPage }">资料与工具</summary>
        <nav aria-label="资料与工具">
          <RouterLink to="/graph">模型关系与依据</RouterLink>
          <RouterLink to="/manage">模型与关系管理</RouterLink>
          <RouterLink v-if="developerMode" to="/analysis">资料分析与审核</RouterLink>
          <RouterLink v-if="developerMode" to="/evaluation">RAG 质量评测</RouterLink>
          <label v-if="account?.is_maintainer" class="developer-toggle"><input v-model="developerMode" type="checkbox" />开发者视图</label>
        </nav>
      </details>
      <button class="service-status" @click="refresh">
        <span :class="['status-dot', { offline: error }]" />
        {{ error || (status ? '服务正常' : '连接中…') }}
      </button>
    </aside>
    <main :class="['workspace', { 'chat-workspace': route.path === '/chat' }]"><header class="workspace-header"><nav aria-label="页面位置"><RouterLink to="/">首页</RouterLink><template v-if="route.path !== '/'"><span>/</span><strong>{{ pageName }}</strong></template></nav><RouterLink v-if="route.path !== '/chat' && route.path !== '/' && !route.params.modelId" to="/chat" class="button">提问</RouterLink><div class="account-actions"><RouterLink to="/account">{{ account?.display_name || account?.username }}</RouterLink><button class="text-button" @click="logout">退出</button></div></header><RouterView /></main>
  </div>
</template>

<style scoped>
.account-actions{display:flex;align-items:center;gap:12px;margin-left:auto;font-size:13px}
.account-actions button{font:inherit;cursor:pointer;padding:6px 10px}

.assistant-nav{border-top:1px solid var(--line);padding-top:10px;margin-top:12px}
.developer-toggle { display:flex; align-items:center; gap:8px; margin:14px 12px 0; font-size:12px; color:var(--muted); cursor:pointer; }
.developer-toggle input { accent-color:var(--green); }
</style>
