<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { account } from '../auth.js'
const options = ref({ models: [], relations: [], verbs: [] }), error = ref(''), success = ref(''), busy = ref(false), editing = ref('')
const form = ref({ name: '', category: '', description: '', aliases: '', tags: '' })
const relation = ref({ source: '', target: '', relation: '采用架构', section_id: '', quote: '' }), sections = ref([])
const selected = computed(() => sections.value.find(s => s.id === relation.value.section_id))
async function load() { options.value = await api('/management') }
function edit(model) { editing.value = model.id; form.value = { ...model, aliases: model.aliases.join(', '), tags: model.tags.join(', ') } }
function reset() { editing.value = ''; form.value = { name: '', category: '', description: '', aliases: '', tags: '' } }
async function save() {
  busy.value = true; error.value = ''; success.value = ''
  try {
    const body = { ...form.value, category: form.value.category || '自定义模型', aliases: form.value.aliases.split(/[,，]/).filter(Boolean), tags: form.value.tags.split(/[,，]/).filter(Boolean) }
    const result = await api(editing.value ? `/models/${editing.value}` : '/models', body, { method: editing.value ? 'PATCH' : 'POST' })
    await load(); reset(); success.value = `${result.name} 已保存，可在文档管理中关联资料。`
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function saveRelation() {
  busy.value = true; error.value = ''; success.value = ''
  try { await api('/relations', relation.value); await load(); success.value = '关系已加入图谱与后续检索。'; relation.value.quote = '' }
  catch (e) { error.value = e.message } finally { busy.value = false }
}
async function toggleRelation(edge) {
  busy.value = true; error.value = ''
  try { await api(`/relations/${edge.id}`, { status: edge.status === 'accepted' ? 'rejected' : 'accepted' }, { method:'PATCH' }); await load() }
  catch (e) { error.value = e.message } finally { busy.value = false }
}
let revision = 0
watch(() => relation.value.source, async id => {
  const turn = ++revision; sections.value = []; relation.value.section_id = ''; relation.value.quote = ''
  if (!id) return
  try { const detail = await api(`/models/${id}`); if (turn === revision) sections.value = detail.sections } catch (e) { if (turn === revision) error.value = e.message }
})
onMounted(async () => { try { await load() } catch (e) { error.value = e.message } })
</script>

<template>
  <section><div class="page-top"><span class="eyebrow">构建自己的模型知识</span><RouterLink to="/documents" class="button">导入模型资料 →</RouterLink></div><h1>模型管理</h1><p class="page-intro">创建模型主题，关联资料，再用原文证据建立模型之间的联系。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="success" class="notice" role="status">{{ success }}</p>
    <div class="management-grid"><form class="surface-form" @submit.prevent="save"><h2>{{ editing ? '编辑模型' : '新增模型' }}</h2><label>模型名称<input v-model="form.name" maxlength="80" required placeholder="例如自己的模型名称或一个新版本" /></label><label>类别<input v-model="form.category" maxlength="40" placeholder="例如视觉与多模态" /></label><label>模型说明<textarea v-model="form.description" maxlength="500" rows="3" placeholder="说明模型用途与适用任务" /></label><label>别名<input v-model="form.aliases" placeholder="多个别名用逗号分隔" /></label><label>标签<input v-model="form.tags" placeholder="例如视觉、分类" /></label><div class="form-actions"><button type="button" v-if="editing" class="button" @click="reset">取消编辑</button><button class="button primary" :disabled="busy">保存模型</button></div></form>
      <div class="surface-form"><h2>模型目录 <small>{{ options.models.length }}</small></h2><div class="managed-models"><article v-for="model in options.models" :key="model.id"><div><RouterLink :to="`/models/${model.id}`"><strong>{{ model.name }}</strong></RouterLink><small>{{ model.category }} · {{ model.origin === 'user' ? '自定义' : '预置' }} · {{ model.document_count || 0 }} 份资料</small></div><button v-if="model.origin === 'user'" class="button" @click="edit(model)">编辑</button></article></div></div></div>
    <form class="surface-form relation-create" @submit.prevent="saveRelation"><div class="page-top"><h2>建立模型关系</h2><RouterLink v-if="account?.is_maintainer" to="/analysis" class="back-link">从资料自动提出候选 →</RouterLink><RouterLink v-else to="/documents" class="back-link">从文献分析报告建立关系 →</RouterLink></div><p class="muted">先选择所属模型及证据章节，摘录必须包含目标模型名称；保存后可在图谱中查证和提问。</p><div class="document-form-grid"><label>所属模型<select v-model="relation.source" required><option value="">选择模型</option><option v-for="model in options.models" :key="model.id" :value="model.id">{{ model.name }}</option></select></label><label>关联模型<select v-model="relation.target" required><option value="">选择目标</option><option v-for="model in options.models.filter(m => m.id !== relation.source)" :key="model.id" :value="model.id">{{ model.name }}</option></select></label><label>关系类型<select v-model="relation.relation"><option v-for="verb in options.verbs" :key="verb">{{ verb }}</option></select></label><label>证据章节<select v-model="relation.section_id" required><option value="">选择正文位置</option><option v-for="s in sections.filter(s => s.source_ids.length)" :key="s.id" :value="s.id">{{ s.title }}{{ s.document_id ? ' · 原文' : ' · 模型卡' }}</option></select></label></div><details v-if="selected" class="evidence-panel" open><summary>所选正文</summary><pre class="document-text">{{ selected.content }}</pre></details><label>原样摘录<textarea v-model="relation.quote" required minlength="15" maxlength="700" rows="4" placeholder="从上面的正文复制支持该关系的一段文字，15–700 字符" /></label><div class="form-actions"><button class="button primary" :disabled="busy || !relation.section_id">保存关系</button></div></form>
    <div v-if="options.relations.length" class="surface-form"><h2>自己建立的关系</h2><div class="managed-models"><article v-for="edge in options.relations" :key="edge.id"><div><RouterLink :to="{ path: '/graph', query: { node: edge.source, edge: edge.id } }">{{ options.models.find(m => m.id === edge.source)?.name }} · {{ edge.relation }} · {{ options.models.find(m => m.id === edge.target)?.name }} →</RouterLink><small>{{ edge.status === 'accepted' ? '已启用 · 参与检索' : '已暂停 · 保留原文记录' }}</small></div><button class="button" :disabled="busy" @click="toggleRelation(edge)">{{ edge.status === 'accepted' ? '暂停' : '重新启用' }}</button></article></div></div>
  </section>
</template>
