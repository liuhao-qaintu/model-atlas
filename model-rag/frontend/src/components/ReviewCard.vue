<script setup>
import { computed, reactive, watch } from 'vue'

const props = defineProps({
  item: { type: Object, required: true },
  models: { type: Array, default: () => [] },
})
const emit = defineEmits(['approve', 'reject', 'edit'])

const roles = ['摘要', '背景', '方法', '实验', '结论', '局限', '其他']
const types = ['概念', '模型', '组件', '数据集', '任务', '指标', '结论', '其他']
const form = reactive({ top: '', sub: '', role: '其他', kind: '其他' })

watch(() => props.item, (item) => {
  const c = item?.proposed?.classification || {}
  form.top = (c.domain_path || [])[0] || ''
  form.sub = (c.domain_path || [])[1] || ''
  form.role = c.section_role || '其他'
  form.kind = c.knowledge_type || '其他'
}, { immediate: true })

const classification = computed(() => props.item?.proposed?.classification || {})
const links = computed(() => props.item?.proposed?.entity_links || [])
const confidence = computed(() => classification.value.confidence)

function modelName(id) {
  return props.models.find(model => model.id === id)?.name || id
}

function decisionText(link) {
  if (!link) return '待定'
  if (link.decision === 'link_existing') return `链接到 ${modelName(link.model_id)}`
  if (link.decision === 'new_node') return '新建节点'
  return '待定'
}

function proposed() {
  return {
    ...(props.item.proposed || {}),
    classification: {
      ...classification.value,
      domain_path: [form.top, form.sub],
      section_role: form.role,
      knowledge_type: form.kind,
    },
  }
}

function save() {
  emit('edit', { item: props.item, proposed: proposed() })
}

function approve() {
  emit('approve', { item: props.item, proposed: proposed() })
}
</script>

<template>
  <article class="review-card">
    <div class="page-top">
      <h3>{{ classification.summary || item.target_id }}</h3>
      <span class="soft-label">置信度 {{ confidence ?? '—' }}</span>
    </div>
    <p v-if="item.section" class="review-context">{{ item.section.locator }} — {{ (item.section.content || '').slice(0, 180) }}</p>
    <div class="review-form">
      <label>领域<input v-model="form.top" aria-label="领域" /></label>
      <label>子领域<input v-model="form.sub" aria-label="子领域" /></label>
      <label>角色<select v-model="form.role" aria-label="章节角色"><option v-for="role in roles" :key="role" :value="role">{{ role }}</option></select></label>
      <label>类型<select v-model="form.kind" aria-label="知识类型"><option v-for="kind in types" :key="kind" :value="kind">{{ kind }}</option></select></label>
    </div>
    <ul class="review-links">
      <li v-for="link in links" :key="link.entity">
        <strong>{{ link.entity }}</strong>
        <small>{{ decisionText(link.link) }}</small>
      </li>
    </ul>
    <div class="document-actions">
      <button class="button" type="button" @click="emit('reject', item)">拒绝</button>
      <button class="button" type="button" @click="save">保存修改</button>
      <button class="button primary" type="button" @click="approve">通过并写入</button>
    </div>
  </article>
</template>

<style scoped>
.review-card { margin: 12px 0; padding: 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.review-context { margin: 6px 0 2px; color: var(--muted); font-size: 12px; line-height: 1.7; }
.review-form { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }
.review-form label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
.review-form input, .review-form select { min-width: 0; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); color: var(--text); }
.review-links { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; padding: 0; list-style: none; font-size: 12px; }
.review-links li { display: flex; gap: 6px; padding: 4px 8px; border-radius: 999px; background: var(--soft); }
.review-links small { color: var(--muted); }
@media (max-width: 760px) { .review-form { grid-template-columns: 1fr 1fr; } }
</style>
