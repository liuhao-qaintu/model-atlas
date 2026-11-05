<script setup>
import { computed, inject, ref } from 'vue'
import QueryPlan from './QueryPlan.vue'
import RetrievalTrace from './RetrievalTrace.vue'
import RelationPaths from './RelationPaths.vue'
import SourceLinks from './SourceLinks.vue'
import FigureGallery from './FigureGallery.vue'
import MarkdownContent from './MarkdownContent.vue'
import { evidenceOrigin, evidenceExcerpt, citationGroups } from '../answerPresentation'
import { sourceTitle } from '../sourcePresentation'

const props = defineProps({ metadata: { type: Object, default: () => ({}) }, targets: { type: Array, default: () => [] }, selectedEvidence: { type: String, default: '' } })
defineEmits(['evidence'])
const developerMode = inject('developerMode', ref(false))
const evidence = computed(() => props.metadata.evidence || [])
const sources = computed(() => evidence.value.flatMap(item => item.sources || []))
const grouped = computed(() => citationGroups(evidence.value))
const incomplete = computed(() => !props.metadata.generated && props.metadata.response_kind !== 'conversation' && !props.metadata.plan?.clarification && evidence.value.length > 0)
const limited = computed(() => props.metadata.generation_diagnostics?.finish_reason === 'length')
const inquiry = computed(() => props.metadata.agentic)
const checks = computed(() => inquiry.value?.assessments?.at(-1) || [])
const checkLabel = status => ({ supported: '找到直接依据', inference: '可按条件推导', missing: '尚缺依据', conflict: '资料存在分歧' })[status] || status
const answerNotes = computed(() => {
  const warnings = (props.metadata.warnings || []).join(' ')
  return [
    /补充核对未完成/.test(warnings) ? '回答尚未完成额外核对，请结合引用检查关键条件。' : '',
    /无效引用|没有对应资料入口/.test(warnings) ? '部分引用或链接未通过检查，已保留可打开的资料入口。' : '',
    /未支持完整跨模型路径|未找到可用正文|本轮最多处理三个模型/.test(warnings) ? '现有资料只覆盖问题的一部分，建议缩小问题范围后继续追问。' : '',
  ].filter(Boolean)
})
</script>

<template>
  <div class="answer-support">
    <p v-if="metadata.retrieval_scope?.expanded" class="answer-context-note">{{ metadata.retrieval_scope.found_supplement ? '所选资料未覆盖全部问题，已补充查找知识库与有依据的模型关系。补充内容的来源见下方引用。' : '已补查知识库与模型关系，仍缺少可用正文，保留资料缺口。' }}</p>
    <p v-if="incomplete" class="notice compact" role="status">本轮未生成完整回答，当前展示资料摘录。请打开引用核对后再继续提问。</p>
    <p v-else-if="limited" class="notice compact" role="status">回答尚未完成，可以继续追问需要展开的部分。</p>
    <template v-if="evidence.length">
      <div class="answer-evidence-heading"><h3>参考来源</h3><small v-if="developerMode">{{ evidenceOrigin(metadata) }}</small></div>
      <div class="citation-list" aria-label="回答依据">
        <div v-for="group in grouped" :key="group.key" class="citation-group"><span :title="group.title">{{ group.title }}</span><div><button v-for="reference in group.references" :key="reference.number" type="button" :aria-label="`查看片段 ${reference.number}：${reference.title}`" :title="reference.title" @click="$emit('evidence', reference.number)">[{{ reference.number }}]</button></div></div>
      </div>
    </template>
    <div class="support-panels">
      <details v-if="inquiry" class="answer-inquiry">
        <summary>查证过程{{ inquiry.tool_calls ? ` · 补查 ${inquiry.tool_calls} 次` : '' }}</summary>
        <p v-if="inquiry.stop_reason === 'assessment_unavailable'" class="answer-context-note">补充查证未完整完成；以下记录不代表所有问题已得到验证。</p>
        <ul><li v-for="(check, index) in checks" :key="index">{{ check.question }} <small>· {{ checkLabel(check.status) }}</small></li></ul>
        <p v-for="step in inquiry.trace || []" :key="step.id" class="answer-context-note">{{ ({ search: '补充检索', graph_lookup: '核对模型关系', read_section: '阅读相关章节' })[step.tool] }}：{{ step.query }}{{ step.found_count ? '' : '（未找到可用正文）' }}</p>
        <small>查证状态由模型辅助判断；事实与推导请结合正文引用核对。</small>
      </details>
      <details v-if="evidence.length" class="evidence-panel answer-excerpts">
        <summary>查看引用位置与来源</summary>
        <article v-for="(item, n) in evidence" :id="targets[n]?.slice(1)" :key="item.id" tabindex="-1" :class="['evidence-item', { 'evidence-selected': selectedEvidence === targets[n]?.slice(1) }]">
          <a v-if="item.document_id" :href="item.original_file_url || `/api/documents/${item.document_id}/file?inline=true`" target="_blank" rel="noopener noreferrer"><b>[{{ n + 1 }}] {{ sourceTitle(item.title) }} ↗</b></a>
          <RouterLink v-else :to="item.reader_url || `/models/${item.model_id}?section=${item.section_slug}`"><b>[{{ n + 1 }}] {{ sourceTitle(item.title) }} →</b></RouterLink>
          <small>{{ item.source_type || (item.document_id ? '导入原文' : '整理笔记') }}{{ item.locator ? ` · ${item.locator}` : '' }}</small>
          <p v-if="item.document_id" class="answer-context-note">{{ item.page ? `原文件第 ${item.page} 页起` : '页码未能识别' }} · {{ item.paragraph_basis === 'parser_inferred' ? '段落编号由解析器生成；完整图表和公式请查看原文件。' : '定位以原文件为准。' }}</p>
          <MarkdownContent v-if="!item.document_id || developerMode" :content="evidenceExcerpt(item, 600)" />
        </article>
        <SourceLinks :sources="sources" />
        <small>整理笔记与教学示例不等于独立文献；打开原文可核对上下文和版本。</small>
      </details>
      <details v-if="metadata.links?.length || metadata.figures?.length" class="answer-further">
        <summary>延伸阅读{{ metadata.figures?.length ? '与论文图解' : '' }}</summary>
        <div v-if="metadata.links?.length" class="module-link-grid"><RouterLink v-for="link in metadata.links" :key="link.url" :to="link.url">{{ link.label }} →</RouterLink></div>
        <FigureGallery v-if="metadata.figures?.length" :figures="metadata.figures" />
      </details>
      <details v-if="!incomplete && answerNotes.length" class="answer-notes"><summary>回答说明</summary><p v-for="note in answerNotes" :key="note" class="notice compact">{{ note }}</p></details>
      <details v-if="developerMode && metadata.response_kind !== 'conversation'" class="answer-audit">
        <summary>开发者信息</summary>
        <QueryPlan v-if="metadata.plan" :plan="metadata.plan" embedded />
        <RetrievalTrace :metadata="metadata" embedded />
        <template v-if="metadata.paths?.length"><h4>使用的模型关系</h4><RelationPaths :paths="metadata.paths" /></template>
        <p class="answer-context-note">{{ metadata.generated ? `依据本地资料，由 ${metadata.execution?.llm_model || '已配置的语言模型'} 生成` : '本地资料或范围提示' }}{{ metadata.reranked ? ' · 已重排' : '' }}</p>
        <p v-for="warning in metadata.warnings || []" :key="warning" class="notice compact">{{ warning }}</p>
      </details>
    </div>
    <p v-if="metadata.response_kind === 'conversation'" class="answer-context-note">根据本会话已保存的消息回答。</p>
  </div>
</template>

<style scoped>
.answer-support { margin-top:22px; border-top:1px solid var(--line); padding-top:14px; }
.answer-evidence-heading { display:flex; align-items:baseline; flex-wrap:wrap; gap:6px 12px; margin-bottom:9px; }
.answer-evidence-heading h3 { font-size:13px; margin:0; }
.answer-evidence-heading small,.answer-context-note { font-size:12px; color:var(--muted); }
.citation-list { display:flex; flex-wrap:wrap; gap:7px; }
.citation-group { display:flex; align-items:center; gap:9px; max-width:100%; padding:5px 9px; border:1px solid #e0e6f0; border-radius:6px; background:#f8faff; font-size:12px; }
.citation-group>span { min-width:0; max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#4c6084; }
.citation-group>div { display:flex; flex:none; gap:3px; }
.citation-list button { flex:none; border:0; border-radius:3px; background:transparent; color:var(--green); padding:3px; font:inherit; font-size:12px; }
.citation-list button:hover { background:#e8eefa; }
.support-panels { display:flex; flex-wrap:wrap; gap:7px 16px; margin-top:12px; align-items:start; }
.answer-excerpts,.answer-audit,.answer-further,.answer-notes { margin:0; border:0; border-radius:0; padding:0; background:transparent; font-size:12px; }
.support-panels>details[open] { width:100%; }
.answer-inquiry { font-size:12px; }
.answer-inquiry li { margin:8px 0; }
summary { cursor:pointer; color:#52688e; padding:5px 0; }
.answer-audit[open],.answer-further[open] { padding-bottom:12px; }
.answer-audit h4 { margin-bottom:8px; }
.answer-excerpts :deep(.markdown) { font-size:12px; margin-top:10px; }
@media(max-width:600px) { .citation-list { display:grid; } }
</style>
