<script setup>
import { computed } from 'vue'

const props = defineProps({ metadata: { type: Object, required: true }, embedded: { type: Boolean, default: false } })
const plan = computed(() => props.metadata.plan)
const steps = computed(() => props.metadata.retrieval_steps || [])
const method = computed(() => ({ llm: 'LLM 查询规划', rules: '规则补全', rules_fallback: '规则回退', local_rules: '本地规则', clarification: '待明确对象' })[plan.value?.method] || '规则补全')
const strategy = computed(() => plan.value?.relation_route?.length ? '沿已存关系跨模型分步检索' : plan.value?.mode === 'comparison' ? '各模型分别检索' : steps.value.some(step => step.kind === 'graph_hop') ? '沿正文与关系分步检索' : '一次检索')
const contextOrigin = computed(() => ({ question: '当前提问', selected_model: '页面选择的模型', last_turn: '上一轮问答的模型', graph_entity: '问题中的关系实体', clarification_reply: '上一轮待澄清的问题' })[plan.value?.context_origin] || '当前提问')
function seconds(value) { return ((value || 0) / 1000).toFixed(2) }
</script>

<template>
  <component :is="embedded ? 'section' : 'details'" :class="['retrieval-trace', { 'evidence-panel': !embedded }]">
    <component :is="embedded ? 'h4' : 'summary'">检索过程 · {{ steps.length }} 步 <span>{{ method }}</span></component>
    <dl class="trace-plan"><template v-if="!embedded"><dt>完整问题</dt><dd>{{ plan.normalized_query }}</dd></template><dt>上下文来源</dt><dd>{{ contextOrigin }}</dd><dt>执行策略</dt><dd>{{ strategy }} · 全轮最多保留 5 个片段</dd><dt>规划 / 检索耗时</dt><dd>{{ seconds(metadata.timing?.planning_ms) }} / {{ seconds(metadata.timing?.retrieval_ms) }} 秒（不含回答生成）</dd></dl>
    <p v-if="plan.source_kinds?.includes('official_code')" class="trace-query">资料范围：固定版本官方源码。</p>
    <ol class="trace-steps">
      <li v-for="(step, index) in steps" :key="step.id" :class="{ missing: step.status === 'missing' }">
        <div class="trace-step-heading"><b>步骤 {{ index + 1 }} · {{ step.kind === 'graph_hop' ? '根据上一步扩展' : step.kind === 'comparison' ? '查找当前模型' : '查找初始依据' }}</b><span>{{ step.status === 'missing' ? '未找到正文' : `命中 ${step.found_count} 个片段` }}</span></div>
        <p class="trace-query">{{ step.query }}</p>
        <div v-for="definition in step.definition_expansions || []" :key="definition.definition_section_id" class="trace-references"><span>补充调用模块定义：</span><RouterLink :to="definition.trigger_url">{{ definition.call_quote }}</RouterLink><span>→</span><RouterLink :to="definition.definition_url">{{ definition.definition_title }}</RouterLink></div>
        <template v-if="step.anchor">
          <p>触发关系：{{ step.anchor.edge.source_name }} → {{ step.anchor.edge.relation }} → {{ step.anchor.edge.target_name }}</p>
          <div class="trace-references"><span>上一步正文依据：</span><RouterLink v-for="proof in step.anchor.proofs" :key="proof.section_id" :to="proof.url">{{ proof.reference ? `[${proof.reference}] ` : '' }}{{ proof.title }} →</RouterLink></div>
        </template>
        <div v-if="step.references.length" class="trace-references"><span>最终保留：</span><RouterLink v-for="reference in step.references" :key="reference.number" :to="reference.url">[{{ reference.number }}] {{ reference.title }} →</RouterLink></div>
        <small v-if="step.found_count && !step.new_evidence_count">本步复用了前面已命中的章节。</small>
        <small v-if="step.status === 'not_retained'">命中的片段未进入本轮最终五条依据。</small>
      </li>
    </ol>
    <small>{{ metadata.retrieval_stop_reason }}。这里展示实际检索日志；同一章节的多次命中不算多份独立文献。</small>
  </component>
</template>
