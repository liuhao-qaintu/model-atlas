<script setup>
defineProps({ plan: { type: Object, required: true }, embedded: { type: Boolean, default: false }, expanded: { type: Boolean, default: false } })
function method(plan) { return ({llm:'模型辅助规划', rules_fallback:'规则回退', local_rules:'本地规则'})[plan.method] || '规则规划' }
</script>

<template>
  <component :is="embedded ? 'section' : 'details'" class="query-plan" :open="!embedded && expanded ? true : undefined">
    <component :is="embedded ? 'h4' : 'summary'">问题理解与查询重构 <span>{{ method(plan) }}</span></component>
    <div class="query-comparison"><div><small>你的原问题</small><p>{{ plan.original_query }}</p></div><div><small>用于检索的完整问题</small><p>{{ plan.normalized_query }}</p></div></div>
    <div class="plan-scope"><span v-for="name in plan.model_names || []" :key="name" class="badge">{{ name }}</span><span>{{ plan.followup ? '连续追问' : '本轮独立问题' }}</span></div>
    <p v-if="plan.source_scope && plan.source_scope.kind !== 'general'" class="plan-history">来源范围：{{ plan.source_scope.label }}{{ plan.source_scope.origin?.includes('last_turn') ? '（结合上一轮限定）' : '' }}。论文表述与具体接口分别核对。</p>
    <ul v-if="plan.planning_basis?.length" class="plan-basis"><li v-for="basis in plan.planning_basis" :key="basis">{{ basis }}</li></ul>
    <details v-if="plan.history_questions?.length" class="plan-history"><summary>参考的最近 {{ plan.history_questions.length }} 个用户问题</summary><ol><li v-for="(question, index) in plan.history_questions" :key="index">{{ question }}</li></ol></details>
    <p v-if="plan.activities?.length" class="plan-history">启用的浏览上下文：{{ plan.activities.map(a => a.name).join('、') }}。</p>
    <div v-if="plan.queries?.length > 1" class="plan-subqueries"><small>实际子查询</small><p v-for="q in plan.queries" :key="q.id">{{ q.query }}</p></div>
  </component>
</template>
