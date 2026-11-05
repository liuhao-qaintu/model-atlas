<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { AE_STEPS, aeLocation } from '../aeJourney'
import { useAeJourney } from '../useAeJourney'
const props = defineProps({ stage: { type: String, required: true } })
const route = useRoute(), { paper } = useAeJourney()
const active = computed(() => route.query.flow === 'ae')
</script>

<template>
  <nav v-if="active" class="learning-trail" aria-label="AE 阅读路线">
    <RouterLink to="/learn/autoencoder" class="trail-title">AE 阅读路线</RouterLink>
    <ol><li v-for="(step, i) in AE_STEPS" :key="step.id">
      <span v-if="step.id === stage" aria-current="step"><small>{{ i + 1 }}</small>{{ step.title }}</span>
      <span v-else-if="step.id === 'followup'" class="trail-later" title="完成比较后，点击“继续追问”">{{ i + 1 }} · {{ step.title }}</span>
      <RouterLink v-else :to="aeLocation(step.id, paper)"><small>{{ i + 1 }}</small>{{ step.title }}</RouterLink>
    </li></ol>
  </nav>
</template>

<style scoped>
.learning-trail{flex:none;display:flex;align-items:center;gap:20px;padding:12px 16px;margin:0 0 22px;border:1px solid #dde5f2;border-radius:10px;background:#f8faff;font-size:12px}.trail-title{font-weight:650;color:#3c568f;white-space:nowrap}.learning-trail ol{list-style:none;display:flex;gap:16px;flex-wrap:wrap;margin:0;padding:0}.learning-trail li>a,.learning-trail li>span{display:flex;align-items:center;gap:6px;color:#6b7890}.learning-trail small{display:grid;place-items:center;width:20px;height:20px;border-radius:50%;background:#eaf0f9}.learning-trail [aria-current=step]{color:#3659b7;font-weight:650}.learning-trail [aria-current=step] small{background:#425fc1;color:white}.learning-trail .trail-later{color:#8c96a9}@media(max-width:720px){.learning-trail{align-items:start;flex-direction:column;gap:10px}.learning-trail ol{gap:10px 14px}}
</style>
