<script setup>
import { computed, inject, ref } from 'vue'
const props = defineProps({ paths: { type: Array, default: () => [] } })
const developerMode = inject('developerMode', ref(false))
// 同一条关系可能出现在多条路径中，阅读视图只展示一次。
const relations = computed(() => [...new Map(props.paths.flatMap(path => path.edges || []).map(edge => [edge.id, edge])).values()])
function evidenceRoute(edge) { return { path: '/graph', query: { node: edge.source, edge: edge.id, hops: '1' } } }
</script>

<template>
  <div class="relation-list">
    <template v-if="developerMode">
      <article v-for="(path, i) in paths" :key="i" class="relation-path">
        <span class="eyebrow">{{ path.hops }} 跳联系</span>
        <div v-for="edge in path.edges" :key="edge.id" class="relation-edge">
          <strong>{{ edge.source_name }}</strong><span class="relation-verb">{{ edge.relation }} →</span><strong>{{ edge.target_name }}</strong>
          <small>{{ edge.locator }} <RouterLink :to="evidenceRoute(edge)">查看原文依据 →</RouterLink></small>
        </div>
      </article>
    </template>
    <article v-for="edge in developerMode ? [] : relations" :key="edge.id" class="relation-path">
      <div class="relation-edge">
        <strong>{{ edge.source_name }}</strong><span class="relation-verb">{{ edge.relation }} →</span><strong>{{ edge.target_name }}</strong>
        <small><RouterLink :to="evidenceRoute(edge)">查看关系与依据 →</RouterLink></small>
      </div>
    </article>
  </div>
</template>
