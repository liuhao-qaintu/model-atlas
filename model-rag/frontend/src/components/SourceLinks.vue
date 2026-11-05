<script setup>
import { computed } from 'vue'
import { sourceType, sourceTitle } from '../sourcePresentation'

const props = defineProps({ sources: { type: Array, default: () => [] } })
const uniqueSources = computed(() => [...new Map(props.sources.map(source => [source.url, source])).values()])
</script>

<template>
  <div class="source-links">
    <template v-for="source in uniqueSources" :key="source.url">
      <RouterLink v-if="source.kind === 'internal' || source.kind === 'document'" :to="source.url"><small>{{ sourceType(source) }}</small>{{ sourceTitle(source.title) }} →</RouterLink>
      <a v-else :href="source.url" target="_blank" rel="noopener noreferrer"><small>{{ sourceType(source) }}</small>{{ sourceTitle(source.title) }} ↗</a>
    </template>
  </div>
</template>
