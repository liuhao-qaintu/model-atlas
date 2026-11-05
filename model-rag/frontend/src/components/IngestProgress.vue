<script setup>
import { computed, inject, ref } from 'vue'
const props = defineProps({ job: Object, events: { type: Array, default: () => [] } })
const developerMode = inject('developerMode', ref(false))
const status = computed(() => ({ pending: '准备分析', running: '正在阅读正文与整理证据', done: '分析报告已生成', failed: '分析未完成', skipped: '尚未调用模型', cancelled: '分析已取消' }[props.job?.status] || '等待分析'))
const percent = computed(() => props.job?.total ? Math.min(100, Math.round(props.job.progress / props.job.total * 100)) : 0)
</script>
<template><article v-if="job" class="ingest-progress" role="status"><div class="page-top"><h2>{{ status }}</h2><small v-if="job.total && ['running','pending'].includes(job.status)">{{ percent }}%</small></div><div v-if="['pending','running'].includes(job.status)" class="progress-track"><span :style="{width: `${percent || 3}%`}" /></div><p v-if="job.error">{{ job.error }}</p><p v-else-if="job.status === 'running'">切片与证据检查在后台完成，无需逐条审核。标签确认不会阻塞文献问答。</p><details v-if="developerMode && events.length"><summary>处理事件</summary><pre>{{ events.slice(-4) }}</pre></details></article></template>
<style scoped>.ingest-progress{margin:18px 0;padding:18px;border:1px solid var(--line);border-radius:12px;background:var(--paper)}h2{font-size:15px;margin:0}p,small{font-size:12px;color:var(--muted);line-height:1.8}.progress-track{height:5px;margin:12px 0;background:#f5f7fc;border-radius:99px;overflow:hidden}.progress-track span{display:block;height:100%;background:var(--green);transition:width .3s}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}</style>
