<script setup>
import { AE_PAPER_URL, AE_PAPER_TITLE, AE_STEPS, aeLocation } from '../aeJourney'
import { useAeJourney } from '../useAeJourney'
const { paper, loaded, error } = useAeJourney()
</script>

<template>
  <section class="learning-example">
    <RouterLink to="/" class="back-link">← 首页</RouterLink>
    <header><span class="eyebrow">一次完整的学习过程</span><h1>从普通 AE，读到去噪 AE。</h1><p>先理解模型，再读一篇论文，比较变化，最后带着依据追问。</p></header>
    <div class="example-layout"><div class="example-steps">
      <article v-for="(step, i) in AE_STEPS" :key="step.id"><span class="step-number">0{{ i + 1 }}</span><div><h2>{{ step.title }}</h2><p>{{ step.note }}</p>
        <RouterLink v-if="step.id !== 'followup' && (step.id === 'model' || (step.id === 'paper' && paper) || paper?.cloud_allowed)" :to="aeLocation(step.id, paper)">{{ ['开始阅读模型', '打开论文报告', '准备比较问题'][i] }} →</RouterLink>
        <small v-else-if="step.id === 'followup'">比较完成后，使用“继续追问”带入结果与引用。</small>
        <small v-else>{{ !loaded ? '正在检查样例资料…' : paper ? '本文仅限本地阅读，开启许可后可进行模型分析。' : '样例论文尚未入库，可先阅读模型。' }}</small>
      </div></article>
    </div><aside><span class="eyebrow">本次阅读的论文</span><h2>去噪自编码器</h2><p class="paper-title">{{ AE_PAPER_TITLE }}</p><p>Vincent 等 · ICML 2008</p><a :href="AE_PAPER_URL" target="_blank" rel="noopener noreferrer">作者公开原文 ↗</a><hr /><h3>读完要能回答</h3><ul><li>编码器接收什么，解码器重构什么？</li><li>去噪训练改变的是哪一步？</li><li>哪些说法来自论文，哪些只是教学配置？</li></ul><p class="example-boundary">这篇论文讨论表示学习。故障诊断效果需要另外的实验依据。</p></aside></div>
    <p v-if="error" class="notice error" role="alert">样例资料检查失败：{{ error }}</p>
  </section>
</template>

<style scoped>
.learning-example{max-width:1100px;margin:auto}.learning-example header{margin:22px 0 34px}.learning-example h1{font-size:clamp(28px,3vw,42px);letter-spacing:-1px;margin:14px 0}.learning-example header p{font-size:15px;color:var(--muted);line-height:1.9}.example-layout{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(270px,1fr);gap:32px;align-items:start}.example-steps{border-top:1px solid var(--line)}.example-steps article{display:flex;gap:22px;padding:24px 6px;border-bottom:1px solid var(--line)}.step-number{font-family:var(--font-display);font-size:27px;color:#8495be}.example-steps h2{font-size:19px;margin:0 0 9px}.example-steps p{font-size:14px;color:var(--muted);line-height:1.8;margin:0 0 12px}.example-steps a{font-size:13px;color:#425fc1}.example-steps small{color:#7d8799;line-height:1.8}aside{border:1px solid #dfe6f2;border-radius:16px;background:#fff;padding:26px}aside h2{font-size:27px;margin:16px 0}.paper-title{font-family:var(--font-display);font-size:17px;line-height:1.6;color:#34466c}aside p,aside a,aside li{font-size:13px;line-height:1.9}aside p,aside li{color:var(--muted)}aside hr{border:0;border-top:1px solid var(--line);margin:24px 0}aside h3{font-size:15px}aside ul{padding-left:18px}.example-boundary{background:#f5f7fc;border-radius:7px;padding:12px;margin-top:20px}@media(max-width:760px){.example-layout{grid-template-columns:1fr}.example-layout aside{order:-1}.example-steps article{gap:16px;padding:22px 0}}
</style>
