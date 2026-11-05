// 样例只组织真实页面和问题；论文按来源匹配，避免把本机数据库 ID 写进前端。
export const AE_PAPER_URL = 'https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf'
export const AE_PAPER_TITLE = 'Extracting and Composing Robust Features with Denoising Autoencoders'
export const AE_COMPARISON = `结合 Autoencoder 学习笔记与《${AE_PAPER_TITLE}》（Vincent 等，ICML 2008），只比较基础 AE 与去噪 AE 的训练输入、重构目标这两项。区分论文设定与教学代码，逐项给出依据；先不展开结构、代码和实验性能，不要补造故障诊断结论。`
export const AE_FOLLOWUPS = [
  '为什么去噪 AE 的重构目标仍是干净输入，而不是被破坏的输入？请结合上一轮比较与原文解释。',
  '那编码器与解码器的结构如何理解？请区分论文 2.2–2.3 节的设定与 Autoencoder 教学代码的维数。',
  '上一轮提到的去噪过程在逐层预训练与向更高层传播表示时有什么区别？请核对论文 2.4 节。',
  '这篇论文是否足以证明去噪 AE 可以直接诊断轴承故障？哪些内容还缺少依据？',
]
export function aePaper(documents = []) {
  return documents.find(d => d.status === 'imported' && d.source_url?.split('#')[0] === AE_PAPER_URL) || null
}
export function aeLocation(stage, paper) {
  if (stage === 'model') return { path: '/models/autoencoder', query: { section: 'io', flow: 'ae' } }
  if (stage === 'paper') return paper ? { path: `/documents/${paper.id}`, query: { flow: 'ae' } } : { path: '/learn/autoencoder' }
  if (stage === 'compare') return { path: '/research', query: { flow: 'ae', question: AE_COMPARISON } }
  return { path: '/learn/autoencoder' }
}
export function aeReadingContent(content, flow, modelId) {
  if (flow !== 'ae' || modelId !== 'autoencoder') return content
  return content.replace(/(\]\(\/models\/autoencoder\?)([^)\s]*)(\))/g, (_, start, query, end) => {
    const params = new URLSearchParams(query); params.set('flow', 'ae')
    return start + params.toString() + end
  })
}
export const AE_STEPS = [
  { id: 'model', title: '理解 AE', note: '看懂输入、潜在表示与重构输出。' },
  { id: 'paper', title: '阅读论文', note: '找到作者改变了什么，以及原文依据。' },
  { id: 'compare', title: '比较差异', note: '先对照训练输入与重构目标，再追问结构和实现。' },
  { id: 'followup', title: '继续追问', note: '带着比较结果追问，并打开引用核对。' },
]
