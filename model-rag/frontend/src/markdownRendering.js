import { Marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import katex from 'katex'
import hljs from 'highlight.js/lib/common'
import { renderBareLink } from './markdownLinks.js'
import { citationExtension, safeCitationTargets } from './citations.js'

// 只兼容常见命令被重复转义的情形，不对全部反斜杠做反转义。
const escapedCommands = new Set([
  'cdot', 'times', 'frac', 'sqrt', 'sum', 'prod', 'int', 'exp', 'log',
  'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'mu', 'sigma', 'tau',
  'epsilon', 'varepsilon', 'pi', 'phi', 'omega', 'Delta', 'Sigma',
  'in', 'notin', 'le', 'leq', 'ge', 'geq', 'neq', 'approx', 'equiv',
  'rightarrow', 'leftarrow', 'to', 'infty', 'partial', 'nabla',
  'mathbf', 'mathrm', 'mathbb', 'mathcal', 'operatorname', 'text', 'left', 'right',
])

export function normalizeMathCommands(text) {
  // 环境内的两个反斜杠是合法行分隔符，原样保留。
  if (/\\(?:begin|end)\s*\{/.test(text)) return text
  return text.replace(/(?<!\\)\\\\([A-Za-z]+)/g, (raw, name) =>
    escapedCommands.has(name) ? '\\' + name : raw)
}

const parser = new Marked()
const options = { throwOnError: false, nonStandard: true }
const math = markedKatex(options)
math.extensions = math.extensions.map(extension => ({
  ...extension,
  renderer(token) {
    return extension.renderer({ ...token, text: normalizeMathCommands(token.text) })
  },
}))
parser.use(math)
parser.use(citationExtension())

// 由 Markdown 词法器识别公式定界符，代码块和行内代码优先保护。
const renderTex = token => katex.renderToString(normalizeMathCommands(token.text),
  { ...options, displayMode: token.displayMode })
parser.use({ extensions: [
  {
    name: 'multilineDollarMath', level: 'inline',
    start(src) { return src.search(/\$\$[ \t]*\r?\n/) },
    tokenizer(src) {
      // 正文紧接公式块时仍能识别；由行内词法器保留代码跨度的优先级。
      const match = /^\$\$[ \t]*\r?\n([\s\S]+?)\r?\n[ \t]*\$\$(?!\$)(?=[ \t]*(?:\r?\n|$))/.exec(src)
      if (match) return { type: 'multilineDollarMath', raw: match[0], text: match[1].trim(), displayMode: true }
    },
    renderer: renderTex,
  },
  {
    name: 'blockTexDelimiters', level: 'block',
    start(src) { return src.search(/^ {0,3}\\\[/m) },
    tokenizer(src) {
      const match = /^ {0,3}\\\[([\s\S]*?)\\\][ \t]*(?:\n|$)/.exec(src)
      if (match) return { type: 'blockTexDelimiters', raw: match[0], text: match[1].trim(), displayMode: true }
    },
    renderer(token) { return renderTex(token) + '\n' },
  },
  {
    name: 'inlineTexDelimiters', level: 'inline',
    start(src) { return src.search(/\\[([]/) },
    tokenizer(src) {
      const match = /^(?:\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\])/.exec(src)
      if (match) return { type: 'inlineTexDelimiters', raw: match[0], text: (match[1] ?? match[2]).trim(), displayMode: match[2] !== undefined }
    },
    renderer: renderTex,
  },
] })
parser.use({ renderer: {
  link(token) { return renderBareLink(token) },
  code({ text, lang }) {
    const language = (lang || '').split(' ')[0]
    const highlighted = language && hljs.getLanguage(language)
      ? hljs.highlight(text, { language }).value : hljs.highlightAuto(text).value
    return `<pre><code class="hljs">${highlighted}</code></pre>`
  },
} })

export function renderMarkdown(content, citationTargets = []) {
  return parser.parse(content, { citationTargets: safeCitationTargets(citationTargets) })
}
