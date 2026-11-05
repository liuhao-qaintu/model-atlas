export function evidenceTargets(message, position) {
  const key = String(message.metadata?.turn_id || message.id || `turn-${position}`).replace(/[^A-Za-z0-9_-]/g, '')
  return (message.metadata?.evidence || []).map((_, i) => `#evidence-${key}-${i + 1}`)
}

export function safeCitationTargets(targets) {
  return targets.map(target => typeof target === 'string' && /^#[A-Za-z][A-Za-z0-9_-]*$/.test(target) ? target : null)
}

export function citationExtension() {
  return { extensions: [{
    name: 'answerCitation', level: 'inline',
    start(src) { return src.search(/(?<![A-Za-z0-9_.])\[\d+(?:\s*[,，、]\s*\d+)*\](?!\s*\()/) },
    tokenizer(src, tokens) {
      if (/[A-Za-z0-9_.]$/.test(tokens.at(-1)?.raw || '')) return
      const match = /^\[(\d+(?:\s*[,，、]\s*\d+)*)\]/.exec(src)
      if (!match) return
      const rest = src.slice(match[0].length)
      if (/^\s*\(/.test(rest) || (rest.startsWith('[') && !/^\[\d+(?:\s*[,，、]\s*\d+)*\]/.test(rest))) return
      const before = tokens.slice(-3).map(token => token.raw || '').join('')
      if (match[1].match(/[,，、]/) && (match[1].split(/[,，、]/).map(Number).includes(0) || /(?:形状|尺寸|维度|坐标|向量|数组|列表|区间|范围|映射到|取值|shape|size|tensor)\s*(?:为|是|在|[:：=])?\s*$/i.test(before))) return
      return { type: 'answerCitation', raw: match[0], numbers: [...new Set(match[1].split(/[,，、]/).map(Number))] }
    },
    renderer(token) {
      return token.numbers.map(number => {
        const href = this.parser.options.citationTargets?.[number - 1]
        return href ? `<a href="${href}" class="answer-citation" data-citation="${number}" aria-label="查看依据 ${number}">[${number}]</a>` : `[${number}]`
      }).join('')
    },
  }] }
}
