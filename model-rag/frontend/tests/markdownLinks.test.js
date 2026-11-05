import test from 'node:test'
import assert from 'node:assert/strict'
import { Marked } from 'marked'
import { renderBareLink } from '../src/markdownLinks.js'

const parser = new Marked({ renderer: { link: renderBareLink } })

test('bare source URL stops at Chinese punctuation and preserves following prose', () => {
  const rendered = parser.parse('原始文件：https://github.com/openai/CLIP/blob/commit/clip/model.py。版权与许可沿用上游仓库 LICENSE。')
  assert.match(rendered, /href="https:\/\/github.com\/openai\/CLIP\/blob\/commit\/clip\/model.py"/)
  assert.match(rendered, /<\/a>。版权与许可沿用上游仓库 LICENSE。/)
})

test('explicit links, line fragments and literal URLs in code remain unchanged', () => {
  const rendered = parser.parse('[原文](https://example.org/路径。正文)\n\nhttps://github.com/a/b.py#L10-L20\n\n`https://example.org/code。literal`')
  assert.match(decodeURI(rendered), /href="https:\/\/example.org\/路径。正文"/)
  assert.match(rendered, /href="https:\/\/github.com\/a\/b.py#L10-L20"/)
  assert.match(rendered, /<code>https:\/\/example.org\/code。literal<\/code>/)
})
