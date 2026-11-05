import test from 'node:test'
import assert from 'node:assert/strict'
import { reportFields, claimContext, reportScopeNote } from '../src/reportPresentation.js'

test('report headings follow the document profile and preserve legacy review semantics', () => {
  assert.equal(Object.fromEntries(reportFields({ document_type: '综述' })).contributions, '综述组织与归纳价值')
  const fields = reportFields({ report_profile: { fields: { contributions: '学习重点' } } })
  assert.equal(Object.fromEntries(fields).contributions, '学习重点')
  assert.equal(fields.length, 5)
})

test('claim context separates baseline proposed and teaching without exposing machine labels', () => {
  assert.equal(claimContext({ role: 'baseline', subject: '基础 AE' }), '基础方法 · 基础 AE')
  assert.equal(claimContext({ role: 'proposed', subject: '去噪 AE' }), '本文方法 · 去噪 AE')
  assert.equal(claimContext({ role: 'unknown' }), '')
})

test('selected chapters are visibly distinguished from whole-document analysis', () => {
  assert.equal(reportScopeNote({ coverage: { scope: 'parsed_body' } }), '')
  assert.match(reportScopeNote({ coverage: { scope: 'selected_pages', selected_pages: [2, 3], selection_label: '摘要和方法章节' } }), /摘要和方法章节.*不代表全文/)
})
