import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldSendOnEnter } from '../src/composerKeys.js'
test('Enter sends; Ctrl+Enter stays compatible', () => {
  assert.equal(shouldSendOnEnter({ key: 'Enter' }), true)
  assert.equal(shouldSendOnEnter({ key: 'Enter', ctrlKey: true }), true)
})
test('Shift+Enter and Alt+Enter keep multiline editing', () => {
  assert.equal(shouldSendOnEnter({ key: 'Enter', shiftKey: true }), false)
  assert.equal(shouldSendOnEnter({ key: 'Enter', altKey: true }), false)
})
test('IME composition and legacy 229 never submit a Chinese candidate', () => {
  assert.equal(shouldSendOnEnter({ key: 'Enter', isComposing: true }), false)
  assert.equal(shouldSendOnEnter({ key: 'Enter', keyCode: 229 }), false)
  assert.equal(shouldSendOnEnter({ key: 'a' }), false)
})
