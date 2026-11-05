import test from 'node:test'
import assert from 'node:assert/strict'
import { account,csrfHeaders,safeNext,userStorageKey } from '../src/auth.js'

test('write requests carry csrf but safe reads do not',()=>{
  assert.deepEqual(csrfHeaders('GET','atlas_csrf=abc'),{})
  assert.deepEqual(csrfHeaders('POST','other=x; atlas_csrf=abc'),{'X-CSRF-Token':'abc'})
  assert.deepEqual(csrfHeaders('PATCH',''),{})
})

test('login redirects remain on this origin',()=>{
  const base='http://127.0.0.1:5174'
  assert.equal(safeNext('/chat?model=ae',base),'/chat?model=ae')
  for(const next of ['https://evil.example','//evil.example','/\\evil.example','/login','javascript:alert(1)']) assert.equal(safeNext(next,base),'/')
})

test('stored reading and session contexts are partitioned by account',()=>{
  account.value={id:'alice'}
  const first=userStorageKey('model-atlas-session')
  account.value={id:'bob'}
  assert.notEqual(userStorageKey('model-atlas-session'),first)
  account.value=null
  assert.notEqual(userStorageKey('model-atlas-session'),first)
})
