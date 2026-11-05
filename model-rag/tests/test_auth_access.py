"""认证与资料隔离回归；必须使用显式SQLite测试配置，绝不重建真实业务库。"""
import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,select,text,update,inspect
from sqlalchemy.orm import Session
import numpy as np
from backend import main
from backend.infra import config,database,auth,ingest_job
from backend.infra.access import Actor,acting_as,current_actor
from backend.infra.database import Base,Account,KnowledgeModel,KnowledgeSection,KnowledgeSource,KnowledgeDocument,KnowledgeRelation,KnowledgeAnalysis,ChatSession,ChatMessage
from backend.infra.secure_retriever import RetrieverRegistry
from backend.infra.ingestion import UPLOAD_ROOT

PASSWORD='TEST_ONLY_password_123!'
UIDS={'alice':'a'*32,'bob':'b'*32,'maintainer':'c'*32,'admin':'d'*32}
DIDS={'alice':'1'*32,'bob':'2'*32}
SIDS={'alice':'11111111-1111-4111-8111-111111111111','bob':'22222222-2222-4222-8222-222222222222'}


def model_payload(identity,name,origin='user'):
    return {'id':identity,'name':name,'aliases':[name],'description':'test fixture','category':'测试','kind':'用户模型' if origin=='user' else '模型','origin':origin,'order':1,'ready':True,'source_ids':['S-test'],'tags':[],'section_count':1}


@unittest.skipUnless(database.engine.dialect.name=='sqlite','需要隔离SQLite测试配置')
class AuthAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash=auth.password_hash(PASSWORD)

    def setUp(self):
        if database.engine.dialect.name!='sqlite':raise RuntimeError('Refuse non-SQLite cleanup')
        Base.metadata.drop_all(database.engine)
        Base.metadata.create_all(database.engine)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        option=patch.object(config,'DATA_ROOT',Path(self.temp.name));option.start();self.addCleanup(option.stop)
        auth.LOGIN_ATTEMPTS.clear();ingest_job._jobs.clear()
        for name,value in [('AUTH_SECRET_KEY','TEST_ONLY_auth_signing_key_not_used_by_live_service'),('MAINTAINER_USER_IDS',[UIDS['maintainer']])]:
            option=patch.object(config,name,value);option.start();self.addCleanup(option.stop)
        with Session(database.engine) as db:
            db.add_all([Account(id=uid,username=name,display_name=name,role='admin' if name in {'admin','maintainer'} else 'user',active=True,password_hash=self.password_hash) for name,uid in UIDS.items()])
            db.add(KnowledgeModel(id='shared',payload=model_payload('shared','SharedModel','model_card')))
            db.add(KnowledgeSource(id='S-test',payload={'id':'S-test','title':'公开来源','url':'https://example.test/shared','topic':'shared','kind':'official'}))
            db.add(KnowledgeSection(id='shared:overview',model_id='shared',payload={'id':'shared:overview','model_id':'shared','slug':'overview','title':'公共原理','order':0,'source_ids':['S-test'],'content':'PUBLIC_BASE_CONTENT 共用模型原理与输入输出。','origin':'model_card','cloud_allowed':False}))
            db.add(KnowledgeDocument(id='legacy',payload={'id':'legacy','model_id':'shared','title':'LEGACY_PRIVATE','filename':'old.md','format':'md','status':'imported','parts':[],'markdown':'LEGACY_SECRET'}))
            db.commit()
        for name in ['alice','bob']:
            marker=name.upper()+'_PRIVATE_ONLY'
            did=DIDS[name]
            with acting_as(Actor(UIDS[name])):
                with Session(database.engine) as db:
                    db.add(KnowledgeModel(id='private-'+name,payload=model_payload('private-'+name,name.upper()+'_MODEL_SECRET')))
                    db.add(KnowledgeDocument(id=did,payload={'id':did,'model_id':'shared','title':name+' private paper','filename':name+'.md','format':'md','status':'imported','parts':[{'content':marker,'locator':'§1'}],'markdown':marker,'cloud_allowed':False}))
                    db.add(KnowledgeSource(id='D'+did,payload={'id':'D'+did,'title':name+' private source','url':'/documents/'+did,'kind':'document','topic':'shared'}))
                    db.add(KnowledgeSection(id='doc:'+did+':0',model_id='shared',payload={'id':'doc:'+did+':0','model_id':'shared','document_id':did,'slug':'doc-'+did+'-0','title':name+' private section','order':1000,'source_ids':['D'+did],'content':marker,'origin':'document','cloud_allowed':False}))
                    db.add(KnowledgeRelation(id='relation-'+name,payload={'id':'relation-'+name,'source':'private-'+name,'target':'shared','source_name':name.upper()+'_MODEL_SECRET','target_name':'SharedModel','relation':'使用模块','kind':'结构组成','owner_model_id':'shared','document_id':did,'source_ids':['D'+did],'locator':'§1','scope':'测试依据','evidence':[{'section_id':'doc:'+did+':0','quote':marker}]}))
                    db.add(KnowledgeAnalysis(id='research-'+name,payload={'id':'research-'+name,'type':'agentic_research_run','protocol':'agentic-evidence-gap-v1','question':marker,'answer':marker,'generated':True,'evidence':[],'generation_diagnostics':{'rejected_answer':'INTERNAL_DRAFT'}}))
                    db.add(ChatSession(id=SIDS[name],title=marker));db.flush()
                    db.add(ChatMessage(id=str(uuid4()),session_id=SIDS[name],role='user',content=marker,payload={}))
                    db.add(ChatMessage(id=str(uuid4()),session_id=SIDS[name],role='assistant',content=marker,payload={'generation_diagnostics':{'rejected_answer':'INTERNAL_DRAFT'}}))
                    db.commit()
            UPLOAD_ROOT.mkdir(parents=True,exist_ok=True)
            (UPLOAD_ROOT/(did+'.md')).write_text(marker,encoding='utf-8')
        main.retriever=RetrieverRegistry()
        self.client=TestClient(main.app)
        self.addCleanup(self.client.close)

    def tearDown(self):
        Base.metadata.drop_all(database.engine)
        Base.metadata.create_all(database.engine)
        main.retriever=None

    def headers(self,name='alice'):
        response=self.client.post('/api/auth/login',json={'username':name,'password':PASSWORD})
        self.assertEqual(response.status_code,200,response.text)
        token=self.client.cookies.get(auth.COOKIE)
        self.client.cookies.clear()
        return {'Authorization':'Bearer '+token}

    def test_anonymous_is_blocked_and_health_is_minimal(self):
        self.assertEqual(self.client.get('/api/health').json(),{'status':'ok'})
        for path in ['/api/models','/api/documents','/api/sessions','/api/research/runs/research-alice','/api/documents/'+DIDS['alice']+'/file']:
            self.assertEqual(self.client.get(path).status_code,401,path)
        self.assertEqual(self.client.get('/docs').status_code,404)

    def test_cookie_login_and_me_do_not_expose_passwords(self):
        response=self.client.post('/api/auth/login',json={'username':'alice','password':PASSWORD})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers.get('cache-control'),'no-store')
        cookies=response.headers.get_list('set-cookie')
        self.assertEqual(len(cookies),2)
        self.assertTrue(any('HttpOnly' in cookie for cookie in cookies))
        self.assertNotIn(PASSWORD,response.text)
        self.assertNotIn(self.password_hash,response.text)
        self.assertEqual(self.client.get('/api/auth/me').json()['id'],UIDS['alice'])

    def test_health_shows_only_current_users_index_readiness(self):
        for name in ['alice','bob']:
            result=self.client.get('/api/health',headers=self.headers(name))
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()['documents'],1)
            self.assertIn('eligible_chunks',result.json()['index'])
            self.assertNotIn('execution',result.json())
            self.assertNotIn('generation_prompt',result.json())
        self.assertEqual(self.client.get('/api/health',headers={'Authorization':'Bearer expired'}).json(),{'status':'ok'})

    def test_document_index_is_scoped_and_does_not_grant_maintenance(self):
        headers=self.headers()
        result=self.client.post('/api/documents/index',headers=headers,json={})
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(result.json()['local_only_chunks'],2)
        self.assertEqual(self.client.post('/api/index',headers=headers,json={}).status_code,403)

    def test_real_chat_and_recall_keep_private_evidence_and_saved_owner(self):
        headers=self.headers()
        result=self.client.post('/api/chat',headers=headers,json={'question':'公共原理和输入输出是什么？','model_id':'shared'})
        self.assertEqual(result.status_code,200,result.text)
        self.assertNotIn('BOB_PRIVATE_ONLY',result.text)
        session=result.json()['session_id']
        followup=self.client.post('/api/chat',headers=headers,json={'question':'我上一个问题是什么？','session_id':session})
        self.assertEqual(followup.status_code,200,followup.text)
        self.assertIn('公共原理和输入输出是什么',followup.json()['answer'])
        self.assertEqual(self.client.get('/api/sessions/'+session,headers=self.headers('bob')).status_code,404)
        with Session(database.engine) as db:
            self.assertEqual(db.get(ChatSession,session).owner_id,UIDS['alice'])
            self.assertEqual({row.owner_id for row in db.scalars(select(ChatMessage).where(ChatMessage.session_id==session))},{UIDS['alice']})

    def test_background_threads_inherit_owner_scope(self):
        async def read():
            with acting_as(Actor(UIDS['alice'])):
                return await asyncio.to_thread(database.records,KnowledgeDocument)
        rows=asyncio.run(read())
        self.assertEqual([row['id'] for row in rows],[DIDS['alice']])

    def test_maintenance_refresh_only_publishes_server_seed_content(self):
        def seed():
            with Session(database.engine) as db:
                db.add(KnowledgeModel(id='new-seed',payload=model_payload('new-seed','NewShared','model_card')))
                db.commit()
        with patch.object(main,'seed_knowledge',seed):
            result=self.client.post('/api/knowledge/refresh',headers=self.headers('maintainer'),json={})
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(self.client.get('/api/models/new-seed',headers=self.headers('bob')).status_code,200)
        with Session(database.engine) as db:
            self.assertTrue(db.get(KnowledgeModel,'new-seed').is_public)
            self.assertIsNone(db.get(KnowledgeDocument,'legacy').owner_id)

    def test_same_shared_evidence_produces_private_relationships_for_each_user(self):
        from backend.api.management import RelationInput,store_relation
        quote='SharedModel uses SharedOther as a feature processing module.'
        with Session(database.engine) as db:
            db.add(KnowledgeModel(id='other-shared',payload=model_payload('other-shared','SharedOther','model_card')))
            row=db.get(KnowledgeSection,'shared:overview');row.payload={**row.payload,'content':quote}
            db.commit()
        value=RelationInput(source='shared',target='other-shared',relation='采用架构',section_id='shared:overview',quote=quote)
        identities=[]
        for name in ['alice','bob']:
            with acting_as(Actor(UIDS[name])):
                identities.append(store_relation(value,main.scoped_retriever())['id'])
        self.assertNotEqual(*identities)
        with Session(database.engine) as db:
            self.assertEqual([db.get(KnowledgeRelation,identity).owner_id for identity in identities],[UIDS['alice'],UIDS['bob']])

    def test_same_named_report_tags_are_separate_for_each_user(self):
        from backend.infra.document_reports import confirm_report
        identities=[]
        for name in ['alice','bob']:
            with acting_as(Actor(UIDS[name])):
                with Session(database.engine) as db:
                    section=db.get(KnowledgeSection,'doc:'+DIDS[name]+':0')
                    section.payload={**section.payload,'content':'Wavelet processing extracts useful signal features.'}
                    report={'id':'tags-'+name,'type':'document_report','document_id':DIDS[name],
                            'tags':[{'id':'tag1','name':'Wavelet','kind':'信号处理','aliases':[],
                                     'proofs':[{'section_id':section.id,'quote':'Wavelet processing extracts useful signal features.','locator':'§1'}]}]}
                    db.add(KnowledgeAnalysis(id=report['id'],payload=report));db.commit()
                value=confirm_report(report['id'],['tag1'])
                identities.append(value['confirmed_tags'][0]['graph_node_id'])
        self.assertNotEqual(*identities)
        with Session(database.engine) as db:
            self.assertEqual([db.get(KnowledgeModel,identity).owner_id for identity in identities],[UIDS['alice'],UIDS['bob']])

    def test_cookie_writes_require_signed_csrf_and_trusted_origin(self):
        self.client.post('/api/auth/login',json={'username':'alice','password':PASSWORD})
        data={'name':'Own model'}
        self.assertEqual(self.client.post('/api/models',json=data).status_code,403)
        self.assertEqual(self.client.post('/api/models',headers={'X-CSRF-Token':'wrong'},json=data).status_code,403)
        headers={'X-CSRF-Token':self.client.cookies.get(auth.CSRF_COOKIE)}
        self.assertEqual(self.client.post('/api/models',headers={**headers,'Origin':'https://evil.example.test'},json=data).status_code,403)
        response=self.client.post('/api/models',headers=headers,json=data)
        self.assertEqual(response.status_code,200,response.text)
        with Session(database.engine) as db:
            row=db.get(KnowledgeModel,response.json()['id'])
            self.assertEqual(row.owner_id,UIDS['alice']);self.assertFalse(row.is_public)

    def test_registration_cannot_choose_role_or_echo_password(self):
        data={'username':'new_user','password':PASSWORD,'role':'admin'}
        response=self.client.post('/api/auth/register',json=data)
        self.assertEqual(response.status_code,422);self.assertNotIn(PASSWORD,response.text)
        data.pop('role')
        response=self.client.post('/api/auth/register',json=data)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['role'],'user')
        self.assertFalse(response.json()['is_maintainer'])

    def test_password_whitespace_is_preserved(self):
        password='  TEST_ONLY_spaces_123!  '
        self.assertEqual(self.client.post('/api/auth/register',json={'username':'space_user','password':password}).status_code,200)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'space_user','password':password}).status_code,200)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'space_user','password':password.strip()}).status_code,401)

    def test_password_change_revokes_old_cookie_and_bearer(self):
        headers=self.headers()
        response=self.client.post('/api/auth/password',headers=headers,json={'old_password':PASSWORD,'new_password':'TEST_ONLY_changed_789!'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.get('/api/auth/me',headers=headers).status_code,401)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'alice','password':'TEST_ONLY_changed_789!'}).status_code,200)

    def test_disabled_account_and_invalid_tokens_fail(self):
        headers=self.headers()
        with Session(database.engine) as db:db.get(Account,UIDS['alice']).active=False;db.commit()
        self.assertEqual(self.client.get('/api/models',headers=headers).status_code,401)
        self.assertEqual(self.client.get('/api/models',headers={'Authorization':'Bearer not-a-jwt'}).status_code,401)

    def test_logout_clears_both_cookies(self):
        self.client.post('/api/auth/login',json={'username':'alice','password':PASSWORD})
        result=self.client.post('/api/auth/logout',headers={'X-CSRF-Token':self.client.cookies.get(auth.CSRF_COOKIE)},json={})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.client.get('/api/auth/me').status_code,401)

    def test_business_admin_is_not_maintenance_and_cannot_read_others(self):
        headers=self.headers('admin')
        self.assertEqual(self.client.get('/api/analysis/status',headers=headers).status_code,403)
        self.assertEqual(self.client.get('/api/evaluation/runs',headers=headers).status_code,403)
        self.assertEqual(self.client.get('/api/documents',headers=headers).json(),[])
        self.assertEqual(self.client.get('/api/research/runs/research-alice',headers=headers).status_code,404)
        headers=self.headers('maintainer')
        self.assertNotIn(self.client.get('/api/analysis/status',headers=headers).status_code,[401,403])
        self.assertEqual(self.client.get('/api/documents',headers=headers).json(),[])

    def test_document_and_section_lists_are_scoped_before_retrieval(self):
        for name in ['alice','bob']:
            headers=self.headers(name)
            result=self.client.get('/api/documents',headers=headers)
            self.assertEqual([item['id'] for item in result.json()],[DIDS[name]])
            model=self.client.get('/api/models/shared',headers=headers)
            self.assertIn(name.upper()+'_PRIVATE_ONLY',model.text)
            other='bob' if name=='alice' else 'alice'
            self.assertNotIn(other.upper()+'_PRIVATE_ONLY',model.text)
            self.assertEqual(self.client.get('/api/documents/'+DIDS[other],headers=headers).status_code,404)
            self.assertEqual(self.client.get('/api/documents/legacy',headers=headers).status_code,404)

    def test_files_are_private_even_when_url_is_known(self):
        owner=self.headers();other=self.headers('bob')
        path='/api/documents/'+DIDS['alice']+'/file'
        result=self.client.get(path,headers=owner)
        self.assertEqual(result.status_code,200);self.assertEqual(result.text,'ALICE_PRIVATE_ONLY')
        self.assertEqual(result.headers.get('cache-control'),'no-store')
        self.assertEqual(self.client.get(path,headers=other).status_code,404)

    def test_personal_library_excludes_shared_documents_but_keeps_citation_access(self):
        with Session(database.engine) as db:
            db.add(KnowledgeDocument(id='shared-paper',is_public=True,payload={'id':'shared-paper','model_id':'shared','title':'公共参考文献',
                   'filename':'public.md','format':'md','status':'imported','parts':[],'markdown':'public evidence','cloud_allowed':False}))
            db.commit()
            row=db.get(KnowledgeDocument,'shared-paper');row.is_public=True;db.commit()
        main.retriever=RetrieverRegistry()
        for name in ['alice','bob']:
            headers=self.headers(name)
            self.assertEqual([row['id'] for row in self.client.get('/api/documents',headers=headers).json()],[DIDS[name]])
            self.assertEqual(self.client.get('/api/documents/shared-paper',headers=headers).status_code,200)
            with acting_as(Actor(UIDS[name])):
                self.assertIn('shared-paper',{row['id'] for row in main.scoped_retriever().documents})

    def test_sessions_lists_history_and_rename_are_private(self):
        owner=self.headers();other=self.headers('bob')
        result=self.client.get('/api/sessions',headers=other)
        self.assertNotIn('ALICE_PRIVATE_ONLY',result.text)
        for path in ['/api/sessions/'+SIDS['alice'],'/api/sessions/'+SIDS['alice']+'/messages']:
            self.assertEqual(self.client.get(path,headers=other).status_code,404)
            response=self.client.get(path,headers=owner)
            self.assertEqual(response.status_code,200,response.text)
            self.assertNotIn('INTERNAL_DRAFT',response.text)
        self.assertEqual(self.client.patch('/api/sessions/'+SIDS['alice'],headers=other,json={'title':'steal'}).status_code,404)

    def test_foreign_session_is_rejected_before_generation(self):
        with patch.object(main,'chat_events') as generate:
            for path in ['/api/chat','/api/chat/stream','/api/search']:
                response=self.client.post(path,headers=self.headers('bob'),json={'question':'Explain model','model_id':'shared','session_id':SIDS['alice']})
                self.assertEqual(response.status_code,404,response.text)
            generate.assert_not_called()

    def test_research_reports_and_open_conversation_check_owner(self):
        headers=self.headers('bob')
        self.assertEqual(self.client.get('/api/research/runs/research-alice',headers=headers).status_code,404)
        self.assertEqual(self.client.post('/api/research/runs/research-alice/conversation',headers=headers,json={}).status_code,404)
        result=self.client.get('/api/research/runs/research-bob',headers=headers)
        self.assertEqual(result.status_code,200);self.assertNotIn('INTERNAL_DRAFT',result.text)

    def test_memory_jobs_are_owned_including_event_and_cancel_routes(self):
        with acting_as(Actor(UIDS['alice'])):job=ingest_job.create_job(DIDS['alice'],UIDS['alice'])
        headers=self.headers('bob')
        for path in ['/api/ingest/jobs/'+job.id,'/api/ingest/jobs/'+job.id+'/events']:
            self.assertEqual(self.client.get(path,headers=headers).status_code,404)
        self.assertEqual(self.client.post('/api/ingest/jobs/'+job.id+'/cancel',headers=headers,json={}).status_code,404)
        self.assertEqual(self.client.get('/api/ingest/jobs/'+job.id,headers=self.headers()).status_code,200)

    def test_same_upload_for_two_users_has_separate_ids_and_files(self):
        ids=[]
        for name in ['alice','bob']:
            headers=self.headers(name)
            result=self.client.post('/api/documents/preview',headers=headers,files={'file':('same.md',b'# A test paper\n\nShared file used by two private users.')},data={'model_id':'shared'})
            self.assertEqual(result.status_code,200,result.text)
            did=result.json()['id'];ids.append(did)
            result=self.client.post('/api/documents/'+did+'/import',headers=headers,json={'title':'same paper','cloud_allowed':False})
            self.assertEqual(result.status_code,200,result.text)
        self.assertNotEqual(*ids)
        self.assertTrue(all((UPLOAD_ROOT/(did+'.md')).is_file() for did in ids))
        self.assertEqual(self.client.get('/api/documents/'+ids[0]+'/file',headers=self.headers('bob')).status_code,404)

    def test_graph_and_vector_cache_exclude_other_users_material(self):
        raw=main.retriever.current()
        cache=config.DATA_ROOT/'embeddings.npz'
        np.savez_compressed(cache,keys=np.array(raw.chunk_keys),vectors=np.ones((len(raw.chunk_keys),8),dtype=np.float32))
        main.retriever=RetrieverRegistry()
        with acting_as(Actor(UIDS['bob'])):
            view=main.retriever.current()
            serialized=json.dumps({'models':view.models,'sections':view.sections,'sources':view.sources,'relations':view.relations,'chunks':view.chunks})
            self.assertNotIn('ALICE_PRIVATE_ONLY',serialized)
            self.assertNotIn('ALICE_MODEL_SECRET',serialized)
            self.assertTrue(set(view.vector_cache).issubset(set(view.chunk_keys)))
            self.assertIn(UIDS['bob'],str(view.cache_path))
        result=self.client.get('/api/graph',headers=self.headers('bob'))
        self.assertEqual(result.status_code,200,result.text)
        self.assertNotIn('ALICE_MODEL_SECRET',result.text)

    def test_orm_merge_and_bulk_updates_cannot_overwrite_foreign_rows(self):
        with acting_as(Actor(UIDS['bob'])):
            with Session(database.engine) as db:
                db.merge(KnowledgeDocument(id=DIDS['alice'],payload={'id':DIDS['alice'],'title':'stolen'}))
                with self.assertRaises(HTTPException):db.commit()
                db.rollback()
            with Session(database.engine) as db:
                db.execute(update(KnowledgeDocument).values(payload={'id':DIDS['bob'],'title':'bob changed'}));db.commit()
        with Session(database.engine) as db:
            self.assertEqual(db.get(KnowledgeDocument,DIDS['alice']).payload['title'],'alice private paper')
            self.assertEqual(db.get(KnowledgeDocument,'legacy').payload['title'],'LEGACY_PRIVATE')

    def test_client_cannot_publish_or_choose_another_owner_on_new_rows(self):
        with acting_as(Actor(UIDS['bob'])):
            with Session(database.engine) as db:
                db.add(KnowledgeDocument(id='forged',owner_id=UIDS['alice'],is_public=True,payload={'id':'forged'}));db.commit()
        with Session(database.engine) as db:
            row=db.get(KnowledgeDocument,'forged')
            self.assertEqual(row.owner_id,UIDS['bob']);self.assertFalse(row.is_public)

    def test_sse_keeps_actor_context_and_removes_internal_diagnostics(self):
        async def generate(*args,**kwargs):
            self.assertEqual(current_actor.get().id,UIDS['bob'])
            yield 'stage',{'label':'working'}
            yield 'done',{'answer':'ok','session_id':str(uuid4()),'evidence':[],'generation_diagnostics':{'rejected_answer':'INTERNAL_DRAFT'},
                         'agentic':{'diagnostics':[{'decision':'INTERNAL_AGENT_DECISION'}],'trace':[{'query':'可公开的依据查找说明'}]}}
        with patch.object(main,'chat_events',generate):
            response=self.client.post('/api/chat/stream',headers=self.headers('bob'),json={'question':'test','model_id':'shared'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('event: done',response.text);self.assertNotIn('INTERNAL_DRAFT',response.text)
        self.assertNotIn('INTERNAL_AGENT_DECISION',response.text)
        self.assertIn('可公开的依据查找说明',response.text)
        self.assertEqual(response.headers.get('cache-control'),'no-store')

    def test_nested_research_diagnostics_are_private_even_via_history(self):
        with acting_as(Actor(UIDS['alice'])):
            with Session(database.engine) as db:
                row=db.get(KnowledgeAnalysis,'research-alice')
                row.payload={**row.payload,'diagnostics':[{'raw_result':'INTERNAL_RESEARCH_PLAN'}],
                             'agentic':{'diagnostics':[{'decision':'INTERNAL_RESEARCH_DECISION'}]}}
                message=db.scalar(select(ChatMessage).where(ChatMessage.session_id==SIDS['alice'],ChatMessage.role=='assistant'))
                message.payload={**message.payload,'agentic':{'diagnostics':[{'detail':'INTERNAL_AGENT_ERROR'}]},'raw_content':'INTERNAL_RAW_CONTENT'}
                db.commit()
        headers=self.headers('alice')
        for path in ['/api/research/runs/research-alice','/api/sessions/'+SIDS['alice'],'/api/sessions/'+SIDS['alice']+'/messages']:
            response=self.client.get(path,headers=headers)
            self.assertEqual(response.status_code,200,response.text)
            self.assertNotIn('INTERNAL_',response.text)

    def test_maintainer_keeps_own_diagnostics_without_cross_user_access(self):
        with acting_as(Actor(UIDS['maintainer'],'admin',True)):
            with Session(database.engine) as db:
                db.add(KnowledgeAnalysis(id='research-maintainer',payload={'id':'research-maintainer','type':'agentic_research_run',
                       'answer':'ok','diagnostics':[{'raw_result':'OWN_MAINTENANCE_TRACE'}]}));db.commit()
        headers=self.headers('maintainer')
        response=self.client.get('/api/research/runs/research-maintainer',headers=headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('OWN_MAINTENANCE_TRACE',response.text)
        self.assertEqual(self.client.get('/api/research/runs/research-alice',headers=headers).status_code,404)

    def test_legacy_claim_requires_explicit_local_admin_action(self):
        spec=importlib.util.spec_from_file_location('atlas_account_cli',Path(__file__).resolve().parents[1]/'tools/manage_accounts.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch.object(sys,'argv',['accounts','create-admin','--username','legacy_owner','--claim-legacy']),patch('builtins.input',return_value=PASSWORD),contextlib.redirect_stdout(io.StringIO()):cli.main()
        with Session(database.engine) as db:
            owner=db.scalar(select(Account).where(Account.username=='legacy_owner'))
            self.assertEqual(db.get(KnowledgeDocument,'legacy').owner_id,owner.id)
            self.assertFalse(db.get(KnowledgeDocument,'legacy').is_public)
            self.assertEqual(db.get(KnowledgeDocument,DIDS['alice']).owner_id,UIDS['alice'])
            self.assertTrue(db.get(KnowledgeModel,'shared').is_public)

    def test_new_columns_migrate_without_publishing_legacy_document(self):
        engine=create_engine('sqlite://')
        self.addCleanup(engine.dispose)
        with engine.begin() as conn:
            conn.execute(text('CREATE TABLE knowledge_documents (id VARCHAR(40) PRIMARY KEY,payload JSON NOT NULL)'))
            conn.execute(text('INSERT INTO knowledge_documents VALUES (:id,:payload)'),{'id':'old','payload':json.dumps({'id':'old','title':'old private'})})
        with patch.object(database,'engine',engine):
            database.initialize_tables();database.initialize_tables()
        columns=[item['name'] for item in inspect(engine).get_columns('knowledge_documents')]
        self.assertEqual(columns.count('owner_id'),1);self.assertEqual(columns.count('is_public'),1)
        with Session(engine) as db:
            old=db.get(KnowledgeDocument,'old')
            self.assertIsNone(old.owner_id);self.assertFalse(old.is_public)
            self.assertEqual(old.payload['title'],'old private')


if __name__=='__main__':unittest.main()
