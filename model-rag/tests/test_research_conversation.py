"""研究转对话必须保留原答案/引文、幂等写入，并重新核对当前许可。"""
import copy
import unittest
from types import SimpleNamespace
from sqlalchemy import BigInteger, Integer, create_engine, select, func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from backend.infra.database import Base, ChatMessage, KnowledgeAnalysis
from backend.infra.research_conversation import conversation_metadata, open_research_conversation


@compiles(BigInteger, 'sqlite')
def sqlite_big_integer(type_, compiler, **kwargs):
    return compiler.visit_INTEGER(Integer(), **kwargs)


class ResearchConversationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.active = SimpleNamespace(models={'autoencoder': {'id': 'autoencoder'}},
            sections=[{'id': 'autoencoder:io', 'model_id': 'autoencoder', 'slug': 'io', 'content': 'AE reconstructs input.'},
                      {'id': 'paper:method', 'model_id': 'autoencoder', 'document_id': 'paper', 'slug': 'method',
                       'content': 'DAE reconstructs clean input.', 'cloud_allowed': True, 'page': 3}],
            documents=[{'id': 'paper', 'cloud_allowed': True}])
        self.report = {'id': 'research-test', 'type': 'agentic_research_run', 'generated': True,
            'question': '比较 AE 与 DAE', 'answer': '输入不同，重构目标均为干净输入。[1] [2]',
            'outline': {'model_ids': ['autoencoder']}, 'document_ids': [],
            'evidence': [{'section_id': 'autoencoder:io', 'excerpt': 'AE reconstructs input.'},
                         {'section_id': 'paper:method', 'excerpt': 'DAE reconstructs clean input.'}]}
        with Session(self.engine) as db:
            db.add(KnowledgeAnalysis(id=self.report['id'], payload=self.report)); db.commit()

    def tearDown(self):
        self.engine.dispose()

    def test_continue_preserves_actual_answer_question_and_citation_order(self):
        value = open_research_conversation('research-test', self.active, self.engine)
        with Session(self.engine) as db:
            rows = db.scalars(select(ChatMessage).order_by(ChatMessage.sequence)).all()
            self.assertEqual([r.content for r in rows], [self.report['question'], self.report['answer']])
            meta = rows[-1].payload
            self.assertEqual([e['section_id'] for e in meta['evidence']], ['autoencoder:io', 'paper:method'])
            self.assertEqual(meta['evidence'][1]['page'], 3)
            self.assertEqual(meta['research_run_id'], 'research-test')
            self.assertEqual(meta['document_ids'], [])
            self.assertEqual(meta['session_id'], value['session_id'])

    def test_repeated_click_reuses_session_without_duplicating_history(self):
        first = open_research_conversation('research-test', self.active, self.engine)
        second = open_research_conversation('research-test', self.active, self.engine)
        self.assertEqual(first['session_id'], second['session_id']); self.assertTrue(second['reused'])
        with Session(self.engine) as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(ChatMessage)), 2)

    def test_revoked_permission_blocks_even_existing_conversation_entry(self):
        open_research_conversation('research-test', self.active, self.engine)
        self.active.documents[0]['cloud_allowed'] = False
        with self.assertRaises(ValueError): open_research_conversation('research-test', self.active, self.engine)

    def test_stale_or_deleted_evidence_cannot_enter_history(self):
        self.active.sections.pop()
        with self.assertRaises(ValueError): open_research_conversation('research-test', self.active, self.engine)
        with Session(self.engine) as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(ChatMessage)), 0)

    def test_incomplete_report_rejected(self):
        self.report['generated'] = False
        with self.assertRaises(ValueError): conversation_metadata(self.report, self.active, 'session')

    def test_missing_report_rejected(self):
        with self.assertRaises(LookupError): open_research_conversation('missing', self.active, self.engine)

    def test_selected_document_scope_and_model_context_survive(self):
        report = copy.deepcopy(self.report); report['document_ids'] = ['paper']
        meta = conversation_metadata(report, self.active, 'session')
        self.assertEqual(meta['document_ids'], ['paper']); self.assertEqual(meta['model_id'], 'autoencoder')
        self.assertEqual(meta['plan']['model_ids'], ['autoencoder'])

    def test_non_research_record_and_empty_evidence_rejected(self):
        for update in ({'type': 'review_item'}, {'evidence': []}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                conversation_metadata({**self.report, **update}, self.active, 'session')


if __name__ == '__main__':
    unittest.main()
