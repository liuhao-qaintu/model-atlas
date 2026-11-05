"""使用隔离内存数据库验证会话浏览，不修改用户MySQL记录。"""
import unittest
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

from auth_fixtures import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.infra import database

from backend import main
from backend.infra.database import ChatMessage, ChatSession, session_is_cloud_safe
from backend.api.sessions import list_sessions, rename_session, session_page


class SessionBrowsingTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        ChatSession.__table__.create(self.engine)
        ChatMessage.__table__.create(self.engine)
        self.patch = patch.object(database, 'engine', self.engine); self.patch.start()
        self.client = TestClient(main.app)
        self.clock = datetime(2026, 10, 14, 8, 0)
        self.a, self.b, self.empty = [str(uuid4()) for _ in range(3)]
        with Session(self.engine) as db:
            db.add_all([ChatSession(id=self.a, title='CNN结构', created_at=self.clock),
                        ChatSession(id=self.b, title='100%_说明', created_at=self.clock),
                        ChatSession(id=self.empty, title='尚无问答', created_at=self.clock)])
            db.flush()
            for sequence, sid, question in [(1, self.a, 'CNN输入'), (3, self.b, 'LSTM状态'), (5, self.a, '分类头是什么')]:
                self.add_pair(db, sequence, sid, question)
            db.commit()

    def tearDown(self):
        self.client.close(); self.patch.stop(); self.engine.dispose()

    def add_pair(self, db, sequence, sid, question):
        db.add(ChatMessage(sequence=sequence, id=str(uuid4()), session_id=sid, role='user',
                           content=question, payload={}, created_at=self.clock))
        db.add(ChatMessage(sequence=sequence + 1, id=str(uuid4()), session_id=sid, role='assistant',
                           content='回答', payload={'model_id': 'cnn-1d', 'section_id': 'cnn-1d:code',
                                                   'mode': 'research', 'cloud_allowed': False,
                                                   'evidence': [{'cloud_allowed': False}]}, created_at=self.clock))

    def test_recent_activity_sort_cursor_counts_and_empty_session(self):
        first = list_sessions(limit=1)
        self.assertEqual(first['items'][0]['id'], self.a)
        self.assertEqual(first['items'][0]['question_count'], 2)
        self.assertEqual(first['items'][0]['last_question'], '分类头是什么')
        self.assertEqual(first['items'][0]['updated_at'], '2026-10-14T08:00:00+00:00')
        self.assertTrue(first['has_more'])
        second = list_sessions(before=first['next_cursor'], limit=1)
        self.assertEqual([item['id'] for item in second['items']], [self.b])
        self.assertFalse(second['has_more'])
        self.assertIsNone(second['next_cursor'])

    def test_search_title_and_latest_question_escapes_sql_wildcards(self):
        self.assertEqual(list_sessions('分类头')['items'][0]['id'], self.a)
        self.assertEqual(list_sessions('%_')['items'][0]['id'], self.b)
        self.assertEqual(list_sessions("' OR 1=1 --")['items'], [])
        self.assertEqual(list_sessions('不存在的主题')['items'], [])

    def test_message_pages_are_chronological_and_isolated(self):
        first = session_page(self.a, limit=2)
        self.assertEqual([m['sequence'] for m in first['messages']], [5, 6])
        self.assertEqual(first['session']['question_count'], 2)
        self.assertTrue(first['has_more'])
        second = session_page(self.a, before=first['next_cursor'], limit=2)
        self.assertEqual([m['sequence'] for m in second['messages']], [1, 2])
        self.assertFalse(second['has_more'])
        self.assertNotIn('LSTM状态', [m['content'] for m in second['messages']])

    def test_rename_preserves_messages_context_activity_and_cloud_restrictions(self):
        original = session_page(self.a)
        result = rename_session(self.a, '  信号卷积复习  ')
        after = session_page(self.a)
        self.assertEqual(result['title'], '信号卷积复习')
        self.assertEqual(after['messages'], original['messages'])
        self.assertEqual(after['session']['question_count'], original['session']['question_count'])
        self.assertEqual(list_sessions()['items'][0]['last_sequence'], 6)
        self.assertFalse(session_is_cloud_safe(self.a))
        self.assertEqual(list_sessions('信号卷积')['items'][0]['id'], self.a)

    def test_invalid_title_and_missing_session(self):
        for title in ('', '   ', 'a' * 121):
            with self.assertRaises(ValueError): rename_session(self.a, title)
        missing = str(uuid4())
        with self.assertRaises(LookupError): rename_session(missing, '标题')
        with self.assertRaises(LookupError): session_page(missing)

    def test_api_read_update_not_found_and_input_limits(self):
        page = self.client.get('/api/sessions', params={'limit': 1}).json()
        self.assertEqual(page['items'][0]['id'], self.a)
        self.assertEqual(self.client.patch('/api/sessions/' + self.a, json={'title': '新标题'}).status_code, 200)
        self.assertEqual(self.client.get('/api/sessions/' + self.a + '/messages').json()['session']['title'], '新标题')
        for url in ('/api/sessions?limit=0', '/api/sessions?before=-1', '/api/sessions?query=' + 'a' * 81,
                    '/api/sessions/' + self.a + '/messages?limit=201'):
            self.assertEqual(self.client.get(url).status_code, 422)
        self.assertEqual(self.client.patch('/api/sessions/' + self.a, json={'title': '  '}).status_code, 422)
        self.assertEqual(self.client.get('/api/sessions/' + str(uuid4()) + '/messages').status_code, 404)
        response = self.client.options('/api/sessions/' + self.a, headers={
            'Origin': 'http://127.0.0.1:5174', 'Access-Control-Request-Method': 'PATCH'})
        self.assertEqual(response.status_code, 200)

    def test_long_history_can_load_every_message_without_duplicate_or_gaps(self):
        with Session(self.engine) as db:
            for number in range(100):
                self.add_pair(db, 10 + number * 2, self.a, f'较早问题{number}')
            db.commit()
        all_messages, before = [], None
        while True:
            page = session_page(self.a, before=before, limit=100)
            all_messages = page['messages'] + all_messages
            if not page['has_more']: break
            before = page['next_cursor']
        self.assertEqual(len(all_messages), 204)
        self.assertEqual(len({m['id'] for m in all_messages}), 204)
        self.assertEqual([m['sequence'] for m in all_messages], sorted(m['sequence'] for m in all_messages))
        self.assertEqual(all_messages[0]['content'], 'CNN输入')


if __name__ == '__main__': unittest.main()
