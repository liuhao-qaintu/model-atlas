"""生成失败可重试并保留摘录；成功回答与资料处理限制不被重试覆盖。"""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend.api import evaluation

from backend import main


class EvaluationRetryTests(unittest.IsolatedAsyncioTestCase):
    def record(self, **updates):
        row = {'case': {'id': 'case', 'question': '残差相加有什么条件？', 'model_id': 'resnet-18', 'cloud_allowed': True},
               'answer': '服务暂不可用，以下为摘录', 'generated': False, 'grade': None,
               'answer_evidence': ['formula'], 'answer_plan': {'query': '原查询'}}
        row.update(updates)
        return {'id': 'run', 'corpus_sha256': 'current', 'rows': [row]}

    async def execute(self, record, chat):
        with patch.object(evaluation, 'get_run', side_effect=lambda _: copy.deepcopy(record)), \
             patch.object(evaluation, 'save', side_effect=lambda value: record.update(value)), \
             patch.object(evaluation, 'corpus_fingerprint', return_value='current'), \
             patch.object(main, 'retriever', SimpleNamespace()), patch.object(main, 'chat', chat):
            return await evaluation.generate_answer('run', 'case')

    async def test_fallback_retry_preserves_previous_attempt_and_returns_real_answer(self):
        chat = AsyncMock(return_value={'answer': '两路需对齐，广播不等于架构语义匹配。', 'generated': True,
                                      'evidence': [{'section_id': 'formula'}], 'plan': {'query': '新查询'}, 'warnings': []})
        result = await self.execute(self.record(), chat)
        row = result['rows'][0]
        self.assertTrue(row['generated']); self.assertEqual(chat.await_count, 1)
        self.assertFalse(row['generation_attempts'][0]['generated'])
        self.assertIn('摘录', row['generation_attempts'][0]['answer'])
        self.assertEqual(row['generation_attempts'][0]['answer_plan']['query'], '原查询')
        self.assertIsNone(row['grade'])

    async def test_completed_answer_is_reused_without_another_model_call(self):
        chat = AsyncMock()
        result = await self.execute(self.record(answer='已完成回答', generated=True, grade='correct'), chat)
        self.assertEqual(result['rows'][0]['grade'], 'correct'); chat.assert_not_awaited()

    async def test_retry_respects_corpus_version_and_cloud_permission(self):
        chat = AsyncMock(); record = self.record(); record['corpus_sha256'] = 'old'
        with self.assertRaises(HTTPException) as stale:
            await self.execute(record, chat)
        self.assertEqual(stale.exception.status_code, 409)
        record = self.record(); record['rows'][0]['case']['cloud_allowed'] = False
        with self.assertRaises(HTTPException) as local:
            await self.execute(record, chat)
        self.assertEqual(local.exception.status_code, 422); chat.assert_not_awaited()


if __name__ == '__main__': unittest.main()
