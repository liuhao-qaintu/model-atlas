"""对照应隔离额外召回、拒绝静默降级，并共享标注且保护人工判断。"""
import asyncio
from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import HTTPException

from backend.api import comparison

from backend import main
from backend.generation.evaluation_metrics import seed_judgments
from backend.rag.planning import rule_plan
from backend.rag.retrieval import Retriever
from backend.rag.pipeline import execute_plan


class ComparisonExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.active = Retriever()
        self.active.vectors = np.ones((len(self.active.chunks), 2), dtype=np.float32) / np.sqrt(2)
        self.active.vector_mask = np.ones(len(self.active.chunks), dtype=bool)
        self.active.embed = AsyncMock(side_effect=lambda texts, **kw: np.tile([.7071, .7071], (len(texts), 1)))
        self.embeddings = comparison.SharedEmbeddings(self.active)

    async def test_dense_does_not_leak_keyword_entity_or_source_boost(self):
        adapter = comparison.ControlledRetriever(self.active, 'dense', self.embeddings)
        result = await adapter.search('Torchvision ResNet-18 官方源码 forward', focus='resnet-18', strict_focus=True, use_rerank=False)
        self.assertEqual(set(result['channels']), {'向量'})
        self.assertTrue(result['evidence'])
        self.assertTrue(all(e['channels'] == ['向量'] for e in result['evidence']))
        self.assertEqual(result['paths'], [])

    async def test_shared_query_vectors_avoid_a_second_embedding(self):
        for variant in ('dense', 'hybrid'):
            result = await comparison.ControlledRetriever(self.active, variant, self.embeddings).search('CLIP 图像编码', use_rerank=False)
            self.assertIn('向量', result['channels'])
        self.assertEqual(self.active.embed.await_count, 1)
        self.assertEqual(self.embeddings.calls, 1)

    async def test_failed_vector_is_not_scored_as_a_hybrid_success(self):
        self.active.vectors = np.ones((len(self.active.chunks), 3))
        with self.assertRaisesRegex(RuntimeError, '向量检索失败'):
            await comparison.ControlledRetriever(self.active, 'hybrid', self.embeddings).search('CLIP', use_rerank=False)

    async def test_baseline_never_hops_or_reranks_even_when_plan_has_a_graph_route(self):
        plan = rule_plan('CLIP 与 ViT 的视觉主干有什么关系？', self.active)
        self.assertEqual(plan['mode'], 'relation')
        adapter = comparison.ControlledRetriever(self.active, 'hybrid', self.embeddings)
        with patch('backend.rag.pipeline.provider_post', AsyncMock()) as remote:
            result = await execute_plan(plan, adapter, graph_enabled=False, definition_expansion=False, rerank_enabled=False)
        remote.assert_not_called()
        self.assertEqual(result['paths'], [])
        self.assertEqual(len(result['steps']), 1)
        self.assertTrue(all('定义补全' not in e['channels'] for e in result['evidence']))

    async def test_local_only_scope_cannot_silently_replace_cloud_baseline(self):
        adapter = comparison.ControlledRetriever(self.active, 'dense', self.embeddings)
        with self.assertRaisesRegex(RuntimeError, '未执行'):
            await adapter.search('CLIP', allow_remote=False)
        self.active.embed.assert_not_called()


class ComparisonJudgmentTests(unittest.TestCase):
    def setUp(self):
        case = {'id': 'q', 'question': 'Q', 'reference_answer': 'A', 'relevant_section_ids': ['a']}
        self.record = {'id': 'r', 'corpus_sha256': 'v1', 'rows': [{'case': case, 'judgments': seed_judgments(case),
            'judgment_history': [], 'results': {name: {'evidence': [{'section_id': 'a'}, {'section_id': 'b'}]} for name in comparison.VARIANTS}}]}

    def judge(self, **kwargs):
        with patch.object(comparison, 'get_comparison', return_value=deepcopy(self.record)), \
             patch.object(comparison, 'corpus_fingerprint', return_value='v1'), \
             patch.object(main, 'retriever', SimpleNamespace(sections=[{'id': 'b', 'content': '原文依据'}])), \
             patch.object(comparison, 'save') as saved:
            result = comparison.judge_comparison('r', 'q', comparison.ComparisonJudgment(section_id='b', note='核查', **kwargs))
        return result, saved

    def test_one_judgment_updates_all_three_and_keeps_provenance(self):
        result, saved = self.judge(label='supporting', origin='assistant', quote='原文')
        for summary in result['summaries'].values():
            self.assertEqual(summary['returned_precision'], 1)
            self.assertEqual(summary['user_reviewed_rows'], 0)
        self.assertEqual(result['rows'][0]['judgment_history'][0]['after']['quote'], '原文')
        saved.assert_called_once()

    def test_unverified_quote_and_overwriting_user_are_rejected(self):
        with self.assertRaises(HTTPException) as invalid:
            self.judge(label='direct', quote='并非原文')
        self.assertEqual(invalid.exception.status_code, 422)
        self.record['rows'][0]['judgments']['b'] = {'label': 'irrelevant', 'origin': 'user', 'note': '人工复核'}
        with self.assertRaises(HTTPException) as conflict:
            self.judge(label='direct', origin='assistant')
        self.assertEqual(conflict.exception.status_code, 409)

    def test_failed_variant_is_excluded_and_marks_comparison_incomplete(self):
        self.record['rows'][0]['results']['dense'] = {'error': 'embedding unavailable', 'evidence': []}
        result = comparison.rescore_comparison(self.record)
        self.assertFalse(result['complete'])
        self.assertEqual(result['summaries']['dense']['evaluated'], 0)
        self.assertIsNone(result['summaries']['dense']['precision_at_5'])
        self.assertNotIn('metrics', self.record['rows'][0]['results']['hybrid'])


class ComparisonSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_corpus_replacement_during_execution_rejects_the_entire_record(self):
        active = SimpleNamespace(vectors=np.ones((1, 2)), sections=[{'id': 'a', 'content': 'original'}],
                                 index_status=lambda: {'indexed_chunks': 1, 'eligible_chunks': 1})
        case = {'id': 'q', 'question': 'Q', 'reference_answer': 'A', 'relevant_section_ids': ['a']}
        async def change_during_search(*args, **kwargs):
            main.retriever = SimpleNamespace(sections=[])
            return {'evidence': [], 'steps': [], 'paths': [], 'channels': {}, 'warnings': [], 'stop_reason': '', 'reranked': False}
        with patch.object(main, 'retriever', active), patch.object(comparison, 'cases', return_value=[case]), \
             patch.object(comparison, 'prepare_plan', return_value={}), patch.object(comparison, 'records', return_value=[]), \
             patch.object(comparison, 'execute_plan', side_effect=change_during_search), patch.object(comparison, 'save') as save:
            with self.assertRaises(HTTPException) as conflict:
                await comparison.run_comparison(comparison.ComparisonInput(case_ids=['q']))
            self.assertEqual(conflict.exception.status_code, 409)
            save.assert_not_called()

    async def test_nonconsenting_question_never_starts_remote_search(self):
        case = {'id': 'q', 'cloud_allowed': False}
        with patch.object(comparison, 'cases', return_value=[case]), patch.object(comparison, 'execute_plan', AsyncMock()) as execute:
            with self.assertRaises(HTTPException) as invalid:
                await comparison.run_comparison(comparison.ComparisonInput(case_ids=['q']))
            self.assertEqual(invalid.exception.status_code, 422)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
