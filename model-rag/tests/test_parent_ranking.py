"""同章窗口不能重复占用候选名额；重排看到实际命中片段。"""
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.rag.retrieval import Retriever
from backend.rag.pipeline import final_ranking


class ParentRankingTests(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_receives_distinct_parents_and_returns_five_when_available(self):
        active = Retriever(); active.vectors = None
        model_chunks = [c for c in active.chunks if c['model_id'] == 'cnn-1d']
        duplicate = next(c['section_id'] for c in model_chunks if sum(x['section_id'] == c['section_id'] for x in model_chunks) > 1)
        scores = np.array([100 if c['section_id'] == duplicate else 1 for c in active.chunks], dtype=float)
        payloads = []
        async def rank(base, key, endpoint, body):
            payloads.append(body)
            return {'results': [{'index': i} for i in range(body['top_n'])]}
        with patch.object(active.bm25, 'get_scores', return_value=scores), patch('backend.infra.config.RERANK_KEY', 'test-only'), patch('backend.rag.retrieval.provider_post', side_effect=rank):
            result = await active.search('Conv1d 输入输出', focus='cnn-1d', strict_focus=True, source_boost=False)
        parents = [active.chunks[active.texts.index(text)]['section_id'] for text in payloads[0]['documents']]
        self.assertEqual(len(parents), len(set(parents)))
        self.assertEqual(len(result['evidence']), 5)
        self.assertTrue(all(e.get('retrieval_excerpt') for e in result['evidence']))

    async def test_combined_rerank_uses_the_matched_function_not_parent_head(self):
        candidate = {'title': 'forward', 'content': 'irrelevant module header ' * 500, 'retrieval_excerpt': 'def forward(self, x): return self.fc(x)'}
        remote = AsyncMock(return_value={'results': [{'index': 0}]})
        with patch('backend.infra.config.RERANK_KEY', 'test-only'), patch('backend.rag.pipeline.provider_post', remote):
            ranked, used = await final_ranking('fc returns', [candidate], [], True)
        self.assertTrue(used)
        self.assertIn('return self.fc(x)', remote.await_args.args[3]['documents'][0])
        self.assertNotIn('irrelevant module header', remote.await_args.args[3]['documents'][0])

    async def test_only_official_scope_cannot_leak_teaching_directory_paths(self):
        active = Retriever(); active.vectors = None
        with patch.object(active, 'paths', return_value=[{'teaching_example': 'must_not_enter_prompt'}]) as paths:
            result = await active.search('Conv1d有哪些结构关系', focus='cnn-1d', strict_focus=True,
                                         source_kinds=['official_code'], use_rerank=False, allow_remote=False)
        self.assertEqual(result['paths'], [])
        paths.assert_not_called()


if __name__ == '__main__': unittest.main()
