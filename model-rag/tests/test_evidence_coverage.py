"""真实底层检索范围、子问题证据覆盖与有界补查候选池。"""
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from backend.rag.adaptive_chat import ChatEvidenceAgent
from backend.rag.retrieval import Retriever, BM25Okapi, tokens
from test_scope_expansion import Fixture


class CrossDocumentSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.r = Retriever.__new__(Retriever)
        self.r.models = {
            'ae': {'id': 'ae', 'name': 'Autoencoder', 'aliases': ['autoencoder'], 'searchable': True},
            'paper': {'id': 'paper', 'name': 'Diagnosis paper', 'aliases': ['diagnosis paper'], 'searchable': True},
        }
        self.r.chunks = [
            dict(id='c1', section_id='ae:basic', model_id='ae', document_id=None, title='AE basics',
                 content='autoencoder encoder decoder reconstruction', parent_content='autoencoder encoder decoder reconstruction', source_ids=[], cloud_allowed=True),
            dict(id='c2', section_id='paper:method', model_id='paper', document_id='paper', title='Method',
                 content='autoencoder rotating machinery speed normal fault diagnosis threshold calibration',
                 parent_content='autoencoder rotating machinery speed normal fault diagnosis threshold calibration', source_ids=[], cloud_allowed=True),
            dict(id='c3', section_id='private:method', model_id='paper', document_id='private', title='Private',
                 content='autoencoder private fault speed', parent_content='PRIVATE SECRET', source_ids=[], cloud_allowed=False),
        ]
        self.r.bm25 = BM25Okapi([tokens(c['content']) for c in self.r.chunks])
        self.r.vectors, self.r.sources, self.r.documents, self.r.document_kinds = None, {}, [], {}
        self.r._priority_time = None
        self.r.paths = lambda *_: []

    async def search(self, **options):
        return await self.r.search('autoencoder rotating machinery speed fault diagnosis', use_rerank=False,
                                   allow_remote=False, allowed_section_ids={'ae:basic', 'paper:method'}, **options)

    async def test_cross_document_scope_is_not_silently_limited_by_a_model_name(self):
        result = await self.search(restrict_models=False)
        ids = {e['section_id'] for e in result['evidence']}
        self.assertIn('paper:method', ids)
        self.assertNotIn('private:method', ids)

    async def test_default_model_reading_keeps_its_existing_scope(self):
        result = await self.search()
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'ae'})

    async def test_explicit_strict_focus_cannot_be_overridden_by_cross_document_flag(self):
        result = await self.search(focus='ae', strict_focus=True, restrict_models=False)
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'ae'})


class EvidenceCoverageTests(unittest.TestCase):
    def pick(self, rows, checks, steps=(), primary='card:0', limit=5):
        from backend.rag.adaptive_chat import select_evidence
        return [e['section_id'] for e in select_evidence(rows, checks, steps, primary, limit)]

    def test_long_first_check_cannot_displace_the_other_subquestions(self):
        rows = [{'section_id': f'card:{n}'} for n in range(5)] + [{'section_id': 'paper:application', 'document_id': 'p'}]
        checks = [{'evidence_ids': [f'card:{n}' for n in range(5)]}, {'evidence_ids': ['paper:application']}]
        selected = self.pick(rows, checks)
        self.assertIn('card:0', selected)
        self.assertIn('paper:application', selected)
        self.assertEqual(len(selected), 5)

    def test_new_paper_from_a_tool_is_retained_even_if_checks_only_quote_basics(self):
        rows = [{'section_id': f'card:{n}'} for n in range(5)] + [{'section_id': 'paper:application', 'document_id': 'p'}]
        steps = [{'new_evidence_ids': ['card:4', 'paper:application']}]
        selected = self.pick(rows, [{'evidence_ids': ['card:0']}], steps)
        self.assertIn('paper:application', selected)

    def test_shared_evidence_covers_several_checks_without_redundant_slots(self):
        rows = [{'section_id': sid} for sid in ['card:0', 'shared', 'a', 'b', 'c', 'd']]
        checks = [{'evidence_ids': ['shared', 'a']}, {'evidence_ids': ['shared', 'b']},
                  {'evidence_ids': ['c']}, {'evidence_ids': ['d']}]
        self.assertEqual(set(self.pick(rows, checks)), {'card:0', 'shared', 'a', 'c', 'd'})

    def test_missing_or_stale_ids_never_create_fabricated_evidence(self):
        self.assertEqual(self.pick([{'section_id': 'real'}], [{'evidence_ids': ['unknown']}], primary='gone'), ['real'])


class AdaptiveCoverageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.r = Fixture()
        self.primary = dict(evidence=[{**self.r.sections[2], 'section_id': 'resnet:overview', 'sources': []}],
                            warnings=[], paths=[], steps=[], channels={}, duration_ms=0, remote_allowed=True, reranked=False)
        self.plan = dict(original_query='残差网络可以用于故障诊断吗？', normalized_query='ResNet fault diagnosis', model_ids=['resnet-18'])

    async def test_global_tool_passes_explicit_scope_and_final_checks_match_final_citations(self):
        decision = AsyncMock(side_effect=[
            {'checks': [{'question': '应用', 'status': 'missing', 'evidence_ids': []}],
             'actions': [{'tool': 'search', 'query': 'ResNet fault diagnosis', 'model_id': None}]},
            {'checks': [{'question': '应用', 'status': 'supported', 'evidence_ids': ['b:1']}], 'actions': []},
        ])
        with patch('backend.rag.adaptive_chat.config.RERANK_KEY', ''):
            result = await ChatEvidenceAgent(self.r, self.plan, deepcopy(self.primary), model_call=decision).run()
        self.assertFalse(self.r.search.await_args.kwargs['restrict_models'])
        selected = [e['section_id'] for e in result['evidence']]
        check = result['agentic']['final_checks'][0]
        self.assertEqual(check['evidence_ids'], ['b:1'])
        self.assertEqual(check['references'], [selected.index('b:1') + 1])
        self.assertNotIn('PRIVATE SECRET', str(decision.call_args_list))

    async def test_a_full_pool_can_replace_stale_candidates_with_a_later_tool_result(self):
        for n in range(10):
            self.r.sections.append(self.r.section(f'old:{n}', 'doc2', 'b', f'Old related detail {n}'))
        self.r.sections.append(self.r.section('new:method', 'doc2', 'b', 'New targeted application method'))
        self.primary['evidence'] = [{**s, 'section_id': s['id']} for s in self.r.sections if s['id'].startswith('old:')]
        self.r.search = AsyncMock(return_value={'evidence': [{**self.r.sections[-1], 'section_id': 'new:method'}]})
        decision = AsyncMock(side_effect=[
            {'checks': [{'question': '缺少应用方法', 'status': 'missing', 'evidence_ids': []}],
             'actions': [{'tool': 'search', 'query': 'new application method', 'model_id': None}]},
            {'checks': [{'question': '应用方法', 'status': 'supported', 'evidence_ids': ['new:method']}], 'actions': []},
        ])
        with patch('backend.rag.adaptive_chat.config.RERANK_KEY', ''):
            agent = ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=decision)
            result = await agent.run()
        self.assertIn('new:method', {e['section_id'] for e in result['evidence']})
        self.assertLessEqual(len(agent.pool), 10)
        self.assertEqual(result['agentic']['stop_reason'], 'evidence_sufficient')


if __name__ == '__main__':
    unittest.main()
