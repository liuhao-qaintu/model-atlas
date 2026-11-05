"""文献先验只能微调相关候选；融合、重排、范围和评测快照均有验证。"""
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.infra.database import KnowledgeModel, KnowledgeDocument, KnowledgeSection, KnowledgeSource, KnowledgeRelation
from backend.infra.public_sources import document_kinds
from backend.rag.document_priority import priority_factor, prioritize
from backend.rag.retrieval import Retriever
from backend.api.evaluation import corpus_fingerprint

NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


class PriorityMathTests(unittest.TestCase):
    def test_comparison_merge_keeps_relevance_found_in_any_subquery(self):
        from backend.rag.pipeline import merge_evidence
        item = {'section_id': 'same', 'document_id': 'new', 'channels': ['实体/章节'], 'retrieval_relevant': False}
        result = merge_evidence([({'id': 's1'}, {'evidence': [item]}),
                                 ({'id': 's2'}, {'evidence': [{**item, 'retrieval_relevant': True}]})])
        self.assertTrue(result[0]['retrieval_relevant'])

    def test_freshness_halves_and_manual_boost_is_capped(self):
        doc = {'id': 'd', 'retrieval_added_at': NOW.isoformat()}
        self.assertAlmostEqual(priority_factor(doc, True, NOW)['factor'], 1.12)
        self.assertAlmostEqual(priority_factor(doc, True, NOW + timedelta(days=7))['freshness'], .06)
        self.assertEqual(priority_factor({**doc, 'retrieval_priority': 1.25}, True, NOW)['factor'], 1.30)
        self.assertEqual(priority_factor({**doc, 'retrieval_priority': 1.25}, False, NOW)['factor'], 1)

    def test_new_relevant_candidate_can_win_a_close_tie_but_not_a_large_gap(self):
        docs = [{'id': 'new', 'retrieval_added_at': NOW.isoformat(), 'retrieval_priority': 1.25}]
        old = {'section_id': 'old', 'score': .9, 'retrieval_relevant': True}
        new = {'section_id': 'new', 'document_id': 'new', 'score': .89, 'retrieval_relevant': True}
        self.assertEqual(prioritize([old, new], docs, NOW)[0]['section_id'], 'new')
        self.assertEqual(prioritize([old, {**new, 'score': .1}], docs, NOW)[0]['section_id'], 'old')
        self.assertEqual(prioritize([old, {**new, 'retrieval_relevant': False}], docs, NOW)[0]['section_id'], 'old')

    def test_missing_rerank_scores_preserve_provider_order(self):
        rows = [{'section_id': 'old'}, {'section_id': 'new', 'document_id': 'new', 'retrieval_relevant': True}]
        actual = prioritize(rows, [{'id': 'new', 'retrieval_priority': 1.25}], NOW, score_key='rerank_score')
        self.assertEqual([r['section_id'] for r in actual], ['old', 'new'])
        self.assertFalse(any(r['retrieval_adjustment']['applied'] for r in actual))


class PriorityRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_running_service_updates_priority_date_without_rebuilding_vectors(self):
        active = self.make_retriever(); active.priority_time = None
        with patch('backend.rag.retrieval.priority_day', return_value=NOW):
            first = active.document_priority_snapshot
            current = priority_factor(self.docs[1], True, active.priority_time)
        with patch('backend.rag.retrieval.priority_day', return_value=NOW + timedelta(days=7)):
            later = active.document_priority_snapshot
            decayed = priority_factor(self.docs[1], True, active.priority_time)
        self.assertNotEqual(first['at'], later['at'])
        self.assertAlmostEqual(decayed['freshness'], current['freshness'] / 2)

    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for target in ('backend.infra.config.DATA_ROOT', 'backend.infra.public_sources.DATA_ROOT'):
            self.stack.enter_context(patch(target, self.root))
        self.docs = [{'id': identity, 'model_id': 'demo', 'status': 'imported', 'sha256': identity,
                      'imported_at': (NOW - timedelta(days=90) if identity == 'old' else NOW).isoformat()}
                     for identity in ('old', 'new', 'irrelevant')]
        self.sections = [{'id': d['id'], 'model_id': 'demo', 'document_id': d['id'], 'slug': 'body',
                          'title': 'Document_' + d['id'], 'content': 'Signal processing uses convolution to extract local temporal features.',
                          'source_ids': ['source'], 'cloud_allowed': True, 'order': index}
                         for index, d in enumerate(self.docs)]
        self.tables = {KnowledgeModel: [{'id': 'demo', 'name': 'Demo', 'aliases': ['demo'], 'order': 0, 'ready': True}],
                       KnowledgeDocument: self.docs, KnowledgeSection: self.sections,
                       KnowledgeSource: [{'id': 'source', 'kind': 'document', 'url': '/documents/old'}], KnowledgeRelation: []}
        self.stack.enter_context(patch('backend.rag.retrieval.records', side_effect=lambda table: self.tables[table]))
        self.active = self.make_retriever()

    def tearDown(self):
        self.stack.close()

    def make_retriever(self):
        active = Retriever(); active.priority_time = NOW; active.vectors = None
        active.bm25.get_scores = lambda query: np.array([1.0 if c['document_id'] != 'irrelevant' else 0 for c in active.chunks])
        return active

    async def test_recent_document_moves_up_and_irrelevant_document_is_not_added(self):
        args = dict(focus='demo', strict_focus=True, retrieval_channels=['关键词'], use_rerank=False, allow_remote=False)
        before = await self.active.search('signal', document_boost=False, **args)
        after = await self.active.search('signal', **args)
        self.assertEqual(before['evidence'][0]['document_id'], 'old')
        self.assertEqual(after['evidence'][0]['document_id'], 'new')
        self.assertNotIn('irrelevant', [e['document_id'] for e in after['evidence']])
        self.assertGreater(after['evidence'][0]['retrieval_adjustment']['factor'], 1)

    async def test_explicit_source_scope_is_never_bypassed(self):
        self.docs[1]['retrieval_priority'] = 1.25
        result = await self.active.search('signal', use_rerank=False, allow_remote=False, allowed_section_ids={'old'})
        self.assertEqual([e['section_id'] for e in result['evidence']], ['old'])

    async def test_entity_only_candidate_gets_no_age_boost_without_relevance_signal(self):
        self.active.bm25.get_scores = lambda query: np.zeros(len(self.active.chunks))
        self.docs[1]['retrieval_priority'] = 1.25
        result = await self.active.search('unrelated', focus='demo', strict_focus=True, retrieval_channels=['实体/章节'], use_rerank=False, allow_remote=False)
        self.assertTrue(all(e['retrieval_adjustment']['factor'] == 1 for e in result['evidence']))

    async def test_reranker_scores_receive_the_same_bounded_prior(self):
        async def rank(base, key, endpoint, body):
            self.assertEqual(body['top_n'], len(body['documents']))
            results = [{'index': i, 'relevance_score': .9 if 'Document_old' in text else .89} for i, text in enumerate(body['documents'])]
            return {'results': sorted(results, key=lambda r: -r['relevance_score'])}
        with patch('backend.infra.config.RERANK_KEY', 'test'), patch('backend.rag.retrieval.provider_post', new=rank):
            result = await self.active.search('signal', focus='demo', strict_focus=True, top_k=1, retrieval_channels=['关键词'])
        self.assertTrue(result['reranked'])
        self.assertEqual(result['evidence'][0]['document_id'], 'new')
        self.assertEqual(result['evidence'][0]['retrieval_adjustment']['score_kind'], 'rerank_score')

    def test_priority_metadata_changes_evaluation_snapshot_not_vector_fingerprint(self):
        before = corpus_fingerprint(self.active)
        self.docs[1]['retrieval_priority'] = 1.25
        other = self.make_retriever()
        self.assertEqual(other.fingerprint, self.active.fingerprint)
        self.assertNotEqual(corpus_fingerprint(other), before)

    def test_only_active_revision_is_indexed_and_verified_source_kind_is_preserved(self):
        self.docs[0]['status'] = 'archived'
        other = self.make_retriever()
        self.assertNotIn('old', {s['id'] for s in other.sections})
        self.assertIn('old', other.archived_section_ids)
        manifest = {'entries': [{'document_id': 'original', 'import_sha256': 'verified', 'model': 'demo', 'type': 'official_code'}]}
        (self.root / 'public-source-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        revision = {'id': 'revision', 'model_id': 'demo', 'sha256': 'verified', 'provenance_document_id': 'original'}
        self.assertEqual(document_kinds([revision]), {'revision': 'official_code'})
        self.assertEqual(document_kinds([{**revision, 'sha256': 'changed'}]), {})


if __name__ == '__main__': unittest.main()
