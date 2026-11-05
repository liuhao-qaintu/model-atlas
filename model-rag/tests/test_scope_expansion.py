import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend import main
from backend.rag import scope_expansion as scope
from backend.rag.document_scope import pin_document_plan, selected_documents
from backend.rag.planning import rule_plan
from backend.rag.source_scope import strict_method_sections


class Fixture:
    def __init__(self):
        self.models = {mid: {'id': mid, 'name': name, 'aliases': aliases, 'kind': kind, 'ready': True, 'searchable': True}
                       for mid, name, aliases, kind in [('doc', 'Paper', ['paper'], '文档'),
                       ('doc2', 'Paper 2', ['paper 2'], '文档'), ('resnet-18', 'ResNet-18', ['resnet', 'resnet-18', '残差网络'], '具体模型')]}
        self.documents = [{'id': 'a', 'title': 'Paper', 'model_id': 'doc', 'cloud_allowed': True},
                          {'id': 'b', 'title': 'Paper 2', 'model_id': 'doc2', 'cloud_allowed': True},
                          {'id': 'private', 'model_id': 'resnet-18', 'cloud_allowed': False}]
        self.sections = [self.section('a:1', 'doc', 'a', 'This paper mentions ResNet but does not define its implementation.'),
                         self.section('b:1', 'doc2', 'b', 'Another public paper.'),
                         self.section('resnet:overview', 'resnet-18', None, 'ResNet uses a residual addition y = F(x) + x.'),
                         {**self.section('secret', 'resnet-18', 'private', 'PRIVATE SECRET'), 'cloud_allowed': False}]
        self.relations, self.sources = [], {}
        self.vectors, self.document_kinds = None, {}
        self.search = AsyncMock(side_effect=self.find)

    def section(self, sid, mid, did, text):
        return dict(id=sid, model_id=mid, document_id=did, content=text, title=sid, slug=sid, cloud_allowed=True)

    def identify_models(self, text, focus=None):
        return [mid for mid, model in self.models.items() if any(a in text.lower() for a in model['aliases'])]

    async def find(self, query, focus, section_id=None, **kwargs):
        rows = [s for s in self.sections if s['id'] in kwargs['allowed_section_ids'] and (not focus or s['model_id'] == focus)]
        if section_id: rows = [s for s in rows if s['id'] == section_id]
        evidence = [{**s, 'section_id': s['id'], 'score': 1, 'channels': ['keyword'], 'reader_url': '/models/resnet-18'} for s in rows[:5]]
        return dict(evidence=evidence, paths=[], warnings=[], channels={'keyword': len(evidence)}, reranked=False)


class ScopeExpansionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.retriever = Fixture()
        self.request = main.ChatRequest(question='请搜索一下残差网络的输入输出', document_ids=['a'], model_id='doc')
        self.original = rule_plan(self.request.question, self.retriever)
        self.plan = pin_document_plan(self.original, self.retriever.documents[0])
        primary = self.retriever.sections[0]
        self.primary = dict(evidence=[{**primary, 'section_id': primary['id'], 'score': 1, 'channels': [], 'reader_url': '/documents/a'}],
                            paths=[], warnings=[], channels={}, steps=[], remote_allowed=True, duration_ms=0)

    async def test_unknown_detail_expands_to_public_graph_and_preserves_selected_documents(self):
        decision = {'sufficient': False, 'query': 'ResNet residual 输入输出', 'model_ids': ['resnet-18']}
        edge = {'id': 'r', 'source': 'resnet-18', 'target': 'residual', 'owner_model_id': 'resnet-18',
                'source_ids': ['S'], 'evidence': [{'section_id': 'resnet:overview', 'quote': 'y = F(x) + x'}]}
        with patch.object(scope, 'grade', AsyncMock(return_value=decision)), patch.object(scope, 'proven_edges', return_value=[edge]) as graph:
            result = await scope.expand_if_needed(self.request.question, self.request, self.original, self.plan, copy.deepcopy(self.primary), self.retriever)
        graph.assert_called_once()
        self.assertTrue(result['scope']['expanded']); self.assertEqual(result['scope']['graph_edges_consulted'], 1)
        self.assertIn('resnet:overview', [e['section_id'] for e in result['evidence']])
        self.assertNotIn('secret', [e['section_id'] for e in result['evidence']])
        self.assertEqual(self.plan['document_ids'], ['a'])
        self.assertTrue(self.plan['source_scope']['expanded'])
        self.assertTrue(all('secret' not in call.kwargs['allowed_section_ids'] for call in self.retriever.search.call_args_list))

    async def test_strict_scope_never_grades_or_expands(self):
        self.request.document_policy = 'strict'
        with patch.object(scope, 'grade', AsyncMock()) as grade:
            result = await scope.expand_if_needed(self.request.question, self.request, self.original, self.plan, copy.deepcopy(self.primary), self.retriever)
        grade.assert_not_called(); self.retriever.search.assert_not_called()
        self.assertFalse(result['scope']['expanded'])

    async def test_sufficient_evidence_does_not_expand(self):
        with patch.object(scope, 'grade', AsyncMock(return_value={'sufficient': True})):
            result = await scope.expand_if_needed('请介绍本文的研究内容', self.request, self.original, self.plan, copy.deepcopy(self.primary), self.retriever)
        self.assertFalse(result['scope']['expanded']); self.retriever.search.assert_not_called()

    async def test_explicit_search_overrides_mistaken_sufficient_assessment(self):
        with patch.object(scope, 'grade', AsyncMock(return_value={'sufficient': True})), patch.object(scope, 'proven_edges', return_value=[]):
            result = await scope.expand_if_needed(self.request.question, self.request, self.original, self.plan, copy.deepcopy(self.primary), self.retriever)
        self.assertTrue(result['scope']['expanded'])

    async def test_private_content_never_sent_for_assessment(self):
        self.primary['evidence'][0]['cloud_allowed'] = False
        with patch.object(scope, 'grade', AsyncMock()) as grade:
            result = await scope.expand_if_needed(self.request.question, self.request, self.original, self.plan, self.primary, self.retriever)
        grade.assert_not_called(); self.assertFalse(result['scope']['expanded'])

    async def test_grade_failure_still_handles_explicit_search(self):
        with patch.object(scope, 'grade', AsyncMock(side_effect=TimeoutError())), patch.object(scope, 'proven_edges', return_value=[]):
            result = await scope.expand_if_needed('请讲解残差网络的输入输出', self.request, self.original, self.plan, copy.deepcopy(self.primary), self.retriever)
        self.assertTrue(result['scope']['expanded']); self.assertEqual(result['scope']['assessment'], 'rules_fallback')

    async def test_document_fact_question_stays_in_document(self):
        with patch.object(scope, 'grade', AsyncMock()) as grade:
            result = await scope.expand_if_needed('本文有没有采用残差网络？', self.request, self.original, self.plan, self.primary, self.retriever)
        grade.assert_not_called(); self.assertFalse(result['scope']['expanded'])

    def test_two_documents_form_union_without_private_or_model_card(self):
        request = main.ChatRequest(question='比较两篇', document_ids=['a', 'b'])
        main.validate_context(request, self.retriever)
        plan = pin_document_plan(self.original, selected_documents(request, self.retriever))
        self.assertEqual(strict_method_sections(plan['source_scope'], self.retriever), {'a:1', 'b:1'})
        self.assertIsNone(plan['model_id']); self.assertEqual(plan['document_ids'], ['a', 'b'])

    def test_public_bundled_card_defaults_public_but_uploaded_document_does_not(self):
        card = {**self.retriever.sections[2]}; card.pop('cloud_allowed')
        private = {**self.retriever.sections[3]}; private.pop('cloud_allowed')
        self.retriever.sections = [card, private]
        self.assertEqual(strict_method_sections({'cloud_only': True}, self.retriever), {'resnet:overview'})

    async def test_each_selected_document_is_searched_independently_before_merging(self):
        from backend.rag.pipeline import execute_plan
        plan = pin_document_plan(self.original, self.retriever.documents[:2])
        result = await execute_plan(plan, self.retriever, rerank_enabled=False)
        self.assertEqual({e['document_id'] for e in result['evidence']}, {'a', 'b'})
        self.assertEqual([call.kwargs['allowed_section_ids'] for call in self.retriever.search.call_args_list], [{'a:1'}, {'b:1'}])

    async def test_followup_keeps_previously_expanded_model_as_lookup_context(self):
        original = {**self.original, 'supplementary_model_ids': ['resnet-18']}
        with patch.object(scope, 'grade', AsyncMock(return_value={'sufficient': False, 'model_ids': ['resnet-18']})) as grade, patch.object(scope, 'proven_edges', return_value=[]):
            result = await scope.expand_if_needed('它的代码呢？', self.request, original, self.plan, copy.deepcopy(self.primary), self.retriever)
        self.assertIn('ResNet-18', grade.call_args.args[0])
        self.assertTrue(result['scope']['expanded'])

    def test_invalid_or_more_than_three_documents_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException): main.validate_context(main.ChatRequest(question='q', document_ids=['missing']), self.retriever)
        from pydantic import ValidationError
        with self.assertRaises(ValidationError): main.ChatRequest(question='q', document_ids=['a', 'b', 'c', 'd'])


if __name__ == '__main__': unittest.main()
