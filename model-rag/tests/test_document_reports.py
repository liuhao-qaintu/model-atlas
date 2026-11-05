import asyncio
import copy
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.rag import document_report as report
from backend.infra import document_reports, ingestion
from backend.infra.database import Base, KnowledgeAnalysis, KnowledgeDocument, KnowledgeModel, KnowledgeSection, KnowledgeRelation, KnowledgeSource, records, upsert_records

QUOTE = 'Autoencoder learns a representation by reconstructing the original input signal.'
SECTION = {'id': 's1', 'document_id': 'paper', 'model_id': 'doc-node', 'content': QUOTE,
           'slug': 'paper-section', 'title': 'Autoencoder', 'reader_url': '/documents/paper?part=paper-section',
           'chapter': '3.2 Autoencoder', 'locator': '3.2 Autoencoder · 解析第 1 段', 'cloud_allowed': True, 'page': 7,
           'pages': [7], 'paragraph_start': 1, 'paragraph_end': 1, 'source_ids': ['Dpaper']}


class ReportGroundingTests(unittest.IsolatedAsyncioTestCase):
    def test_batch_covers_long_section_tail_and_excludes_references(self):
        units = report.units_for([{**SECTION, 'content': 'a' * 8000 + 'TAIL'},
                                  {**SECTION, 'id': 'ref', 'structure_role': 'references'}])
        self.assertEqual(''.join(u['content'] for u in units), 'a' * 8000 + 'TAIL')
        self.assertEqual(sum(len(b) for b in report.batches_for(units)), len(units))

    def test_rejects_oversized_document_instead_of_silent_truncation(self):
        with self.assertRaises(ValueError):
            report.batches_for([{'unit_id': str(i), 'content': 'x' * 13000} for i in range(report.MAX_BATCHES + 1)])

    def test_quote_and_names_must_match_source(self):
        unit = report.units_for([SECTION])[0]
        raw = {'findings': [{'field': 'summary', 'text': '重构输入', 'proofs': [{'unit_id': unit['unit_id'], 'quote': QUOTE}]}],
               'tags': [{'name': '自编码器', 'kind': '模型', 'aliases': ['Autoencoder'], 'proofs': [{'unit_id': 'U1', 'quote': QUOTE}]},
                        {'name': 'Transformer', 'kind': '模型', 'proofs': [{'unit_id': 'U1', 'quote': QUOTE}]}]}
        findings, tags = report.normalize_map(raw, {'U1': unit})
        self.assertEqual(len(findings), 1); self.assertEqual([t['name'] for t in tags], ['自编码器'])
        raw['findings'][0]['proofs'][0]['quote'] = 'This is an invented quotation never present in the source.'
        self.assertEqual(report.normalize_map(raw, {'U1': unit})[0], [])

    def test_whitespace_normalization_keeps_actual_original(self):
        self.assertEqual(report._quote(QUOTE.replace('a representation', 'a\nrepresentation'), QUOTE), QUOTE.replace('a representation', 'a\nrepresentation'))

    def test_generic_cnn_is_not_fuzzy_linked_to_lenet(self):
        tags = report.merge_tags([{'name': 'CNN', 'kind': '模型', 'aliases': [], 'proofs': []}],
                                 [{'id': 'lenet', 'name': '2D CNN / LeNet', 'aliases': ['2d cnn', 'lenet']}])
        self.assertIsNone(tags[0]['model_id'])

    def test_generic_resnet_does_not_prove_resnet18(self):
        proof = {'section_id': 's', 'quote': 'The proposed method uses ResNet for feature extraction.'}
        target = {'id': 'resnet18', 'name': 'ResNet-18', 'aliases': ['resnet', 'resnet18']}
        result = report.merge_tags([{'name': 'ResNet', 'kind': '模型', 'aliases': ['resnet'], 'proofs': [proof]}], [target])
        self.assertIsNone(result[0]['model_id'])
        self.assertTrue(report.target_version_supported(target, [{'quote': 'This variant uses ResNet-18 as its backbone.'}]))

    def test_same_original_alias_merges_chinese_translations(self):
        result = report.merge_tags([{'name': name, 'kind': '模型', 'aliases': ['DBN'], 'proofs': []} for name in ('深度信念网络', '深度置信网络')], [])
        self.assertEqual(len(result), 1)

    async def test_private_analysis_never_calls_llm(self):
        with patch.object(report, '_chat', new_callable=AsyncMock) as chat:
            with self.assertRaises(PermissionError):
                await report.analyze_document({'id': 'paper', 'title': 'Paper', 'cloud_allowed': False}, [SECTION], [])
            chat.assert_not_called()

    async def test_reducer_cannot_invent_reference_ids(self):
        raw = {'findings': [{'field': 'summary', 'text': '讨论自编码器', 'proofs': [{'unit_id': 'U1', 'quote': QUOTE}]}]}
        reduced = {'summary': [{'text': '讨论自编码器', 'card_ids': ['C1']}],
                   'contributions': [{'text': '发明自编码器', 'card_ids': ['FAKE']}],
                   'methods': [{'text': '越栏借用证据', 'card_ids': ['C1']}]}
        with patch.object(report, '_chat', new_callable=AsyncMock, side_effect=[raw, reduced]):
            result = await report.analyze_document({'id': 'paper', 'title': 'Paper', 'cloud_allowed': True}, [SECTION], [])
        self.assertEqual(len(result['content']['summary']), 1)
        self.assertEqual(result['content']['contributions'], [])
        self.assertEqual(result['content']['methods'], [])
        self.assertEqual(result['coverage']['llm_calls'], 2)


class ReportStorageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        Base.metadata.create_all(self.engine)
        self.stack.enter_context(patch('backend.infra.database.engine', self.engine))
        upsert_records(KnowledgeModel, [{'id': 'doc-node', 'name': 'Paper', 'kind': '文档', 'tags': []},
                                      {'id': 'ae', 'name': 'Autoencoder', 'aliases': ['autoencoder']}])
        upsert_records(KnowledgeDocument, [{'id': 'paper', 'status': 'imported', 'model_id': 'doc-node'}])
        upsert_records(KnowledgeSection, [SECTION])
        self.tag = {'id': 'tag-ae', 'name': '自编码器', 'kind': '模型', 'aliases': ['Autoencoder'], 'model_id': 'ae',
                    'proofs': [{'section_id': 's1', 'quote': QUOTE, 'locator': SECTION['locator']}]}
        self.value = {'id': 'dr-report', 'type': 'document_report', 'document_id': 'paper', 'status': 'draft',
                      'created_at': '2026-11-04', 'content': {'summary': [{'text': 'Summary'}]}, 'tags': [self.tag]}
        upsert_records(KnowledgeAnalysis, [self.value])

    def tearDown(self): self.stack.close(); self.engine.dispose()

    def test_one_confirmation_creates_proven_edge_and_is_idempotent(self):
        first = document_reports.confirm_report('dr-report', ['tag-ae'])
        self.assertEqual(first['status'], 'approved')
        document_reports.confirm_report('dr-report', ['tag-ae'])
        self.assertEqual(len(records(KnowledgeRelation)), 1)
        self.assertEqual(records(KnowledgeRelation)[0]['evidence'][0]['quote'], QUOTE)
        self.assertEqual(records(KnowledgeDocument)[0]['analysis_tags'][0]['name'], '自编码器')

    def test_invalid_selection_cannot_write(self):
        with self.assertRaises(ValueError): document_reports.confirm_report('dr-report', ['fake'])
        self.assertEqual(records(KnowledgeRelation), [])

    def test_new_topic_node_is_a_label_with_graph_evidence(self):
        from backend.rag.graph import proven_edges
        topic = {**self.tag, 'id': 'topic', 'name': '表示学习', 'kind': '方法', 'aliases': ['representation'], 'model_id': None}
        value = {**self.value, 'tags': [topic]}
        upsert_records(KnowledgeAnalysis, [value])
        upsert_records(KnowledgeSource, [{'id': 'Dpaper', 'kind': 'document', 'url': '/documents/paper'}])
        document_reports.confirm_report('dr-report', ['topic'])
        topic_model = next(m for m in records(KnowledgeModel) if m['id'].startswith('tag-'))
        self.assertEqual(topic_model['kind'], '知识标签'); self.assertFalse(topic_model['ready'])
        retriever = SimpleNamespace(sections=[SECTION], sources={'Dpaper': records(KnowledgeSource)[0]},
                                    models={m['id']: m for m in records(KnowledgeModel)}, relations=records(KnowledgeRelation))
        self.assertEqual(proven_edges(retriever)[0]['relation'], '讨论主题')

    def test_stale_quote_rolls_back_entire_confirmation(self):
        value = copy.deepcopy(self.value)
        value['tags'].append({**copy.deepcopy(self.tag), 'id': 'bad', 'name': 'Other', 'proofs': [{'section_id': 's1', 'quote': 'Invented'}]})
        upsert_records(KnowledgeAnalysis, [value])
        with self.assertRaises(ValueError): document_reports.confirm_report('dr-report', ['tag-ae', 'bad'])
        self.assertEqual(records(KnowledgeRelation), [])
        self.assertEqual(document_reports.latest_report('paper')['status'], 'draft')

    def test_archived_document_cannot_be_confirmed(self):
        upsert_records(KnowledgeDocument, [{'id': 'paper', 'status': 'archived', 'model_id': 'doc-node'}])
        with self.assertRaises(ValueError): document_reports.confirm_report('dr-report', ['tag-ae'])

    def test_placeholders_retired_without_deleting_historical_review(self):
        upsert_records(KnowledgeAnalysis, [{'id': 'old', 'type': 'review_item', 'status': 'pending', 'document_id': 'paper',
                                          'proposed': {'classification': {'skipped': 'cloud_not_allowed'}}},
                                         {'id': 'real', 'type': 'review_item', 'status': 'approved', 'document_id': 'paper'}])
        document_reports.retire_placeholders('paper')
        values = {r['id']: r for r in records(KnowledgeAnalysis)}
        self.assertEqual(values['old']['status'], 'superseded'); self.assertEqual(values['real']['status'], 'approved')


class ParagraphLocationTests(unittest.TestCase):
    def test_markdown_locations_track_pages_and_fenced_headings(self):
        markdown = '<!-- page: 3 -->\n## 3.2 Method\n\nFirst paragraph.\n\n<!-- page: 4 -->\nSecond paragraph.\n\n```python\n# not a chapter\nx=1\n```'
        parts = ingestion.markdown_parts(markdown)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]['pages'], [3, 4])
        self.assertEqual(parts[0]['paragraph_start'], 1)
        self.assertIn('# not a chapter', parts[0]['content'])
        self.assertEqual(parts[0]['paragraph_basis'], 'parser_inferred')


class DocumentScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_pins_model_queries_to_selected_document_without_changing_original(self):
        from backend.rag.document_scope import pin_document_plan
        plan = {'model_id': 'autoencoder', 'model_ids': ['autoencoder'], 'mode': 'relation',
                'queries': [{'model_id': 'autoencoder', 'query': 'Autoencoder'}], 'normalized_query': 'Autoencoder',
                'section_id': 'autoencoder:io', 'clarification': '请选择模型'}
        result = pin_document_plan(plan, {'id': 'paper', 'model_id': 'doc-node', 'cloud_allowed': True})
        self.assertEqual(result['model_ids'], ['doc-node'])
        self.assertEqual(result['queries'][0]['model_id'], 'doc-node')
        self.assertEqual(result['source_scope']['document_id'], 'paper')
        self.assertIsNone(result['section_id']); self.assertIsNone(result['clarification'])
        self.assertEqual(plan['model_id'], 'autoencoder')

    def test_document_node_scope_can_be_inferred_for_older_links(self):
        from backend.rag.document_scope import selected_document
        retriever = SimpleNamespace(models={'doc-node': {'kind': '文档'}}, documents=[{'id': 'paper', 'model_id': 'doc-node'}])
        self.assertEqual(selected_document(SimpleNamespace(document_id=None, model_id='doc-node'), retriever)['id'], 'paper')

    async def test_pipeline_passes_document_whitelist_and_blocks_external_expansion(self):
        from backend.rag.document_scope import pin_document_plan
        from backend.rag.pipeline import execute_plan
        plan = pin_document_plan({'warnings': [], 'queries': [{'model_id': 'autoencoder', 'query': 'Autoencoder'}],
                                  'normalized_query': 'Autoencoder', 'original_query': '本文用AE做什么', 'section_id': None},
                                 {'id': 'paper', 'model_id': 'doc-node', 'cloud_allowed': True})
        evidence = {'id': 's1:0', 'section_id': 's1', 'model_id': 'doc-node', 'title': 'Paper',
                    'content': QUOTE, 'score': 1, 'channels': ['关键词'], 'reader_url': '/documents/paper'}
        retriever = SimpleNamespace(sections=[SECTION, {**SECTION, 'id': 'other', 'document_id': 'private', 'cloud_allowed': False}],
                                    vectors=None, document_kinds={}, search=AsyncMock(return_value={
                                        'evidence': [evidence], 'paths': [{'external': 'edge'}], 'channels': {}, 'warnings': [], 'reranked': False}))
        with patch('backend.rag.pipeline.expand_definitions') as expand:
            result = await execute_plan(plan, retriever)
        self.assertEqual(retriever.search.call_args.kwargs['allowed_section_ids'], {'s1'})
        self.assertTrue(retriever.search.call_args.kwargs['allow_remote'])
        self.assertEqual(result['paths'], []); expand.assert_not_called()


if __name__ == '__main__': unittest.main()
