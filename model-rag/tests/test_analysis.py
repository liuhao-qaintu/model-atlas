import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from backend.rag import analysis
from backend.rag.graph import proven_edges


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.section = {'id': 'doc:test:0', 'model_id': 'llava', 'slug': 'part', 'document_id': 'test',
                        'title': '方法', 'content': 'LLaVA uses CLIP as its vision encoder.', 'source_ids': ['Dtest']}
        self.models = {'llava': {'name': 'LLaVA', 'aliases': ['llava']}, 'clip': {'name': 'CLIP', 'aliases': ['clip']}}
        self.retriever = SimpleNamespace(models=self.models, sections=[self.section], relations=[],
                                        documents=[{'id': 'test', 'model_id': 'llava', 'sha256': uuid4().hex, 'cloud_allowed': False}])

    def test_invalid_quotes_objects_and_negated_adoption_are_rejected(self):
        entities = analysis.catalog(self.retriever)
        good = {'section_id': self.section['id'], 'target': 'clip', 'quote': self.section['content'], 'relation': '使用模块'}
        self.assertIsNotNone(analysis.validate_candidate(good, {self.section['id']: self.section}, 'llava', entities))
        for field, value in [('quote', 'LLaVA uses an invented CLIP implementation.'), ('target', 'invented'), ('target', 'llava'), ('relation', '获得99%性能'), ('section_id', 'missing')]:
            candidate = {**good, field: value}
            self.assertIsNone(analysis.validate_candidate(candidate, {self.section['id']: self.section}, 'llava', entities))
        self.section['content'] = 'LLaVA does not use the CLIP text encoder.'
        self.assertIsNone(analysis.validate_candidate({**good, 'quote': self.section['content']}, {self.section['id']: self.section}, 'llava', entities))

    def test_longer_model_name_does_not_become_the_older_model(self):
        self.models.update({'mamba': {'name':'Mamba','aliases':['mamba']}, 'mamba2': {'name':'Mamba-2','aliases':['mamba-2']}})
        self.section['content'] = 'The new architecture is based on Mamba-2.'
        raw = {'section_id':self.section['id'],'target':'mamba','quote':self.section['content'],'relation':'改进方法'}
        self.assertIsNone(analysis.validate_candidate(raw, {self.section['id']:self.section}, 'llava', analysis.catalog(self.retriever)))
        self.assertIsNotNone(analysis.validate_candidate({**raw,'target':'mamba2'}, {self.section['id']:self.section}, 'llava', analysis.catalog(self.retriever)))

    def test_whitespace_recovery_returns_original_span_and_rejects_word_changes(self):
        self.section['content'] = 'LLaVA uses\nCLIP as its vision encoder.'
        raw={'section_id':self.section['id'],'target':'clip','quote':'LLaVA uses CLIP as its vision encoder.','relation':'使用模块'}
        candidate=analysis.validate_candidate(raw,{self.section['id']:self.section},'llava',analysis.catalog(self.retriever))
        self.assertEqual(candidate['quote'],self.section['content'])
        self.assertTrue(candidate['whitespace_restored'])
        self.assertIsNone(analysis.validate_candidate({**raw,'quote':'LLaVA uses CLIP as its language encoder.'},{self.section['id']:self.section},'llava',analysis.catalog(self.retriever)))
        self.assertIsNone(analysis.original_quote_span('LLaVA uses CLIP', 'LLaVA uses\nCLIP. LLaVA uses\nCLIP.'))

    def test_local_only_document_never_calls_the_provider_and_does_not_publish(self):
        with patch('backend.rag.analysis.get_analysis', return_value=None), patch('backend.rag.analysis.Session'), \
                patch('backend.rag.analysis.config.LLM_KEY', 'test-only'), patch('backend.rag.analysis.provider_post', AsyncMock()) as provider:
            result = asyncio.run(analysis.extract_relations('test', self.retriever))
        provider.assert_not_awaited()
        self.assertFalse(result['cloud_used'])
        self.assertEqual(result['candidates'][0]['status'], 'candidate')
        self.assertEqual(self.retriever.relations, [])

    def test_cloud_result_requires_verbatim_proof_and_is_cached_without_repeat_request(self):
        self.retriever.documents[0]['cloud_allowed'] = True
        response = {'choices':[{'message':{'content':'{"relations":[{"section_id":"doc:test:0","target":"clip","relation":"使用模块","quote":"LLaVA uses CLIP as its vision encoder."},{"section_id":"doc:test:0","target":"clip","relation":"使用模块","quote":"This quote is invented with CLIP."}]}'}}]}
        with patch('backend.rag.analysis.get_analysis', return_value=None), patch('backend.rag.analysis.Session'), \
                patch('backend.rag.analysis.config.LLM_KEY', 'test-only'), patch('backend.rag.analysis.provider_post', AsyncMock(return_value=response)):
            result = asyncio.run(analysis.extract_relations('test', self.retriever))
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['method'], 'llm')
        self.assertTrue(result['warnings'])
        with patch('backend.rag.analysis.get_analysis', return_value=result), patch('backend.rag.analysis.provider_post', AsyncMock()) as provider:
            cached = asyncio.run(analysis.extract_relations('test', self.retriever))
        provider.assert_not_awaited()
        self.assertTrue(cached['cached'])

    def test_review_revalidates_original_and_publishes_only_confirmed_edges(self):
        candidate = analysis.validate_candidate({'section_id':self.section['id'],'target':'clip','quote':self.section['content'],'relation':'使用模块'}, {self.section['id']:self.section}, 'llava', analysis.catalog(self.retriever))
        record = {'id':'test-analysis','document_id':'test','owner_model_id':'llava','owner_name':'LLaVA','method':'local_rules','candidates':[candidate]}
        with patch('backend.rag.analysis.get_analysis', return_value=copy.deepcopy(record)), patch('backend.rag.analysis.Session') as store:
            reviewed = analysis.review_relation('test-analysis', candidate['id'], 'accepted', self.retriever)
            edge = store.return_value.__enter__.return_value.merge.call_args_list[0].args[0].payload
        self.assertEqual(reviewed['candidates'][0]['status'], 'accepted')
        fixture = SimpleNamespace(models=self.models, sections=[self.section], relations=[edge],
                                  sources={'Dtest':{'url':'/documents/test','kind':'document'}})
        self.assertEqual(len(proven_edges(fixture)), 1)
        edge['status'] = 'rejected'
        self.assertEqual(proven_edges(fixture), [])
        self.section['content'] = 'Changed source text.'
        with patch('backend.rag.analysis.get_analysis', return_value=record):
            with self.assertRaises(ValueError):analysis.review_relation('test-analysis', candidate['id'], 'accepted', self.retriever)

    def test_ast_comparison_reports_calls_and_parameters_without_execution(self):
        self.retriever.sections = [
            {**self.section,'id':'a','content':'```python\nimport torch.nn as nn\na = nn.Linear(8, 4)\nb = nn.ReLU()\nraise RuntimeError("must never execute")\n```'},
            {**self.section,'id':'b','content':'```python\nimport torch.nn as nn\nx = nn.Linear(16, 2)\ny = nn.Tanh()\n```'},
        ]
        result = analysis.compare_code('a','b',self.retriever)
        self.assertEqual(result['shared_operators'], {'Linear':1})
        self.assertEqual(result['left_only'], {'ReLU':1})
        self.assertEqual(result['right_only'], {'Tanh':1})
        self.assertAlmostEqual(result['operator_overlap'], 1/3)
        self.assertIn('8, 4',result['left']['operators'][0]['call'])
        self.assertFalse(result['executed_code'])
        self.retriever.sections[1]['content'] = '```text\noutput = model(input)\n```'
        with self.assertRaises(ValueError):analysis.compare_code('a','b',self.retriever)


if __name__ == '__main__':unittest.main()
