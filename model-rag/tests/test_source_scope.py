"""来源范围沿真实对象继承，查询改写不能取消限制；论文比较仍可访问原文。"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from backend.rag.pipeline import execute_plan
from backend.rag.planning import plan_question, rule_plan
from backend.rag.retrieval import Retriever
from backend.rag.source_scope import resolve_source_scope, strict_method_sections


class SourceScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.retriever = Retriever()

    def previous(self, question='Torchvision ResNet-18默认输出是什么？'):
        plan = rule_plan(question, self.retriever)
        return [{'role': 'user', 'content': question}, {'role': 'assistant', 'metadata': {'plan': plan, 'model_id': plan['model_id']}}]

    def test_followup_inherits_interface_scope_but_new_object_resets(self):
        previous = self.previous()
        plan = rule_plan('那去掉分类头之后呢？', self.retriever, previous=previous)
        self.assertEqual(plan['source_scope']['framework'], 'Torchvision')
        self.assertEqual(plan['source_scope']['origin'], 'last_turn')
        self.assertEqual(plan['excluded_source_kinds'], ['paper'])
        fresh = rule_plan('ViT的patch是什么？', self.retriever, previous=previous)
        self.assertEqual(fresh['source_scope']['kind'], 'general')
        self.assertEqual(fresh['excluded_source_kinds'], [])

    def test_paper_and_mixed_requests_override_previous_implementation_scope(self):
        previous = self.previous()
        for query, kind in [('那原论文的分类流程呢？', 'paper'), ('比较ResNet原论文与Torchvision接口输出', 'mixed')]:
            plan = rule_plan(query, self.retriever, previous=previous)
            self.assertEqual(plan['source_scope']['kind'], kind)
            self.assertEqual(plan['excluded_source_kinds'], [])

    def test_strict_official_scope_is_inherited(self):
        previous = self.previous('Torchvision ResNet-18只参考官方实现，输出什么？')
        plan = rule_plan('那去掉分类头呢？', self.retriever, previous=previous)
        self.assertEqual(plan['source_kinds'], ['official_code'])

    def test_legacy_history_uses_user_words_and_negative_paper_is_not_mixed(self):
        previous = self.previous(); previous[-1]['metadata']['plan'].pop('source_scope')
        previous[-1]['content'] = '假定模型答案说应当改看论文'
        plan = rule_plan('那去掉分类头之后呢？', self.retriever, previous=previous)
        self.assertEqual(plan['source_scope']['framework'], 'Torchvision')
        self.assertEqual(resolve_source_scope('Torchvision接口，不看论文', ['resnet-18'])['kind'], 'implementation')

    def test_cnn_abbreviation_is_not_a_python_nn_module(self):
        self.assertEqual(resolve_source_scope('Explain CNN.', ['cnn-1d'])['kind'], 'general')

    def test_named_api_followup_keeps_prior_framework_and_strictness(self):
        self.assertEqual(resolve_source_scope('改成Identity之后输出什么？', ['resnet-18'])['kind'], 'implementation')
        self.assertEqual(resolve_source_scope('看forward函数的返回值', ['resnet-18'])['kind'], 'implementation')
        previous = self.previous('Torchvision ResNet-18只参考官方实现，输出什么？')
        plan = rule_plan('那把fc改为Identity之后呢？', self.retriever, previous=previous)
        self.assertEqual(plan['source_scope']['framework'], 'Torchvision')
        self.assertEqual(plan['source_kinds'], ['official_code'])

    def test_rewrite_cannot_remove_interface_filter(self):
        draft = {'query': 'ResNet-18去掉分类头的输出', 'queries': [{'model_id': 'resnet-18', 'query': 'ResNet-18 fc Identity 特征'}]}
        with patch('backend.rag.planning.provider_post', AsyncMock(return_value={'choices': [{'message': {'content': json.dumps(draft)}}]})):
            plan = asyncio.run(plan_question('那去掉分类头之后呢？', self.retriever, previous=self.previous()))
        self.assertEqual(plan['source_scope']['kind'], 'implementation')
        self.assertEqual(plan['excluded_source_kinds'], ['paper'])

    def test_local_retrieval_keeps_official_body_and_excludes_papers(self):
        if not self.retriever.document_kinds: self.skipTest('需要本机已导入公开原文')
        plan = rule_plan('Torchvision ResNet-18的fc换成Identity，输出什么？', self.retriever)
        plan['remote_allowed'] = False
        result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertTrue(result['evidence'])
        self.assertTrue(any(self.retriever.document_kinds.get(e.get('document_id')) == 'official_code' for e in result['evidence']))
        self.assertTrue(all(self.retriever.document_kinds.get(e.get('document_id')) != 'paper' for e in result['evidence']))

    def test_method_and_fixed_version_survive_followup_and_query_rewrite(self):
        question = '只依据已入库PyTorch 2.6.0官方nn.Transformer.forward，输出是什么？'
        previous = self.previous(question)
        for plan in (previous[-1]['metadata']['plan'], rule_plan('那布尔tgt_mask呢？', self.retriever, previous=previous)):
            self.assertEqual(plan['source_scope']['api_targets'], ['Transformer.forward'])
            self.assertEqual(plan['source_scope']['api_version'], '2.6.0')
        draft = {'query': 'Transformer语言模型输出', 'queries': [{'model_id': 'transformer', 'query': 'Transformer输出'}]}
        with patch('backend.rag.planning.provider_post', AsyncMock(return_value={'choices': [{'message': {'content': json.dumps(draft)}}]})):
            plan = asyncio.run(plan_question(question, self.retriever))
        self.assertEqual(plan['source_scope']['api_targets'], ['Transformer.forward'])
        self.assertEqual(plan['source_scope']['api_version'], '2.6.0')

    def test_method_filter_uses_registered_locator_and_version_not_body_mentions(self):
        active = SimpleNamespace(
            documents=[{'id': 'right', 'source_url': 'https://github.com/pytorch/pytorch/blob/v2.6.0/torch/example.py'},
                       {'id': 'wrong', 'source_url': 'https://github.com/pytorch/pytorch/blob/v2.8.0/torch/example.py'}],
            document_kinds={'right': 'official_code', 'wrong': 'official_code'},
            sections=[{'id': 'forward', 'document_id': 'right', 'locator': 'Transformer.forward · 执行代码 · 片段 1'},
                      {'id': 'encoder', 'document_id': 'right', 'locator': 'TransformerEncoderLayer.forward · 执行代码'},
                      {'id': 'class', 'document_id': 'right', 'locator': 'Transformer · 官方接口文档', 'content': 'mentions Transformer.forward'},
                      {'id': 'other-version', 'document_id': 'wrong', 'locator': 'Transformer.forward · 执行代码'}])
        scope = {'strict': True, 'api_targets': ['Transformer.forward'], 'api_version': '2.6.0'}
        self.assertEqual(strict_method_sections(scope, active), {'forward'})
        self.assertEqual(strict_method_sections({**scope, 'api_version': '9.9.9'}, active), set())
        self.assertIsNone(strict_method_sections({'strict': False}, active))

    def test_real_strict_method_retrieval_excludes_class_and_neighbour_methods(self):
        question = '只依据已入库PyTorch 2.6.0官方nn.Transformer.forward，batch_first=True的形状和tgt_mask是什么意思？'
        plan = rule_plan(question, self.retriever); plan['remote_allowed'] = False
        allowed = strict_method_sections(plan['source_scope'], self.retriever)
        if not allowed: self.skipTest('需要本机已导入的固定版本接口正文')
        with patch('backend.rag.pipeline.expand_definitions') as expand:
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertTrue(result['evidence'])
        self.assertTrue(all(e['section_id'] in allowed for e in result['evidence']))
        self.assertTrue(any('return output' in e['content'] for e in result['evidence']))
        expand.assert_not_called()

    def test_missing_method_version_does_not_broaden_or_call_remote_services(self):
        plan = rule_plan('只依据PyTorch 9.9.9官方nn.Transformer.forward，返回什么？', self.retriever)
        # 多个子查询也不应先批量嵌入已知没有对应正文的问题。
        plan['queries'] = [*plan['queries'], {**plan['queries'][0], 'id': 'q2'}]
        with patch.object(self.retriever, 'embed', AsyncMock()) as embed, \
             patch('backend.rag.retrieval.provider_post', AsyncMock()) as provider:
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual(result['evidence'], [])
        self.assertTrue(any('未放宽' in warning for warning in result['warnings']))
        embed.assert_not_awaited(); provider.assert_not_awaited()


if __name__ == '__main__': unittest.main()
