import asyncio
from copy import deepcopy
import unittest
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient
from backend import main
from backend.rag.planning import rule_plan, explicit_models
from backend.rag.pipeline import execute_plan


class RecentKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=TestClient(main.app);cls.client=cls.context.__enter__()
    @classmethod
    def tearDownClass(cls):cls.context.__exit__(None,None,None)

    def test_recent_cards_have_nine_sections_fixed_sources_and_original_figures(self):
        for mid in ['mamba','mamba2','mamba3','siglip2','dinov3']:
            detail=self.client.get('/api/models/'+mid).json()
            card=[s for s in detail['sections'] if not s.get('document_id')]
            self.assertEqual(len(card),9)
            from backend.infra.figures import figure_records
            self.assertEqual(detail['figures'],figure_records(mid))
            code=next(s for s in card if s['slug']=='code')
            self.assertIn('```text',code['content'])
            self.assertTrue(all(sid in main.retriever.sources for s in card for sid in s['source_ids']))
        self.assertEqual(explicit_models('Mamba-3的输出是什么？',main.retriever),['mamba3'])
        self.assertEqual(explicit_models('比较SigLIP 2与CLIP',main.retriever),['siglip2','clip'])

    def test_method_evolution_uses_a_real_intermediate_model_in_two_hops(self):
        plan=rule_plan('Mamba-3与Mamba有什么联系？请沿方法演进说明。',main.retriever)
        self.assertEqual(plan['model_ids'],['mamba3','mamba2','mamba'])
        self.assertEqual(len(plan['relation_route']),2)
        with patch.object(main.retriever,'vectors',None),patch('backend.infra.config.RERANK_KEY',''):
            result=asyncio.run(execute_plan(plan,main.retriever))
        self.assertEqual([s['model_id'] for s in result['steps']],['mamba3','mamba2','mamba'])
        self.assertLessEqual(len(result['evidence']),5)
        self.assertEqual(result['paths'][-1]['hops'],2)

    def test_family_property_and_named_mechanism_followup_keep_the_owner(self):
        self.assertEqual(explicit_models('DINOv3的所有发布变体都是ViT吗？',main.retriever),['dinov3'])
        self.assertEqual(explicit_models('DINOv3基础视觉预训练需要文字吗？所有发布变体都是ViT吗？',main.retriever),['dinov3'])
        self.assertEqual(explicit_models('比较DINOv3和ViT的训练目标',main.retriever),['dinov3','vit'])
        previous=[{'role':'assistant','content':'前一轮比较','metadata':{'plan':{'model_ids':['dinov3','vit']}}}]
        plan=rule_plan('那Gram anchoring约束的是哪些关系？',main.retriever,previous=previous)
        self.assertEqual(plan['model_ids'],['dinov3'])
        self.assertTrue(plan['followup'])

    def test_parallel_automatic_evidence_preserves_curated_evolution_anchor(self):
        curated = next(edge for edge in main.retriever.relations
                       if edge['source'] == 'mamba3' and edge['target'] == 'mamba2' and edge['kind'] != '自动提取')
        automatic = deepcopy(curated)
        automatic.update(id='auto-parallel-evolution-test', kind='自动提取')
        with patch.object(main.retriever, 'relations', [automatic, *main.retriever.relations]):
            plan = rule_plan('Mamba-3与Mamba有什么联系？请沿方法演进说明。', main.retriever)
        self.assertEqual(plan['relation_route'][0]['edge_id'], curated['id'])
        self.assertEqual(plan['model_ids'], ['mamba3', 'mamba2', 'mamba'])
