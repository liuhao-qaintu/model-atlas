import asyncio
import unittest
from unittest.mock import patch

from auth_fixtures import TestClient
from backend import main
from backend.rag.graph import build_graph
from backend.rag.pipeline import execute_plan, retain_evidence
from backend.rag.planning import explicit_models, rule_plan
from backend.rag.retrieval import Retriever
from knowledge_fixtures import without_model_material


class ApplicationCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.retriever = Retriever()
        self.retriever.vectors = None
        self.ranking = patch('backend.infra.config.RERANK_KEY', '')
        self.ranking.start()
        self.addCleanup(self.ranking.stop)

    def test_application_readers_distinguish_checked_examples_from_pseudocode(self):
        sources = {s['id']: s for s in self.client.get('/api/sources').json()}
        for model, paper, example in [('detr','S32','A01'),('llava','S33','A02'),('dinov2','S34','A03')]:
            detail = self.client.get('/api/models/'+model).json()
            sections = {s['slug']:s for s in detail['sections']}
            self.assertEqual(detail['category'],'应用案例')
            self.assertEqual(sections['applications']['source_ids'],[paper])
            if model == 'detr':
                self.assertIn('```python', sections['code']['content'])
                self.assertEqual(sources[example]['status'], '局部运算已核对')
            else:
                self.assertIn('伪代码', sections['code']['content'])
                self.assertEqual(sources[example]['status'], '结构已核对')
            self.assertNotIn('代码已核查',sources[example]['status'])

    def test_each_case_crosses_two_sourced_steps_to_the_existing_model(self):
        for source,target,question in [('detr','transformer','DETR与Transformer有什么联系？'),('llava','clip','LLaVA与CLIP有什么联系？'),('dinov2','vit','DINOv2与ViT有什么联系？')]:
            with self.subTest(source=source):
                plan = rule_plan(question,self.retriever)
                result = asyncio.run(execute_plan(plan,self.retriever))
                self.assertEqual([s['model_id'] for s in result['steps']],[source,source,target])
                self.assertEqual(result['paths'][-1]['hops'],2)
                self.assertEqual({e['model_id'] for e in result['evidence']},{source,target})
                self.assertLessEqual(len(result['evidence']),5)
                for step in result['steps'][1:]:
                    self.assertTrue(all(p['reference'] for p in step['anchor']['proofs']))
                    for dependency in step.get('definition_expansions',[]):
                        self.assertTrue({dependency['trigger_section_id'],dependency['definition_section_id']} <= set(step['evidence_ids']))

    def test_owned_component_does_not_create_an_extra_model_object(self):
        for question,expected in [('DETR的Transformer编码器如何使用图像？','detr'),('LLaVA的CLIP视觉分支输出什么？','llava'),('DINOv2的ViT骨干输入是什么？','dinov2')]:
            self.assertEqual(explicit_models(question,self.retriever),[expected])

    def test_parallel_application_comparison_covers_all_three_cases(self):
        plan = rule_plan('比较DETR、LLaVA和DINOv2的输入输出与任务差异',self.retriever)
        result = asyncio.run(execute_plan(plan,self.retriever))
        self.assertEqual(plan['mode'],'comparison')
        self.assertEqual({e['model_id'] for e in result['evidence']},{'detr','llava','dinov2'})
        self.assertEqual(len(result['steps']),3)
        self.assertLessEqual(len(result['evidence']),5)

    def test_missing_clip_target_never_borrows_the_application_body(self):
        retriever = without_model_material(self.retriever,'clip')
        plan = rule_plan('LLaVA与CLIP有什么联系？',retriever)
        result = asyncio.run(execute_plan(plan,retriever))
        self.assertEqual(result['steps'][-1]['model_id'],'clip')
        self.assertEqual(result['steps'][-1]['status'],'missing')
        self.assertTrue(all(e['model_id']=='llava' for e in result['evidence']))

    def test_filtered_out_definition_does_not_consume_a_final_evidence_slot(self):
        candidates=[{'section_id':str(n),'model_id':'vit','score':1} for n in range(5)]
        steps=[{'evidence_ids':['0','1','2','3','4'],'definition_expansions':[{'trigger_section_id':'0','definition_section_id':'filtered-out'}]}]
        self.assertEqual(len(retain_evidence(candidates,steps,['vit'])),5)
        graph=build_graph(self.retriever,kind='论文应用')
        self.assertEqual(len(graph['edges']),17)
        self.assertEqual(graph['stats']['excluded_edges'],0)
