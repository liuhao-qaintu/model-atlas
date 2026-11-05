"""规划、范围、顺序多跳与可追溯引用；外部服务使用替身。"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
from auth_fixtures import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend import main
from backend.infra.database import ChatMessage, ChatSession, engine, history
from backend.rag.pipeline import execute_plan
from backend.rag.planning import plan_question, rule_plan, validate_draft
from backend.rag.retrieval import Retriever
from knowledge_fixtures import without_model_material


def response(text):
    return {"choices": [{"message": {"content": text}}]}


def plan_response(query, queries):
    return response(json.dumps({"query": query, "queries": [{"model_id": mid, "query": text} for mid, text in queries]}, ensure_ascii=False))


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()
        cls.sessions = []

    @classmethod
    def tearDownClass(cls):
        with Session(engine) as db:
            db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(cls.sessions)))
            db.execute(delete(ChatSession).where(ChatSession.id.in_(cls.sessions)))
            db.commit()
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.retriever = Retriever()
        self.retriever.vectors = None
        self.key = patch("backend.infra.config.LLM_KEY", "test-placeholder")
        self.rerank = patch("backend.infra.config.RERANK_KEY", "")
        self.key.start(); self.rerank.start()
        self.addCleanup(self.key.stop); self.addCleanup(self.rerank.stop)

    def chat(self, **fields):
        session_id = fields.get("session_id") or str(uuid4())
        if session_id not in self.sessions:
            self.sessions.append(session_id)
        with patch.object(main.retriever, "vectors", None):
            return self.client.post("/api/chat", json={"session_id": session_id, **fields})

    def test_simple_question_has_no_extra_planner_call(self):
        provider = AsyncMock()
        with patch("backend.rag.planning.provider_post", provider):
            plan = asyncio.run(plan_question("1D CNN 输入输出是什么？", self.retriever))
            result = asyncio.run(execute_plan(plan, self.retriever))
        provider.assert_not_awaited()
        self.assertEqual(plan["method"], "rules")
        self.assertEqual(len(result["steps"]), 1)
        self.assertLessEqual(len(result["evidence"]), 5)

    def test_followup_uses_llm_rewrite_without_previous_generated_answer(self):
        previous = [{"role": "user", "content": "ResNet-18 的默认输出是什么？", "metadata": {}},
                    {"role": "assistant", "content": "DO_NOT_COPY_PREVIOUS_ANSWER", "metadata": {"model_id": "resnet-18", "evidence": []}}]
        provider = AsyncMock(return_value=plan_response("ResNet-18 去掉分类头之后输出什么？", [("resnet-18", "ResNet-18 fc Identity 特征输出")]))
        with patch("backend.rag.planning.provider_post", provider):
            plan = asyncio.run(plan_question("它去掉分类头之后呢？", self.retriever, previous=previous))
        self.assertEqual(plan["method"], "llm")
        self.assertNotIn("它", plan["normalized_query"])
        self.assertEqual(plan["model_id"], "resnet-18")
        self.assertNotIn("DO_NOT_COPY_PREVIOUS_ANSWER", str(provider.await_args.args[3]))

    def test_explicit_new_model_overrides_old_selector_and_section(self):
        plan = rule_plan("ResNet-18 的输出是什么？", self.retriever, "cnn-1d", "cnn-1d:io")
        self.assertEqual(plan["model_id"], "resnet-18")
        self.assertIsNone(plan["section_id"])
        provider = AsyncMock(return_value=response("ResNet-18 的解释。[1]"))
        with patch("backend.main.provider_post", provider):
            result = self.chat(question="ResNet-18 的输出是什么？", model_id="cnn-1d", section_id="cnn-1d:io").json()
        self.assertEqual(result["model_id"], "resnet-18")
        self.assertTrue(all(item["model_id"] == "resnet-18" for item in result["evidence"]))

    def test_invalid_plan_and_timeout_fall_back_without_new_objects(self):
        base = rule_plan("1D CNN 和 ResNet-18 的区别？", self.retriever)
        for draft in [
            {"query": "改问 CLIP", "queries": [{"model_id": "clip", "query": "CLIP"}]},
            {"query": "比较", "queries": []},
            {"query": "比较", "queries": [{"model_id": "cnn-1d", "query": "1D CNN"}] * 4},
        ]:
            with self.subTest(draft=draft), self.assertRaises(ValueError):
                validate_draft(json.dumps(draft), base, self.retriever)
        provider = AsyncMock(return_value=response("不是 JSON"))
        with patch("backend.rag.planning.provider_post", provider):
            plan = asyncio.run(plan_question(base["original_query"], self.retriever))
        self.assertEqual(plan["method"], "rules_fallback")
        bounded = rule_plan("1D CNN、ResNet-18、LSTM 和 Transformer 的区别？", self.retriever)
        self.assertEqual(len(bounded["queries"]), 3)
        self.assertIn("Transformer", " ".join(bounded["warnings"]))
        self.assertEqual(set(plan["model_ids"]), {"cnn-1d", "resnet-18"})

        async def slow(*args):
            await asyncio.sleep(1)
        with patch("backend.rag.planning.provider_post", slow), patch("backend.rag.planning.PLANNING_TIMEOUT", .01):
            plan = asyncio.run(plan_question(base["original_query"], self.retriever))
        self.assertEqual(plan["method"], "rules_fallback")

    def test_local_only_scope_or_history_never_reaches_planner(self):
        provider = AsyncMock()
        previous = [{"role": "assistant", "content": "private text", "metadata": {"model_id": "resnet-18", "evidence": [{"cloud_allowed": False}]}}]
        with patch("backend.rag.planning.provider_post", provider):
            plan = asyncio.run(plan_question("它的结构如何？", self.retriever, previous=previous))
            self.assertEqual(plan["method"], "local_rules")
            self.retriever.chunks[0] = {**self.retriever.chunks[0], "model_id": "resnet-18", "cloud_allowed": False}
            plan = asyncio.run(plan_question("ResNet-18 的结构如何？", self.retriever))
            self.assertEqual(plan["method"], "local_rules")
        provider.assert_not_awaited()

    def test_ambiguous_followup_after_comparison_requires_named_object(self):
        previous = [{"role": "assistant", "content": "对比结果", "metadata": {"model_id": None, "plan": {"model_ids": ["cnn-1d", "resnet-18"]}}}]
        provider = AsyncMock()
        with patch("backend.rag.planning.provider_post", provider):
            plan = asyncio.run(plan_question("它去掉分类头之后呢？", self.retriever, previous=previous))
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual(plan["method"], "clarification")
        self.assertEqual(result["steps"], [])
        provider.assert_not_awaited()
        clarified = rule_plan("ResNet-18", self.retriever, previous=[{"role": "user", "content": "它去掉分类头之后呢？", "metadata": {}}, {"role": "assistant", "content": plan["clarification"], "metadata": {"plan": plan}}])
        self.assertEqual(clarified["model_id"], "resnet-18")
        self.assertIn("去掉分类头", clarified["normalized_query"])
        self.assertIsNone(clarified["clarification"])

    def test_comparison_covers_each_model_and_never_calls_it_graph_hops(self):
        plan = rule_plan("1D CNN 和 ResNet-18 输入输出有什么不同？", self.retriever)
        result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual({item["model_id"] for item in result["evidence"]}, {"cnn-1d", "resnet-18"})
        self.assertEqual(len(result["steps"]), 2)
        self.assertTrue(all(step["kind"] == "comparison" and step["references"] for step in result["steps"]))
        self.assertEqual(result["paths"], [])
        self.assertLessEqual(len(result["evidence"]), 5)
        self.assertEqual(len({item["section_id"] for item in result["evidence"]}), len(result["evidence"]))

    def test_missing_model_does_not_borrow_other_models_evidence(self):
        fixture = without_model_material(self.retriever, 'autoencoder')
        plan = rule_plan("1D CNN 和 Autoencoder 的输入输出区别？", fixture)
        result = asyncio.run(execute_plan(plan, fixture))
        missing = next(step for step in result["steps"] if step["model_id"] == "autoencoder")
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["references"], [])
        self.assertTrue(all(item["model_id"] == "cnn-1d" for item in result["evidence"]))
        self.assertIn("Autoencoder", " ".join(result["warnings"]))

    def test_sequential_graph_queries_depend_on_previous_body_and_keep_proofs(self):
        question = "ResNet-18 与 BasicBlock、残差连接有什么联系？"
        plan = rule_plan(question, self.retriever)
        spy = AsyncMock(wraps=self.retriever.search)
        with patch.object(self.retriever, "search", spy):
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual(spy.await_count, 3)
        self.assertIn("BasicBlock", spy.await_args_list[1].args[0])
        self.assertIn("残差相加", spy.await_args_list[2].args[0])
        self.assertEqual([step["depends_on"] for step in result["steps"]], [None, "s1", "s2"])
        self.assertEqual(max(path["hops"] for path in result["paths"]), 2)
        for step in result["steps"][1:]:
            proof = step["anchor"]["proofs"][0]
            prior = next(s for s in result["steps"] if s["id"] == step["depends_on"])
            self.assertIn(proof["section_id"], prior["evidence_ids"])
            self.assertIsNotNone(proof["reference"])
            self.assertTrue(step["references"])
        self.assertLessEqual(len(result["evidence"]), 5)

    def test_unmentioned_graph_node_cannot_trigger_further_retrieval(self):
        self.retriever.relations = [{"id": "test-ghost-edge", "source": "resnet-18", "target": "ghost", "source_name": "ResNet-18", "target_name": "未在正文出现的机制", "source_ids": ["S08"], "relation": "使用模块", "locator": "test-only"}]
        result = asyncio.run(execute_plan(rule_plan("ResNet-18 的模块联系？", self.retriever), self.retriever))
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["paths"], [])

    def test_comparison_batches_embeddings_and_reranks_once(self):
        self.retriever = Retriever()
        dimension = 8
        self.retriever.vectors = np.full((len(self.retriever.chunks), dimension), 1 / np.sqrt(dimension), dtype=np.float32)
        self.retriever.vector_mask = np.ones(len(self.retriever.chunks), dtype=bool)
        embedding = AsyncMock(return_value=np.full((2, dimension), 1 / np.sqrt(dimension), dtype=np.float32))

        async def rerank(base, key, endpoint, body):
            self.assertEqual(endpoint, "rerank")
            return {"results": [{"index": index} for index in range(len(body["documents"]))]}

        ranking = AsyncMock(side_effect=rerank)
        with patch.object(self.retriever, "embed", embedding), patch("backend.infra.config.RERANK_KEY", "test-placeholder"), patch("backend.rag.pipeline.provider_post", ranking):
            result = asyncio.run(execute_plan(rule_plan("1D CNN 和 ResNet-18 输入输出区别？", self.retriever), self.retriever))
        self.assertEqual(embedding.await_count, 1)
        self.assertEqual(len(embedding.await_args.args[0]), 2)
        self.assertEqual(ranking.await_count, 1)
        self.assertTrue(result["reranked"])
        self.assertEqual({item["model_id"] for item in result["evidence"]}, {"cnn-1d", "resnet-18"})

    def test_trace_is_saved_with_the_same_citation_numbers_as_answer(self):
        planning = AsyncMock(return_value=plan_response("1D CNN 和 ResNet-18 的输入输出区别？", [("cnn-1d", "1D CNN 输入输出"), ("resnet-18", "ResNet-18 输入输出")]))
        generation = AsyncMock(return_value=response("两个模型的说明。[1][2]"))
        with patch("backend.rag.planning.provider_post", planning), patch("backend.main.provider_post", generation):
            result = self.chat(question="1D CNN 和 ResNet-18 的输入输出区别？").json()
        self.assertEqual(result["plan"]["method"], "llm")
        self.assertTrue(result["generated"])
        saved = history(result["session_id"])[-1]["metadata"]
        self.assertEqual(saved["retrieval_steps"], result["retrieval_steps"])
        for step in result["retrieval_steps"]:
            for ref in step["references"]:
                self.assertEqual(ref["title"], result["evidence"][ref["number"] - 1]["title"])
        sent = generation.await_args.args[3]["messages"][-1]["content"]
        # 执行日志仍保存给用户查看，但不再混入模型的事实证据。
        self.assertNotIn("检索步骤与缺失项", sent)
        self.assertEqual(json.loads(sent)['用户当前原问题'], '1D CNN 和 ResNet-18 的输入输出区别？')
