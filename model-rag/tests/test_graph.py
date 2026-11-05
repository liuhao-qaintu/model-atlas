"""验证浏览方向、范围与依据完整性；不调用云端或评估模型准确率。"""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient

from backend import main
from backend.rag.graph import build_graph, proven_edges
from knowledge_fixtures import without_model_material


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.fixture = SimpleNamespace(**{field: copy.deepcopy(getattr(main.retriever, field))
                                         for field in ("models", "sections", "sources", "relations")})

    def test_every_relation_has_existing_quote_and_clickable_reader(self):
        result = self.client.get("/api/graph").json()
        self.assertEqual(sum(edge.get('origin') is None for edge in result['edges']), 99)
        self.assertEqual(result["stats"]["excluded_edges"], 0)
        sections = {section["id"]: section for section in self.fixture.sections}
        for edge in result["edges"]:
            for proof in edge["evidence"]:
                self.assertIn(proof["quote"], sections[proof["section_id"]]["content"])
                self.assertTrue(proof["reader_url"].startswith(("/models/", "/documents/")))
                if proof['reader_url'].startswith('/documents/'):
                    document_id = proof['reader_url'].split('/documents/')[1].split('?')[0]
                    original = self.client.get(f'/api/documents/{document_id}')
                    self.assertEqual(original.status_code, 200)
                    self.assertTrue(any(section['id'] == proof['section_id'] and proof['quote'] in section['content'] for section in original.json()['sections']))
            self.assertEqual({source["id"] for source in edge["sources"]}, set(edge["source_ids"]))

    def test_missing_or_stale_or_wrong_owner_proof_is_excluded(self):
        for mutation in ("missing_source", "stale_quote", "wrong_owner", "empty_proof"):
            with self.subTest(mutation=mutation):
                fixture = copy.deepcopy(self.fixture)
                edge = fixture.relations[0]
                if mutation == "missing_source":
                    edge["source_ids"] = ["NONEXISTENT"]
                elif mutation == "stale_quote":
                    edge["evidence"][0]["quote"] = "不存在的原文"
                elif mutation == "wrong_owner":
                    edge["evidence"][0]["section_id"] = "cnn-1d:code"
                else:
                    edge["evidence"] = []
                result = build_graph(fixture)
                self.assertNotIn(edge["id"], {item["id"] for item in result["edges"]})
                self.assertEqual(result["stats"]["excluded_edges"], 1)

    def test_unsafe_source_is_not_a_clickable_evidence(self):
        self.fixture.sources["S08"]["url"] = "javascript:alert(1)"
        self.assertFalse(any("S08" in edge["source_ids"] for edge in proven_edges(self.fixture)))

    def test_one_and_two_hops_keep_stored_direction_and_reverse_steps(self):
        one = build_graph(self.fixture, "conv2d", 1)
        two = build_graph(self.fixture, "conv2d", 2)
        self.assertNotIn("resnet-18", {node["id"] for node in one["nodes"]})
        self.assertIn("resnet-18", {node["id"] for node in two["nodes"]})
        edge = next(item for item in two["edges"] if item["id"] == "basicblock-conv2d")
        self.assertEqual((edge["source"], edge["target"]), ("basicblock", "conv2d"))
        path = next(item for item in two["paths"] if item["target"] == "resnet-18")
        self.assertTrue(all(step["reverse"] for step in path["steps"]))
        self.assertEqual(path["hops"], 2)
        self.assertEqual(path["steps"][0]["from"], "conv2d")
        self.assertEqual(path["steps"][1]["from"], path["steps"][0]["to"])

    def test_cycle_does_not_repeat_nodes_in_shortest_paths(self):
        extra = copy.deepcopy(self.fixture.relations[0])
        extra.update(id="test-cycle", source="conv2d", source_name="二维卷积", target="resnet-18", target_name="ResNet-18")
        self.fixture.relations.append(extra)
        result = build_graph(self.fixture, "resnet-18", 2)
        self.assertEqual(len({path["target"] for path in result["paths"]}), len(result["paths"]))
        for path in result["paths"]:
            visited = ["resnet-18"] + [step["to"] for step in path["steps"]]
            self.assertEqual(len(visited), len(set(visited)))
            self.assertLessEqual(path["hops"], 2)

    def test_shared_structure_requires_different_model_owners(self):
        result = build_graph(self.fixture, "resnet-18")
        self.assertEqual({item["node_id"] for item in result["shared"]}, {"global-average-pooling", "linear-head", "conv2d", "residual"})
        self.assertTrue(all(len(item["implementations"]) >= 2 for item in result["shared"]))
        self.fixture.relations = [edge for edge in self.fixture.relations if edge['owner_model_id'] == 'cnn-1d']
        self.assertEqual(build_graph(self.fixture)["shared"], [])

    def test_pending_model_is_visible_without_borrowing_relations(self):
        with patch('backend.main.retriever', without_model_material(main.retriever, 'autoencoder')):
            result = self.client.get("/api/graph?root=autoencoder").json()
        self.assertEqual([node["id"] for node in result["nodes"]], ["autoencoder"])
        self.assertFalse(result["nodes"][0]["ready"])
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["shared"], [])

    def test_relation_filter_applies_before_neighbor_exploration(self):
        result = build_graph(self.fixture, "resnet-18", kind="共享结构")
        self.assertEqual({edge["kind"] for edge in result["edges"]}, {"共享结构"})
        self.assertNotIn("basicblock", {node["id"] for node in result["nodes"]})

    def test_invalid_parameters_are_rejected_without_cloud_calls(self):
        provider = AsyncMock()
        with patch("backend.main.provider_post", provider):
            self.assertEqual(self.client.get("/api/graph?root=unknown").status_code, 404)
            for query in ("hops=0", "hops=3", "hops=bad", "kind=unknown"):
                self.assertEqual(self.client.get(f"/api/graph?{query}").status_code, 422)
            self.assertEqual(self.client.get("/api/graph?root=resnet-18&hops=2").status_code, 200)
        provider.assert_not_awaited()
