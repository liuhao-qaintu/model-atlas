"""验证流式正文、完整保存、失败回退与取消；外部模型均使用替身。"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from auth_fixtures import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend import main

from backend.infra import streaming
from knowledge_fixtures import without_model_material
from backend.infra.database import ChatMessage, ChatSession, engine, history


def events(response):
    result = []
    for frame in response.text.split("\n\n"):
        if not frame.startswith("event:"):
            continue
        name, payload = frame.split("\ndata: ", 1)
        result.append((name[7:], json.loads(payload)))
    return result


class StreamingTests(unittest.TestCase):
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
        for target, value in [("backend.infra.config.RERANK_KEY", ""),
                              ("backend.rag.planning.provider_post", AsyncMock(side_effect=RuntimeError("test fallback")))]:
            context = patch(target, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(main.retriever, "vectors", None)
        context.start()
        self.addCleanup(context.stop)

    def request(self, **fields):
        sid = str(uuid4())
        self.sessions.append(sid)
        return self.client.post("/api/chat/stream", json={"session_id": sid, **fields})

    def test_deltas_precede_final_validation_and_complete_history(self):
        async def fake(*args):
            yield {"text": "输入为 [B, C, L]。", "finish_reason": None}
            yield {"text": "[1][99] [虚构入口](/made-up)", "finish_reason": "stop"}

        turn = str(uuid4())
        with patch("backend.main.stream_completion", fake), patch("backend.main.provider_post", AsyncMock()) as regular:
            response = self.request(question="1D CNN 的输入输出是什么？", turn_id=turn)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        parsed = events(response)
        self.assertEqual([item[1]["stage"] for item in parsed if item[0] == "stage"],
                         ["planning", "retrieving", "generating", "saving"])
        self.assertEqual(sum(name == "delta" for name, _ in parsed), 2)
        self.assertEqual(parsed[-1][0], "done")
        result = parsed[-1][1]
        self.assertTrue(result["generated"])
        self.assertNotIn("[99]", result["answer"])
        self.assertNotIn("/made-up", result["answer"])
        saved = history(result["session_id"])
        self.assertEqual([m["role"] for m in saved], ["user", "assistant"])
        self.assertEqual(saved[-1]["content"], result["answer"])
        self.assertEqual(saved[-1]["metadata"]["turn_id"], turn)
        regular.assert_not_awaited()

    def test_partial_generation_failure_replaces_draft_with_explicit_excerpt(self):
        async def fake(*args):
            yield {"text": "UNFINISHED_DRAFT", "finish_reason": None}
            raise RuntimeError("test-only service failure")

        with patch("backend.main.stream_completion", fake):
            result = events(self.request(question="1D CNN 输入输出？"))[-1][1]
        self.assertFalse(result["generated"])
        self.assertNotIn("UNFINISHED_DRAFT", result["answer"])
        self.assertIn("未完成模型生成", " ".join(result["warnings"]))
        self.assertNotIn("UNFINISHED_DRAFT", history(result["session_id"])[-1]["content"])

    def test_strict_total_deadline_closes_provider_and_replaces_partial_draft(self):
        closed = []
        async def waiting(*args):
            try:
                yield {'text': 'UNFINISHED_STRICT_DRAFT', 'finish_reason': None}
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        from backend.rag.planning import rule_plan
        question='只依据PyTorch 2.6.0官方nn.Transformer.forward，返回什么？'
        plan=rule_plan(question,main.retriever,'transformer');plan['duration_ms']=0
        plan['source_scope']={'strict':True,'kind':'implementation','label':'官方接口'}
        evidence={'section_id':'fixture:official','section_slug':'official','model_id':'transformer',
                  'title':'官方函数测试夹具','content':'def forward(self,src,tgt):\n    return output',
                  'cloud_allowed':True,'sources':[]}
        retrieval={'evidence':[evidence],'warnings':[],'paths':[],'steps':[],'duration_ms':0,
                   'stop_reason':'完成','channels':{},'reranked':False,'remote_allowed':True}
        with patch('backend.main.stream_completion', waiting), patch.object(main, 'STRICT_GENERATION_SECONDS', 0.01), \
             patch.object(main,'plan_question',AsyncMock(return_value=plan)),patch.object(main,'execute_plan',AsyncMock(return_value=retrieval)):
            parsed = events(self.request(question=question, model_id='transformer'))
        self.assertEqual(parsed[-1][0], 'done')
        result = parsed[-1][1]
        self.assertFalse(result['generated'])
        self.assertEqual(closed, [True])
        self.assertNotIn('UNFINISHED_STRICT_DRAFT', result['answer'])
        self.assertEqual(result['generation_diagnostics']['rejected_partial_answer'], 'UNFINISHED_STRICT_DRAFT')
        self.assertEqual(result['generation_diagnostics']['error_type'], 'TimeoutError')
        self.assertEqual(history(result['session_id'])[-1]['content'], result['answer'])
        self.assertTrue(any('超时' in warning for warning in result['warnings']))

    def test_local_only_evidence_never_calls_stream_provider(self):
        retrieval = {"evidence": [{"title": "本地章节", "content": "本地正文", "cloud_allowed": False, "sources": []}],
                     "warnings": [], "paths": [], "steps": [], "stop_reason": "仅本地资料", "duration_ms": 0,
                     "channels": {}, "reranked": False, "remote_allowed": False}
        with patch("backend.main.execute_plan", AsyncMock(return_value=retrieval)), \
                patch("backend.main.stream_completion") as provider:
            parsed = events(self.request(question="1D CNN 输入输出？"))
        provider.assert_not_called()
        self.assertFalse(any(name == "delta" for name, _ in parsed))
        self.assertFalse(parsed[-1][1]["generated"])
        self.assertIn("本地正文", parsed[-1][1]["answer"])

    def test_pending_model_has_no_generation_stage(self):
        with patch('backend.main.retriever', without_model_material(main.retriever, 'autoencoder')), patch("backend.main.stream_completion") as provider:
            parsed = events(self.request(question="Autoencoder 输入输出？"))
        provider.assert_not_called()
        self.assertFalse(any(data.get("stage") == "generating" for _, data in parsed))
        self.assertEqual(parsed[-1][1]["evidence"], [])

    def test_validation_remains_http_422_and_errors_are_sanitized(self):
        self.assertEqual(self.request(question="   ").status_code, 422)
        with patch("backend.main.plan_question", AsyncMock(side_effect=RuntimeError("SECRET_MUST_NOT_LEAK"))):
            response = self.request(question="1D CNN 输入输出？")
        parsed = events(response)
        self.assertEqual(parsed[-1][0], "error")
        self.assertNotIn("SECRET_MUST_NOT_LEAK", response.text)

    def test_closing_partial_chat_does_not_save_and_closes_provider(self):
        closed = []

        async def fake(*args):
            try:
                yield {"text": "部分正文", "finish_reason": None}
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        async def run():
            stream = main.chat_events(main.ChatRequest(question="1D CNN 输入输出？"), main.retriever, streaming=True)
            async for name, _ in stream:
                if name == "delta":
                    break
            await stream.aclose()

        with patch("backend.main.stream_completion", fake), patch("backend.main.save_turn") as save:
            asyncio.run(run())
        save.assert_not_called()
        self.assertEqual(closed, [True])

    def test_relay_disconnect_cancels_in_flight_work(self):
        closed = []

        async def run():
            disconnected = False

            async def check():
                return disconnected

            async def source():
                try:
                    yield "stage", {"label": "检索中"}
                    await asyncio.Event().wait()
                finally:
                    closed.append(True)

            relay = streaming.relay_events(source(), check)
            self.assertIn("event: stage", await anext(relay))
            disconnected = True
            with self.assertRaises(StopAsyncIteration):
                await anext(relay)

        asyncio.run(run())
        self.assertEqual(closed, [True])


class ProviderStreamTests(unittest.TestCase):
    def read(self, body, status=200, content_type="text/event-stream"):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(status, content=body.encode(), headers={"content-type": content_type})

        async def run():
            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with patch("backend.infra.streaming.httpx.AsyncClient", return_value=client):
                return [item async for item in streaming.stream_completion("https://example.test/v1", "test-key", {"model": "test", "messages": []})]

        return asyncio.run(run()), requests

    def test_real_protocol_ignores_reasoning_usage_and_keeps_body(self):
        body = ': ping\n\ndata: {"choices":[{"delta":{"reasoning_content":"DO_NOT_DISPLAY"}}]}\n\n'
        body += 'data: {"choices":[{"delta":{"content":"输入输出"},"finish_reason":null}]}\n\n'
        body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        body += 'data: {"choices":[],"usage":{"total_tokens":10}}\n\ndata: [DONE]\n\n'
        items, requests = self.read(body)
        self.assertEqual("".join(item["text"] for item in items), "输入输出")
        self.assertNotIn("DO_NOT_DISPLAY", str(items))
        self.assertTrue(requests[0]["stream"])

    def test_missing_finish_and_invalid_protocol_are_rejected(self):
        for body, kind in [('data: {"choices":[{"delta":{"content":"部分正文"}}]}\n\n', "text/event-stream"),
                           ('data: not-json\n\n', "text/event-stream"), ('{}', "application/json")]:
            with self.subTest(kind=kind, body=body), self.assertRaises(RuntimeError):
                self.read(body, content_type=kind)

    def test_http_error_does_not_expose_response_body(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            self.read("SECRET_MUST_NOT_LEAK", status=401)


if __name__ == "__main__":
    unittest.main()
