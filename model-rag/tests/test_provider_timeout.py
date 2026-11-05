"""长回答的等待窗口不能继续沿用短请求，也不能扩大嵌入和重排等待。"""
import unittest
from unittest.mock import patch

import httpx

from backend.rag.retrieval import provider_post


class ProviderTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_completion_has_bounded_deadline_and_unchanged_body(self):
        received, timeouts = [], []
        async def handle(request):
            received.append(request)
            return httpx.Response(200, json={'choices': [{'message': {'content': '完成'}}]})
        client_type = httpx.AsyncClient
        def client_factory(**kwargs):
            timeouts.append(kwargs['timeout'])
            return client_type(transport=httpx.MockTransport(handle), **kwargs)
        with patch('backend.rag.retrieval.httpx.AsyncClient', side_effect=client_factory):
            await provider_post('https://example.invalid/v1', 'LOCAL_TEST', 'chat/completions', {'max_tokens': 3200, 'messages': []})
            await provider_post('https://example.invalid/v1', 'LOCAL_TEST', 'rerank', {'max_tokens': 3200})
            await provider_post('https://example.invalid/v1', 'LOCAL_TEST', 'chat/completions', {'max_tokens': 800, 'messages': []})
        self.assertEqual(timeouts, [120, 65, 65])
        self.assertIn(b'"max_tokens":3200', received[0].content)
        self.assertEqual(len(received), 3)


if __name__ == '__main__':
    unittest.main()
