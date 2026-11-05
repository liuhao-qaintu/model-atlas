"""批量导入复用真实Cookie协议；请求只进入MockTransport。"""
import json
import unittest
from unittest.mock import patch
import httpx
from tools.api_client import app_client


class ApiClientTests(unittest.TestCase):
    def test_login_cookie_and_csrf_are_carried_to_document_write(self):
        requests=[]
        def handle(request):
            requests.append(request)
            if request.url.path=='/api/auth/login':
                self.assertEqual(json.loads(request.content),{'username':'fixture','password':'TEST_ONLY_password'})
                return httpx.Response(200,json={'username':'fixture'},headers=[('set-cookie','atlas_session=fixture; Path=/; HttpOnly'),('set-cookie','atlas_csrf=fixture-csrf; Path=/')])
            self.assertEqual(request.url.path,'/api/documents/index')
            self.assertIn('atlas_session=fixture',request.headers['cookie'])
            self.assertEqual(request.headers['x-csrf-token'],'fixture-csrf')
            return httpx.Response(200,json={'ok':True})
        original=httpx.Client
        with patch('tools.api_client.httpx.Client',side_effect=lambda **options:original(transport=httpx.MockTransport(handle),**options)):
            with app_client(username='fixture',password='TEST_ONLY_password') as client:
                self.assertTrue(client.post('/documents/index',json={}).json()['ok'])
        self.assertEqual(len(requests),2)

    def test_failed_login_does_not_continue_to_import(self):
        original=httpx.Client
        transport=httpx.MockTransport(lambda request:httpx.Response(401,json={'detail':'用户名或密码错误'}))
        with patch('tools.api_client.httpx.Client',side_effect=lambda **options:original(transport=transport,**options)):
            with self.assertRaises(httpx.HTTPStatusError):
                with app_client(username='fixture',password='TEST_ONLY_wrong'):
                    self.fail('登录失败后不应进入导入')

    def test_missing_csrf_cookie_rejects_incomplete_login(self):
        original=httpx.Client
        transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'username':'fixture'}))
        with patch('tools.api_client.httpx.Client',side_effect=lambda **options:original(transport=transport,**options)):
            with self.assertRaisesRegex(RuntimeError,'校验凭据'):
                with app_client(username='fixture',password='TEST_ONLY_password'):
                    self.fail('凭据不完整时不应进入导入')
