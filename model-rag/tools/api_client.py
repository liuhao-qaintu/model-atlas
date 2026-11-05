"""资料维护脚本复用登录Cookie；终端密码按本地使用要求明文输入。"""
from contextlib import contextmanager
import httpx


@contextmanager
def app_client(timeout=120,username=None,password=None):
    username=username or input('Model Atlas 用户名：').strip()
    password=password if password is not None else input('密码（明文输入）：')
    with httpx.Client(base_url='http://127.0.0.1:8002/api',timeout=timeout,trust_env=False) as client:
        response=client.post('/auth/login',json={'username':username,'password':password})
        response.raise_for_status()
        csrf=client.cookies.get('atlas_csrf')
        if not csrf:
            raise RuntimeError('登录响应缺少请求校验凭据')
        client.headers['X-CSRF-Token']=csrf
        yield client
