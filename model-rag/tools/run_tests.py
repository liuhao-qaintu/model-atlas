"""在临时目录和SQLite中运行回归，禁止访问真实数据库和外部HTTP。"""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import httpx

ROOT=Path(__file__).resolve().parents[1]


async def no_async_http(*args,**kwargs):
    raise httpx.ConnectError('External HTTP is disabled during offline tests')


def no_http(*args,**kwargs):
    raise httpx.ConnectError('External HTTP is disabled during offline tests')


def main():
    with tempfile.TemporaryDirectory(prefix='model-atlas-tests-') as folder:
        target=Path(folder)
        for name in ['evaluation-cases.json','evaluation-challenge-cases.json','answer-evaluation-cases.json',
                     'public-source-manifest.json','product-corpus-manifest.json','interface-source-manifest.json']:
            source=ROOT/'data'/name
            if source.is_file():shutil.copyfile(source,target/name)
        os.environ.update(DATABASE_URL='sqlite://',MODEL_ATLAS_DATA_ROOT=str(target),
                          AUTH_SECRET_KEY='TEST_ONLY_offline_key_not_for_live_services',
                          LLM_API_KEY='TEST_ONLY_OFFLINE',EMBEDDING_API_KEY='TEST_ONLY_OFFLINE',RERANK_API_KEY='TEST_ONLY_OFFLINE')
        os.chdir(ROOT)
        sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
        from backend.infra import figures
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
        with patch.object(figures,'DATA_ROOT',ROOT/'data'), \
             patch.object(httpx.AsyncHTTPTransport,'handle_async_request',no_async_http), \
             patch.object(httpx.HTTPTransport,'handle_request',no_http):
            result=unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
