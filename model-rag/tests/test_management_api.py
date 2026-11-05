"""在隔离数据库和上传目录中验证新增模型到图谱的完整接口链路。"""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from auth_fixtures import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend import main


class ManagementApiTests(unittest.TestCase):
    def test_custom_model_upload_relation_and_refresh_preserve_identity(self):
        engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        previous = main.retriever
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for target in ['backend.infra.database.engine', 'backend.api.management.engine', 'backend.infra.ingestion.engine']:
                stack.enter_context(patch(target, engine))
            stack.enter_context(patch('backend.infra.config.DATA_ROOT', Path(temp)))
            stack.enter_context(patch('backend.infra.ingestion.UPLOAD_ROOT', Path(temp) / 'uploads'))
            with TestClient(main.app) as client:
                model = client.post('/api/models', json={'name': 'Isolation Model', 'aliases': ['isolated-model']})
                self.assertEqual(model.status_code, 200); identity = model.json()['id']
                quote = '本模型采用 ResNet-18 作为图像编码骨干，并读取其特征表示。'
                body = '# 隔离验收资料\n\n## 结构\n\n' + quote
                preview = client.post('/api/documents/preview', files={'file': ('isolated.md', body.encode('utf-8'))}, data={'model_id': identity})
                self.assertEqual(preview.status_code, 200); doc = preview.json()['id']
                imported = client.post(f'/api/documents/{doc}/import', json={'title': '隔离验收资料', 'cloud_allowed': False})
                self.assertEqual(imported.status_code, 200)
                detail = client.get('/api/models/' + identity).json(); section = next(s for s in detail['sections'] if quote in s['content'])
                self.assertTrue(detail['searchable']); self.assertEqual(detail['document_count'], 1)
                created = client.post('/api/relations', json={'source': identity, 'target': 'resnet-18', 'relation': '采用架构', 'section_id': section['id'], 'quote': quote})
                self.assertEqual(created.status_code, 200)
                graph = client.get('/api/graph', params={'root': identity, 'hops': 1}).json()
                self.assertTrue(any(e['id'] == created.json()['id'] for e in graph['edges']))
                status = client.patch('/api/relations/' + created.json()['id'], json={'status': 'rejected'})
                self.assertEqual(status.status_code, 200)
                self.assertFalse(client.get('/api/graph', params={'root': identity, 'hops': 1}).json()['edges'])
                self.assertEqual(len(client.get('/api/management').json()['relations']), 1)
                self.assertEqual(client.patch('/api/relations/' + created.json()['id'], json={'status': 'accepted'}).status_code, 200)
                updated = client.patch('/api/models/' + identity, json={'name': 'Isolation v2'})
                self.assertEqual(updated.status_code, 200); self.assertIn('Isolation Model', updated.json()['aliases'])
                graph = client.get('/api/graph', params={'root': identity, 'hops': 1}).json()
                self.assertEqual(next(e for e in graph['edges'] if e['id'] == created.json()['id'])['source_name'], 'Isolation v2')
                refresh = client.post('/api/knowledge/refresh', json={})
                self.assertEqual(refresh.status_code, 200)
                self.assertEqual(client.get('/api/models/' + identity).json()['document_count'], 1)
            main.retriever = previous
        engine.dispose()


if __name__ == '__main__': unittest.main()
