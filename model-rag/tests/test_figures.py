import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auth_fixtures import TestClient
from backend import main
from backend.infra.figures import figure_records, figure_path, answer_figures


class FigureTests(unittest.TestCase):
    def test_all_original_figures_have_verifiable_image_and_source_identity(self):
        figures = figure_records()
        if not figures: self.skipTest('未安装可选论文图示资料')
        self.assertEqual(len(figures), 11)
        with TestClient(main.app) as client:
            for figure in figures:
                image = client.get(figure['image_url'])
                self.assertEqual(image.status_code, 200)
                self.assertEqual(image.headers['content-type'], 'image/png')
                self.assertTrue(image.content.startswith(b'\x89PNG'))
                self.assertEqual(hashlib.sha256(image.content).hexdigest(), figure['sha256'])
                self.assertIn('#page='+str(figure['physical_page']), figure['source_url'])
            self.assertEqual(client.get('/api/figures/missing/image').status_code, 404)
        self.assertEqual([f['model_id'] for f in answer_figures([{'model_id':'llava'}])], ['llava'])

    def test_unregistered_paths_and_changed_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'figures').mkdir()
            record = {'figures':[{'id':'x','file':'../outside.png','sha256':'bad'}]}
            (root/'figure-manifest.json').write_text(json.dumps(record),encoding='utf-8')
            with patch('backend.infra.figures.DATA_ROOT',root):
                with self.assertRaises(LookupError):figure_path('x')
                record['figures'][0]['file'] = 'test.png'
                (root/'figures/test.png').write_bytes(b'changed')
                (root/'figure-manifest.json').write_text(json.dumps(record),encoding='utf-8')
                with self.assertRaises(ValueError):figure_path('x')
