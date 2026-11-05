"""交付包检查用独立临时样本，防止打包配置或未登记文件。"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from export_project import DOCS, TOOLS, build_archive


class ExportProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.root = self.base / 'model-rag'; self.content = self.base / '模型知识库资料'
        for name in ['README.md', 'requirements.txt', '.gitignore', 'start-backend.ps1', 'start-frontend.ps1',
                     'frontend/package.json', 'frontend/package-lock.json', 'frontend/index.html', 'frontend/vite.config.js',
                     'backend/__init__.py', 'frontend/src/main.js', 'tests/test_sample.py',
                     *['tools/' + name for name in TOOLS], *['docs/' + name for name in DOCS]]:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('sample', encoding='utf-8')
        (self.root / 'backend/infra').mkdir(parents=True, exist_ok=True)
        (self.root / 'backend/infra/knowledge.py').write_text("TOPICS = [{'file': '01_Test.md'}]", encoding='utf-8')
        for name in ['模型卡/01_Test.md', '首批资料清单.md', '公开原始资料清单.md', '公开原始资料/paper.pdf']:
            path = self.content / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'public sample')
        self.source = self.content / '公开原始资料/paper.pdf'
        entry = {'source_path': str(self.source), 'import_path': str(self.source),
                 'source_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                 'import_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}
        (self.root / 'data').mkdir()
        (self.root / 'data/public-source-manifest.json').write_text(json.dumps({'entries': [entry]}), encoding='utf-8')
        (self.root / 'data/figure-manifest.json').write_text(json.dumps({'figures': []}), encoding='utf-8')
        (self.root / 'data/evaluation-cases.json').write_text('[]', encoding='utf-8')
        (self.root / 'data/evaluation-challenge-cases.json').write_text('[]', encoding='utf-8')
        (self.root / '.env').write_text('PRIVATE_VALUE_NOT_EXPORTED', encoding='utf-8')
        (self.root / 'docs').mkdir(exist_ok=True)
        (self.root / 'docs/private-change-record.md').write_text('PRIVATE_VALUE_NOT_EXPORTED',encoding='utf-8')
        (self.root / '产品整改修改记录.md').write_text('PRIVATE_VALUE_NOT_EXPORTED',encoding='utf-8')
        (self.root / 'data/private-chat.json').write_text('PRIVATE_VALUE_NOT_EXPORTED', encoding='utf-8')
        self.output = self.base / 'bundle.zip'

    def tearDown(self): self.temp.cleanup()

    def test_export_is_portable_and_only_contains_registered_material(self):
        result = build_archive(self.root, self.content, self.output)
        with ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertNotIn('ModelAtlas/model-rag/.env', names)
            self.assertNotIn('ModelAtlas/model-rag/data/private-chat.json', names)
            self.assertNotIn('ModelAtlas/model-rag/docs/private-change-record.md',names)
            self.assertNotIn('ModelAtlas/model-rag/产品整改修改记录.md',names)
            self.assertIn('ModelAtlas/model-rag/tools/manage_accounts.py',names)
            self.assertIn(b'AUTH_SECRET_KEY=\n',archive.read('ModelAtlas/model-rag/.env.example'))
            self.assertEqual(archive.read('ModelAtlas/model-rag/README.md'), b'sample')
            self.assertIn('ModelAtlas/model-rag/data/evaluation-challenge-cases.json', names)
            self.assertTrue(all(b'PRIVATE_VALUE_NOT_EXPORTED' not in archive.read(name) for name in names))
            manifest = json.loads(archive.read('ModelAtlas/model-rag/data/public-source-manifest.json'))
            self.assertEqual(manifest['entries'][0]['source_path'], '../模型知识库资料/公开原始资料/paper.pdf')
            for item in result['files']:
                self.assertEqual(hashlib.sha256(archive.read(item['path'])).hexdigest(), item['sha256'])
        with self.assertRaises(FileExistsError): build_archive(self.root, self.content, self.output)

    def test_changed_registered_document_is_rejected_before_writing_archive(self):
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): build_archive(self.root, self.content, self.output)
        self.assertFalse(self.output.exists())

    def test_source_export_works_without_optional_originals_or_figures(self):
        (self.root/'data/public-source-manifest.json').unlink()
        (self.root/'data/figure-manifest.json').unlink()
        result=build_archive(self.root,self.content,self.output)
        self.assertEqual(result['source_documents'],0)
        self.assertEqual(result['original_figures'],0)
        with ZipFile(self.output) as archive:
            self.assertFalse(any(name.endswith(('.pdf','.png')) for name in archive.namelist()))

    def test_source_outside_the_registered_content_directory_is_rejected(self):
        path = self.root / 'data/public-source-manifest.json'; manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['entries'][0]['source_path'] = str(self.root / '.env'); path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError): build_archive(self.root, self.content, self.output)
        self.assertFalse(self.output.exists())
