import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.infra.public_sources import document_kinds


class ProductSourceTests(unittest.TestCase):
    def test_additional_official_sources_require_matching_digest_and_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {'document_id': 'doc1', 'format': 'code', 'model': 'resnet-18', 'import_sha256': 'verified'}
            (root / 'product-corpus-manifest.json').write_text(json.dumps({'entries': [entry]}), encoding='utf-8')
            doc = {'id': 'doc1', 'model_id': 'resnet-18', 'sha256': 'verified'}
            with patch('backend.infra.public_sources.DATA_ROOT', root):
                self.assertEqual(document_kinds([doc]), {'doc1': 'official_code'})
                self.assertEqual(document_kinds([{**doc, 'sha256': 'changed'}]), {})
                self.assertEqual(document_kinds([{**doc, 'model_id': 'private-copy'}]), {})


if __name__ == '__main__': unittest.main()
