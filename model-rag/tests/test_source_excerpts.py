import ast
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.import_public_sources import code_excerpt, constructor_ranges
from backend.infra import public_sources
from backend.rag.pipeline import retain_evidence


class SourceExcerptTests(unittest.TestCase):
    def fixture(self):
        source = '''class Example:
    def __init__(self, representation_size=None):
        self.before = "unrelated"
        heads_layers: dict = {}
        if representation_size is None:
            heads_layers["head"] = "linear"
        else:
            heads_layers["pre_logits"] = "linear"
            heads_layers["act"] = "tanh"
        self.heads = nn.Sequential(heads_layers)
        if hasattr(self.heads, "pre_logits"):
            self.initialized = True
'''
        return source, ast.parse(source).body[0].body[0]

    def test_excerpts_preserve_both_branches_and_exact_source_line_links(self):
        source, _ = self.fixture()
        entry = {'title': 'checked', 'version': 'fixed', 'source_url': 'https://example.test/fixed.py',
                 'symbols': ['Example.__init__'], 'blocks': ['signature', 'heads', 'head_initialization']}
        rendered = code_excerpt(entry, source.encode()).decode()
        self.assertIn('fixed.py#L4-L10', rendered)
        self.assertIn('if representation_size is None:', rendered)
        self.assertIn('else:', rendered)
        self.assertIn('heads_layers["act"] = "tanh"', rendered)
        self.assertNotIn('self.before', rendered)
        self.assertIn('不是独立可执行函数', rendered)

    def test_unknown_missing_and_duplicate_markers_fail_closed(self):
        source, node = self.fixture()
        with self.assertRaises(ValueError):
            constructor_ranges(node, source.splitlines(), ['unknown'])
        with self.assertRaises(ValueError):
            constructor_ranges(node, source.replace('heads_layers: dict', 'other: dict').splitlines(), ['heads'])
        with self.assertRaises(ValueError):
            constructor_ranges(node, source.replace('self.before = "unrelated"', 'heads_layers: dict = {}').splitlines(), ['heads'])

    def test_definition_dependency_requires_verified_documents_and_actual_call(self):
        entries = [{'key': 'call', 'document_id': 'call', 'model': 'vit', 'type': 'official_code', 'import_sha256': 'ok', 'source_url': 'https://example.test/source'},
                   {'key': 'definition', 'document_id': 'definition', 'model': 'vit', 'type': 'official_code', 'import_sha256': 'ok', 'source_url': 'https://example.test/source',
                    'definition_dependency': {'source_key': 'call', 'call_quote': 'self.heads(x)', 'definition_quote': 'heads_layers:'}}]
        documents = [{'id':e['document_id'],'model_id':'vit','sha256':'ok','source_url':e['source_url']} for e in entries]
        trigger = {'document_id':'call','section_id':'call-section','model_id':'vit','content':'return self.heads(x)','reader_url':'/documents/call'}
        chunk = {'document_id':'definition','section_id':'definition-section','model_id':'vit','parent_content':'heads_layers: dict = {}','source_ids':['Ddefinition'],'reader_url':'/documents/definition','title':'definition','cloud_allowed':True}
        retriever = SimpleNamespace(documents=documents,chunks=[chunk],sources={'Ddefinition': {'id':'Ddefinition'}})
        def result():
            return {'evidence':[trigger.copy(),{'section_id':'other','model_id':'vit','content':'other'}],'channels':{}}
        with tempfile.TemporaryDirectory() as folder, patch.object(public_sources,'DATA_ROOT',Path(folder)):
            Path(folder,'public-source-manifest.json').write_text(json.dumps({'entries':entries}),encoding='utf-8')
            expanded = public_sources.expand_definitions(retriever,result(),2)
            self.assertEqual(len(expanded['evidence']),2)
            self.assertEqual(expanded['evidence'][-1]['section_id'],'definition-section')
            self.assertEqual(expanded['definition_expansions'][0]['call_quote'],'self.heads(x)')
            for key,bad in [('sha256','stale'),('model_id','clip'),('source_url','https://other.test')]:
                old=documents[1][key]; documents[1][key]=bad
                self.assertNotIn('definition_expansions',public_sources.expand_definitions(retriever,result(),2))
                documents[1][key]=old
            trigger['content']='unrelated call'
            self.assertNotIn('definition_expansions',public_sources.expand_definitions(retriever,result(),2))

    def test_retention_keeps_call_definition_and_both_official_models_within_five(self):
        candidates=[{'section_id':name,'model_id':'clip' if name.startswith('clip') else 'vit','document_id':name,'score':0.1}
                    for name in ['vit-overview','clip-overview','vit-call','clip-official','vit-definition','vit-other']]
        steps=[{'evidence_ids':[c['section_id'] for c in candidates], 'definition_expansions':[{'trigger_section_id':'vit-call','definition_section_id':'vit-definition'}]}]
        kept=retain_evidence(candidates,steps,['vit','clip'],{'vit-call':'official_code','vit-definition':'official_code','clip-official':'official_code'},True)
        self.assertLessEqual(len(kept),5)
        self.assertTrue({'vit-call','vit-definition','clip-official'} <= {c['section_id'] for c in kept})
