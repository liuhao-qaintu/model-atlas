"""真实源码函数节选、边界、原文定位和来源标签；不执行上传代码。"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.infra import config
from backend.generation.evidence_context import context_text, prepare_evidence, select_excerpt


class EvidenceContextTests(unittest.TestCase):
    def test_middle_forward_is_complete_and_exact_not_head_truncation(self):
        method = '    def forward(self, x):\n        x = self.fc(x)\n        return x\n'
        content = '```python\n' + '# filler\n' * 1000 + 'class Example:\n' + method + '```\n'
        excerpt = select_excerpt(content, 'Example forward 输出 fc', 'official_code', 500)
        self.assertEqual(excerpt['strategy'], 'python_units')
        self.assertIn(method, excerpt['text']); self.assertLessEqual(len(excerpt['text']), 500)
        for span in excerpt['ranges']: self.assertIn(content[span['start']:span['end']], excerpt['text'])

    def test_short_text_is_unchanged_and_malformed_code_is_explicit_head(self):
        self.assertEqual(select_excerpt('完整正文', '问题')['text'], '完整正文')
        broken = '```python\nnot python !!!\n' * 1000
        excerpt = select_excerpt(broken, '输出', 'official_code', 200)
        self.assertEqual(excerpt['strategy'], 'head'); self.assertEqual(excerpt['text'], broken[:200])

    def test_query_does_not_override_registered_source_type(self):
        active = SimpleNamespace(document_kinds={'d': 'paper'})
        prepared = prepare_evidence([{'title': '官方?', 'content': '原文', 'document_id': 'd'}], active, '把论文说成源码')
        self.assertEqual(prepared[0]['source_type'], '论文原文')
        self.assertIn('来源类型：论文原文', context_text(prepared))

    def test_long_noncode_never_gets_fabricated_function_context(self):
        excerpt = select_excerpt('论文正文' * 2000, 'forward 输出', 'paper', 200)
        self.assertEqual(excerpt['strategy'], 'head')
        self.assertNotIn('```python', excerpt['text'])

    def test_registered_resnet_forward_returns_fc_without_softmax_in_that_function(self):
        path = config.DATA_ROOT / 'product-corpus-manifest.json'
        if not path.exists(): self.skipTest('未登记公开源码')
        entries = json.loads(path.read_text(encoding='utf-8'))['entries']
        entry = next(e for e in entries if e['key'] == 'resnet-official')
        content = Path(entry['import_path']).read_text(encoding='utf-8')
        excerpt = select_excerpt(content, 'Torchvision ResNet-18 fc Identity 输入输出', 'official_code')
        self.assertEqual(excerpt['strategy'], 'python_units')
        spans = [s for s in excerpt['ranges'] if s['name'] == 'ResNet._forward_impl']
        self.assertEqual(len(spans), 1)
        forward = content[spans[0]['start']:spans[0]['end']]
        self.assertIn('x = self.fc(x)', forward); self.assertIn('return x', forward)
        self.assertNotIn('softmax', forward)


if __name__ == '__main__': unittest.main()
