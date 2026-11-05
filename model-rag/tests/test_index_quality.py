"""验证清理范围、完整来源保留和源码静态窗口；不用模型分数自证质量。"""
import unittest
from types import SimpleNamespace

from backend.rag.index_quality import bounded_windows, exclusion_reason, python_units
from backend.infra.ingestion import split_text


class IndexQualityTests(unittest.TestCase):
    def test_only_metadata_is_excluded_and_short_facts_are_kept(self):
        header = {'content': '来源：https://example.test/code\n\n固定版本：v1。仅整理原文。'}
        self.assertEqual(exclusion_reason(header, 'official_code'), 'source_metadata_only')
        self.assertEqual(exclusion_reason({'content': '源码定位：https://example.test#L1\n\n以下签名和执行代码来自所列原文范围；不是独立模块。'}, 'official_code'), 'source_metadata_only')
        self.assertIsNone(exclusion_reason(header, None))
        for text in ['Input: [B,C,L]', 'Return the output tensor.', '来源：https://example.test\n\nThe mask prevents future-token attention.']:
            self.assertIsNone(exclusion_reason({'content': text}, 'official_code'))

    def test_legacy_orphan_requires_original_cut_boundary_and_page_footer(self):
        previous = {'content': 'a' * 1798 + 'co'}
        orphan = {'content': 'mputed as a weighted sum\n3', 'page': 3, 'part': 2}
        self.assertEqual(exclusion_reason(orphan, 'paper', previous, True), 'legacy_pdf_split_tail')
        for changed, prior, last in [({**orphan, 'content': 'E = mc^2'}, previous, True), (orphan, {'content': '完整段落。'}, True), (orphan, previous, False)]:
            self.assertIsNone(exclusion_reason(changed, 'paper', prior, last))

    def test_new_prose_never_leaves_a_tiny_tail_or_cuts_ordinary_words(self):
        text = ' '.join(['attention'] * 200)
        pieces = split_text(text, 1800)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(120 <= len(p) <= 1800 for p in pieces))
        self.assertEqual(' '.join(pieces), text)
        self.assertTrue(all(p.endswith('attention') for p in pieces))

    def test_large_python_is_static_and_preserves_methods_and_class_conditions(self):
        code = "class Model:\n    layout = 'BCL'\n    def forward(self, x):\n        return x\n\nraise RuntimeError('must never execute')"
        units = python_units('来源：官方源码\n\n```python\n' + code + '\n```')
        self.assertTrue(any(u['symbol'] == 'Model.forward' and 'return x' in u['content'] for u in units))
        self.assertTrue(any(u['symbol'] == 'Model' and "layout = 'BCL'" in u['content'] for u in units))
        self.assertTrue(any("must never execute" in u['content'] for u in units))

    def test_long_lines_and_incomplete_code_are_bounded_without_execution(self):
        for code in ['x = ' + '1' * 5000, 'def broken(:\n' + 'partial\n' * 700]:
            units = python_units('```python\n' + code + '\n```')
            self.assertTrue(units)
            self.assertTrue(all(len(u['content']) <= 1800 for u in units))
        self.assertIsNone(python_units('普通解释和$$公式$$'))
        self.assertEqual(bounded_windows('short'), ['short'])

    def test_evaluation_version_includes_index_policy_not_just_raw_text(self):
        from backend.api.evaluation import corpus_fingerprint
        source = [{'id': 's', 'content': 'same original'}]
        before = SimpleNamespace(sections=source, index_policy='legacy', index_exclusions=[])
        after = SimpleNamespace(sections=source, index_policy='quality', index_exclusions=[{'section_id': 's', 'reason': 'metadata'}])
        self.assertNotEqual(corpus_fingerprint(before), corpus_fingerprint(after))

    def test_versioned_official_request_remains_strict_and_inherits(self):
        from backend.rag.source_scope import resolve_source_scope, source_filters
        for question in ['只依据已入库的PyTorch 2.8.0官方Conv1d接口', '仅看固定版本的Torchvision官方源码']:
            scope = resolve_source_scope(question, ['cnn-1d'])
            self.assertTrue(scope['strict'])
            self.assertEqual(source_filters(scope)['source_kinds'], ['official_code'])
        self.assertFalse(resolve_source_scope('PyTorch资料和教学例都可以参考', ['cnn-1d'])['strict'])


if __name__ == '__main__': unittest.main()
