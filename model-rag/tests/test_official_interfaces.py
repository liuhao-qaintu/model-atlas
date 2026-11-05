"""官方源码整理保留接口事实与原样执行语句；动态表达式不能执行。"""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('interface_preparer', ROOT / 'tools/prepare_official_interfaces.py')
preparer = importlib.util.module_from_spec(spec); spec.loader.exec_module(preparer)


class OfficialInterfaceTests(unittest.TestCase):
    def test_static_string_format_is_only_applied_where_upstream_formats_it(self):
        node = ast.parse("doc = r'\\frac{x}{y}' + '{note}'.format(**notes)").body[0].value
        self.assertEqual(preparer.literal_text(node, {'note': 'shape condition'}), r'\frac{x}{y}shape condition')
        with self.assertRaises(ValueError):
            preparer.literal_text(ast.parse("__import__('os').system('must_not_run')", mode='eval').body)

    def test_downloaded_conv1d_keeps_shape_groups_and_actual_forward(self):
        recipe = preparer.SPECS[0]; path = preparer.ARCHIVE / recipe['filename']
        if not path.exists(): self.skipTest('固定官方文件尚未下载')
        text = preparer.prepare(recipe, path.read_bytes()).decode('utf-8')
        for expected in ['N` is a batch size', 'must both be divisible', 'padding=\'same\'', 'Conv1d._conv_forward', 'F.conv1d(', 'self.groups']:
            self.assertIn(expected, text)
        self.assertNotIn('{groups_note}', text)

    def test_transformer_documents_and_forward_remain_separate_and_traceable(self):
        recipe = preparer.SPECS[1]; path = preparer.ARCHIVE / recipe['filename']
        if not path.exists(): self.skipTest('固定官方文件尚未下载')
        text = preparer.prepare(recipe, path.read_bytes()).decode('utf-8')
        for expected in ['Transformer.forward · 参数、形状与返回值', 'Transformer.forward · 执行代码', 'True', 'scaled_dot_product_attention', 'self.encoder(', 'self.decoder(', '#L']:
            self.assertIn(expected, text)


if __name__ == '__main__': unittest.main()
