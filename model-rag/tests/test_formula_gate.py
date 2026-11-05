import unittest
from types import SimpleNamespace

from backend.generation.evidence_context import prepare_evidence
from backend.infra.text_quality import formula_soup

# 真实片段（第 4 页，截断）：pypdf 把矩阵和上下标抽成了符号堆。
SOUP = (
    '𝑡!"# 𝑡!"#𝑡! 𝑡!\n'
    '!𝑒!!(#!$%)\t\t𝐵𝜏𝑥𝜏𝑑𝜏\t𝑡!"#\n'
    '\t𝑡!\n≈𝛼!:!\t×\n𝛼%:!\t×\n𝛼&:!\t× 𝛼%:%\t×\n'
    '𝛼&:%\t× 𝛼&:&\t×\n'
    "𝛾' 𝛾! 𝛾% 𝛾&\n"
    'ℳ= 𝛽! 𝛽% 𝛽&\n'
    'Figure 1: The structured mask induced by the exponential-trapezoidal rule.'
)
CLEAN = 'Mamba-3 generalizes Mamba-2 with a more expressive L matrix.'
CJK = 'Mamba-3 沿状态空间序列模型路线改进 Mamba-2，但不是简单增大状态维数。'
CODE = 'def forward(self, x):\n    return self.fc(x)\n' * 3


class FormulaSoupTests(unittest.TestCase):
    def test_detects_pypdf_formula_soup(self):
        self.assertTrue(formula_soup(SOUP))

    def test_ignores_clean_prose_cjk_and_code(self):
        for text in (CLEAN, CJK, CODE, ''):
            self.assertFalse(formula_soup(text))

    def test_gate_replaces_document_excerpt_but_keeps_content(self):
        active = SimpleNamespace(document_kinds={'d1': 'paper'})
        item = {'title': 'Mamba-3 · 第 4 页', 'content': SOUP, 'document_id': 'd1',
                'origin': 'document', 'locator': '第 4 页 · 片段 1'}
        card = {'title': 'Mamba-3 · 核心公式', 'content': SOUP, 'origin': 'model_card', 'locator': None}
        prepared = prepare_evidence([item, card], active, 'Mamba-3 如何实现')

        self.assertEqual(prepared[0]['generation_selection']['strategy'], 'formula_gate')
        self.assertIn('第 4 页', prepared[0]['generation_excerpt'])
        self.assertNotIn('𝛼', prepared[0]['generation_excerpt'])
        self.assertEqual(prepared[0]['content'], SOUP)
        self.assertNotEqual(prepared[1]['generation_selection']['strategy'], 'formula_gate')


if __name__ == '__main__':
    unittest.main()
