import unittest
from backend.generation.answer_validation import citation_numbers, keep_known_links, validate_citations


class AnswerValidationTests(unittest.TestCase):
    def test_intervals_are_not_converted_into_citations(self):
        text = 'Sigmoid 映射到 [0,1] 区间，范围为 [1,5]，依据[1,2]。'
        self.assertEqual(validate_citations(text, 2), ('Sigmoid 映射到 [0,1] 区间，范围为 [1,5]，依据[1][2]。', False))

    def test_grouped_numbers_split_and_invalid_members_are_checked(self):
        text = '说明[1, 3, 3][2]。另一项[2，99]。'
        answer, invalid = validate_citations(text, 3)
        self.assertEqual(answer, '说明[1][3][2]。另一项[2]。')
        self.assertTrue(invalid)

    def test_grouped_references_do_not_rewrite_shapes_links_code_or_math(self):
        text = r'形状为 [2, 32]，x[1,3]，`[1,3]`，$[1,3]$，[1,3](https://example.org)，[1,3][source]。结论[1,3]'
        answer, invalid = validate_citations(text, 3)
        self.assertEqual(answer, text[:-5] + '[1][3]')
        self.assertFalse(invalid)

    def test_bare_evidence_labels_become_links_and_still_check_range(self):
        text = '见证据1和证据 3；未命中证据99。`证据1`，$\\text{证据2}$'
        answer, invalid = validate_citations(text, 3)
        self.assertTrue(invalid)
        self.assertEqual(answer, '见[1]和[3]；未命中。`证据1`，$\\text{证据2}$')
        self.assertEqual(validate_citations(answer, 3), (answer, False))

    def test_provider_math_delimiters_preserve_indices_and_code(self):
        text = r'结论[证据2]，\(x[99]\)；\[a[0]+b[1]\]，$c[88]$。`\(literal[0]\)`'
        text += '\n```python\n' + r's = "\[code[99]\]"' + '\n```'
        answer, invalid = validate_citations(text, 2)
        self.assertFalse(invalid)
        self.assertIn(r'结论[2]，\(x[99]\)；\[a[0]+b[1]\]，$c[88]$。', answer)
        self.assertIn(r'`\(literal[0]\)`', answer)
        self.assertIn(r's = "\[code[99]\]"', answer)
        self.assertEqual(citation_numbers(text), [])
        self.assertEqual(citation_numbers(answer), [2])
        self.assertEqual(validate_citations(answer, 2), (answer, False))

    def test_multiple_chinese_numbers_split_into_valid_clickable_references(self):
        text = '结论[证据1, 证据3]；另一项[证据 2、99]。`a[证据1,3]`'
        answer, invalid = validate_citations(text, 3)
        self.assertEqual(answer, '结论[1][3]；另一项[2]。`a[证据1,3]`')
        self.assertTrue(invalid)

    def test_chinese_evidence_numbers_become_clickable_but_code_and_math_stay_literal(self):
        text = '正文[证据 1]，超界[证据99]。`x[证据1]`，$a[证据2]$\n```python\nx="[证据1]"\n```'
        answer, invalid = validate_citations(text, 2)
        self.assertTrue(invalid)
        self.assertIn('正文[1]，超界。', answer)
        self.assertIn('`x[证据1]`', answer)
        self.assertIn('$a[证据2]$', answer)
        self.assertIn('x="[证据1]"', answer)
        self.assertEqual(citation_numbers(answer), [1])

    def test_invalid_prose_reference_is_removed_without_changing_python(self):
        text = '说明[1]，错误引用[99]。`x[0]` 与 `torch.arange(x.shape[0])`。\n```python\nx = a[99]\ny = callbacks[0](x)\n```'
        answer, invalid = validate_citations(text, 2)
        self.assertTrue(invalid)
        self.assertIn('说明[1]，错误引用。', answer)
        self.assertIn('torch.arange(x.shape[0])', answer)
        self.assertIn('x = a[99]', answer)
        self.assertEqual(citation_numbers(answer), [1])

    def test_math_tilde_fences_and_double_backticks_are_preserved(self):
        text = '$x[0]$\n$$\nx[99]\n$$\n~~~python\ny[0]\n~~~\n``code `[0]` literal``\n正文[2]'
        answer, invalid = validate_citations(text, 2)
        self.assertEqual(answer, text)
        self.assertFalse(invalid)
        self.assertEqual(citation_numbers(text), [2])

    def test_link_checks_preserve_indexed_calls_and_code_markdown(self):
        text = '[不存在](/fake) `callbacks[0](value)`\n```python\nx = a[0](b)\nlabel = "[doc](https://example.test)"\n```\n[资料](/models/clip)'
        answer, dropped = keep_known_links(text, {'/models/clip'})
        self.assertTrue(dropped)
        self.assertNotIn('](/fake)', answer)
        self.assertIn('callbacks[0](value)', answer)
        self.assertIn('x = a[0](b)', answer)
        self.assertIn('[doc](https://example.test)', answer)
        self.assertIn('[资料](/models/clip)', answer)
