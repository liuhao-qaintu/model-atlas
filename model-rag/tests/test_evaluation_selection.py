"""定向评测不运行未选择的问题，也不静默忽略失效题目。"""
import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from backend.api.evaluation import RunInput, select_cases


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = [{'id': 'a', 'question': '第一题'}, {'id': 'b', 'question': '第二题'}, {'id': 'c', 'question': '第三题'}]

    def test_subset_keeps_requested_order_and_deduplicates(self):
        selected = select_cases(self.dataset, RunInput(case_ids=['c', 'a', 'c']).case_ids)
        self.assertEqual([case['id'] for case in selected], ['c', 'a'])
        self.assertEqual([case['id'] for case in self.dataset], ['a', 'b', 'c'])

    def test_unselected_scope_is_full_and_invalid_scope_is_rejected(self):
        self.assertEqual(select_cases(self.dataset, RunInput().case_ids), self.dataset)
        with self.assertRaises(HTTPException) as missing:
            select_cases(self.dataset, ['a', 'deleted'])
        self.assertEqual(missing.exception.status_code, 422)
        with self.assertRaises(ValidationError): RunInput(case_ids=[])
        with self.assertRaises(ValidationError): RunInput(case_ids=['a'] * 51)


if __name__ == '__main__': unittest.main()
