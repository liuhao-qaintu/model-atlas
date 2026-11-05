"""逐项标准、引用真实性、缺失值、来源隔离与已有失败回答捕获。"""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.generation import answer_assessment as assessment

from backend.api import evaluation
from backend.generation.evaluation_metrics import aggregate, rescore_row
from backend.infra.knowledge import TOPICS, SECTION_NAMES, CONTENT_ROOT, parse_sections


def row_fixture():
    return {'case': {'id': 'q', 'question': '模型输出？', 'reference_answer': '特征不是概率', 'cloud_allowed': True,
                     'answer_contract': {'criteria': [
                         {'id': 'a', 'text': '特征不是概率', 'dimension': 'correctness', 'critical': True,
                          'evidence_spans': [{'section_id': 's', 'quote': '输出是特征'}]},
                         {'id': 'b', 'text': '说明维数', 'dimension': 'completeness', 'critical': False,
                          'evidence_spans': [{'section_id': 's', 'quote': '两维特征'}]}]}},
            'answer': '这是两维特征，不是类别概率。', 'generated': True, 'answer_cloud_allowed': True,
            'answer_evidence_details': [{'section_id': 's', 'content': '输出是特征。两维特征。未发送部分。',
                                         'generation_excerpt': '输出是特征。两维特征。', 'cloud_allowed': True}]}


def payload_fixture():
    return {'checks': [{'criterion_id': key, 'verdict': 'pass', 'answer_quote': '两维特征', 'note': '与参考定义一致',
                        'evidence_verdict': 'supported', 'evidence_quotes': [{'section_id': 's', 'quote': '两维特征'}]}
                       for key in ('a', 'b')], 'issues': []}


class AssessmentValidationTests(unittest.TestCase):
    def test_non_object_judge_output_is_a_validation_failure(self):
        for payload in (None, [], 'invalid'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assessment.validate_assessment(row_fixture(), payload, 'model')

    def test_critical_error_cannot_be_averaged_into_correct(self):
        row, payload = row_fixture(), payload_fixture()
        payload['checks'][0].update(verdict='fail', note='核心概念错误')
        result = assessment.validate_assessment(row, payload, 'model', 'judge')
        self.assertEqual(result['overall'], 'incorrect')
        self.assertEqual(result['critical_failures'], ['a'])
        self.assertEqual(result['criterion_pass_rate'], .5)
        self.assertEqual(result['checks'][0]['diagnosis'], 'generation_error')

    def test_missing_duplicate_and_unknown_items_rejected(self):
        for ids in [('a',), ('a', 'a'), ('a', 'x')]:
            payload = payload_fixture()
            payload['checks'] = [{**payload['checks'][0], 'criterion_id': key} for key in ids]
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                assessment.validate_assessment(row_fixture(), payload, 'model')

    def test_answer_quote_is_exact_and_positive_mark_requires_one(self):
        for quote in ('不存在的原句', ''):
            payload = payload_fixture(); payload['checks'][0]['answer_quote'] = quote
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                assessment.validate_assessment(row_fixture(), payload, 'user')
        payload['checks'][0].update(verdict='fail', note='缺项')
        self.assertEqual(assessment.validate_assessment(row_fixture(), payload, 'assistant')['overall'], 'incorrect')

    def test_returned_items_follow_standard_order_even_when_judge_reorders_them(self):
        payload = payload_fixture(); payload['checks'].reverse()
        result = assessment.validate_assessment(row_fixture(), payload, 'model')
        self.assertEqual([c['criterion_id'] for c in result['checks']], ['a', 'b'])

    def test_block_ids_extract_original_markdown_and_reject_unknown_ids(self):
        row = row_fixture(); row['answer'] = '这是**两维特征**，不是类别概率。'
        payload = {'checks': [{'criterion_id': 'a', 'answer_block': 0, 'evidence_refs': [{'section_id': 's', 'block': 0}]}], 'issues': []}
        resolved = assessment.materialize_model_quotes(row, payload)
        self.assertEqual(resolved['checks'][0]['answer_quote'], row['answer'])
        self.assertEqual(resolved['checks'][0]['evidence_quotes'][0]['quote'], row['answer_evidence_details'][0]['generation_excerpt'])
        for identity in (999, True, '0'):
            payload['checks'][0]['answer_block'] = identity
            with self.subTest(identity=identity), self.assertRaises(ValueError): assessment.materialize_model_quotes(row, payload)

    def test_evidence_quote_must_be_in_actual_excerpt_not_full_parent(self):
        payload = payload_fixture(); payload['checks'][0]['evidence_quotes'][0]['quote'] = '未发送部分'
        with self.assertRaises(ValueError): assessment.validate_assessment(row_fixture(), payload, 'model')
        row = row_fixture(); row['answer_evidence_details'][0]['generation_excerpt'] = ''
        with self.assertRaises(ValueError): assessment.validate_assessment(row, payload_fixture(), 'model')

    def test_evidence_shortage_and_unknown_are_not_full_credit(self):
        payload = payload_fixture(); payload['checks'][0].update(evidence_verdict='missing', evidence_quotes=[])
        result = assessment.validate_assessment(row_fixture(), payload, 'model')
        self.assertEqual(result['overall'], 'partial'); self.assertEqual(result['checks'][0]['diagnosis'], 'evidence_gap')
        payload['checks'][0]['evidence_verdict'] = 'unjudged'
        result = assessment.validate_assessment(row_fixture(), payload, 'model')
        self.assertEqual(result['overall'], 'pending'); self.assertIsNone(result['criterion_pass_rate'])

    def test_reference_evidence_cannot_be_marked_not_applicable(self):
        payload = payload_fixture(); payload['checks'][0]['evidence_verdict'] = 'not_applicable'
        with self.assertRaises(ValueError): assessment.validate_assessment(row_fixture(), payload, 'model')

    def test_additional_contradiction_invalidates_otherwise_correct_answer(self):
        payload = payload_fixture()
        payload['issues'] = [{'kind': 'contradiction', 'answer_quote': '不是类别概率', 'note': '测试关键缺陷', 'critical': True}]
        result = assessment.validate_assessment(row_fixture(), payload, 'model')
        self.assertEqual(result['overall'], 'incorrect')
        payload['issues'][0]['answer_quote'] = '伪造引文'
        with self.assertRaises(ValueError): assessment.validate_assessment(row_fixture(), payload, 'model')

    def test_fallback_is_not_graded_as_a_complete_answer(self):
        row = row_fixture(); row['generated'] = False
        with self.assertRaises(ValueError): assessment.validate_assessment(row, payload_fixture(), 'user')

    def test_stale_answer_or_standard_invalidates_cached_assessment(self):
        row = row_fixture(); row['machine_assessment'] = assessment.validate_assessment(row, payload_fixture(), 'model')
        self.assertIsNotNone(assessment.current_assessment(row, 'machine_assessment'))
        for change in ('answer', 'reference', 'evidence'):
            changed = copy.deepcopy(row)
            if change == 'answer': changed['answer'] += '新内容'
            if change == 'reference': changed['case']['reference_answer'] += '新标准'
            if change == 'evidence': changed['answer_evidence_details'][0]['generation_excerpt'] += '新节选'
            self.assertIsNone(assessment.current_assessment(changed, 'machine_assessment'))

    def test_machine_and_assistant_marks_never_count_as_human_accuracy(self):
        row = row_fixture(); row['machine_assessment'] = assessment.validate_assessment(row, payload_fixture(), 'model')
        result = aggregate([row, {'grade': 'correct', 'grade_origin': 'assistant'}, {'grade': 'correct'}])
        self.assertEqual(result['answer_assessment']['machine_reviewed'], 1)
        self.assertEqual(result['user_graded_answers'], 0); self.assertIsNone(result['user_answer_accuracy'])
        self.assertIsNone(result['answer_assessment']['human_pass_rate'])
        row['human_assessment'] = assessment.validate_assessment(row, payload_fixture(), 'user')
        self.assertEqual(assessment.assessment_summary([row])['human_pass_rate'], 1)

    def test_behavioral_missing_evidence_case_has_no_retrieval_score(self):
        row = {'case': {'id': 'x', 'answer_contract': {'evidence_required': False, 'criteria': []}}, 'evidence': []}
        rescore_row(row)
        self.assertIsNone(row['metrics']['precision_at_5']); self.assertFalse(row['metrics']['retrieval_applicable'])

    def test_eight_contracts_have_matching_source_spans_and_pending_review(self):
        data = json.loads((evaluation.config.DATA_ROOT / 'answer-evaluation-cases.json').read_text(encoding='utf-8'))
        sections = []
        for topic in TOPICS:
            if not topic.get('file'): continue
            for section in parse_sections((CONTENT_ROOT / '模型卡' / topic['file']).read_text(encoding='utf-8')):
                slug = SECTION_NAMES.get(section['title'])
                if slug: sections.append({'id': topic['id'] + ':' + slug, 'content': section['content']})
        self.assertEqual(len(data), 8)
        for case in data:
            with self.subTest(case=case['id']):
                assessment.validate_contract(case, sections)
                self.assertEqual(case['reference_review'], 'pending')

    def test_calculation_references_with_local_operators(self):
        import torch
        from fractions import Fraction
        layer = torch.nn.Conv1d(6, 9, 5, groups=3, bias=True)
        self.assertEqual(tuple(layer.weight.shape), (9, 2, 5))
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 99)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(3)
        self.assertEqual(torch.isneginf(mask[1]).tolist(), [False, False, True])
        self.assertTrue(torch.equal(mask.diag(), torch.zeros(3)))
        self.assertEqual(Fraction(1, 7) - Fraction(2, 9), Fraction(-5, 63))


class AssessmentAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_sees_frozen_answer_reference_and_actual_excerpt(self):
        row = row_fixture()
        payload = payload_fixture()
        for check in payload['checks']:
            check.update(answer_block=0, evidence_refs=[{'section_id': 's', 'block': 0}])
        provider = AsyncMock(return_value={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload)}}]})
        with patch.object(assessment, 'provider_post', provider): result = await assessment.assess_with_model(row)
        self.assertEqual(result['origin'], 'model'); provider.assert_awaited_once()
        body = provider.call_args.args[3]; request = json.loads(body['messages'][1]['content'])
        self.assertEqual(request['answer_blocks'][0]['text'], row['answer'])
        self.assertNotIn('未发送部分', request['actual_evidence'][0]['blocks'][0]['text'])

    async def test_private_or_unrecorded_permission_prevents_external_call(self):
        for field in ('case', 'answer', 'evidence'):
            row = row_fixture()
            if field == 'case': row['case']['cloud_allowed'] = False
            if field == 'answer': row.pop('answer_cloud_allowed')
            if field == 'evidence': row['answer_evidence_details'][0].pop('cloud_allowed')
            with patch.object(assessment, 'provider_post', AsyncMock()) as provider:
                with self.assertRaises(ValueError): await assessment.assess_with_model(row)
                provider.assert_not_awaited()

    async def test_truncated_judge_is_not_scored_or_retried(self):
        with patch.object(assessment, 'provider_post', AsyncMock(return_value={'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]})) as provider:
            with self.assertRaises(ValueError): await assessment.assess_with_model(row_fixture())
            provider.assert_awaited_once()


class AssessmentApiTests(unittest.TestCase):
    def test_contract_cannot_bypass_items_with_old_overall_grade_endpoint(self):
        row = row_fixture(); record = {'rows': [row]}
        with patch.object(evaluation, 'get_run', return_value=record):
            with self.assertRaises(HTTPException) as caught:
                evaluation.grade_answer('r', 'q', evaluation.GradeInput(grade='correct'))
        self.assertEqual(caught.exception.status_code, 422)

    def test_capture_keeps_original_answer_and_rejects_other_question(self):
        row = row_fixture(); row.pop('answer'); row['case']['prior_turns'] = [{'question': '前序'}]
        metadata = {'turn_id': '11111111-1111-1111-1111-111111111111', 'generated': True,
                    'evidence': row['answer_evidence_details'], 'plan': {}}
        messages = [SimpleNamespace(role='user', content='前序', payload={}), SimpleNamespace(role='assistant', content='前答', payload={}),
                    SimpleNamespace(role='user', content='模型输出？', payload={}), SimpleNamespace(role='assistant', content='历史原回答', payload=metadata, id='m')]
        session = MagicMock(); session.return_value.__enter__.return_value.scalars.return_value.all.return_value = list(reversed(messages))
        value = evaluation.CaptureInput(session_id='22222222-2222-2222-2222-222222222222', turn_id=metadata['turn_id'])
        for question, okay in [('模型输出？', True), ('另一问题', False)]:
            copied = copy.deepcopy(row); copied['case']['question'] = question
            with patch.object(evaluation, 'get_run', return_value={'rows': [copied]}), patch.object(evaluation, 'Session', session), \
                 patch.object(evaluation, 'session_is_cloud_safe', return_value=True), patch.object(evaluation, 'save'):
                if okay:
                    result = evaluation.capture_existing_answer('r', 'q', value)
                    self.assertEqual(result['rows'][0]['answer'], '历史原回答')
                    self.assertEqual(result['rows'][0]['answer_origin'], 'captured_session')
                else:
                    with self.assertRaises(HTTPException): evaluation.capture_existing_answer('r', 'q', value)


if __name__ == '__main__': unittest.main()
