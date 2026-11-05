"""指标缺失、跨模型候选、审阅来源与真实追问隔离的回归。"""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.api import evaluation

from backend import main
from backend.generation.evaluation_metrics import aggregate, normalize_run, qrels_key, retrieval_metrics, reuse_judgments


def mark(label, origin='user'):
    return {'label': label, 'origin': origin, 'note': '核对原文'}


class ProtocolMetricsTests(unittest.TestCase):
    def test_unjudged_is_not_a_negative_and_bounds_are_explicit(self):
        result = retrieval_metrics(['a', 'b'], ['a'])
        self.assertIsNone(result['precision_at_5']); self.assertIsNone(result['returned_precision'])
        self.assertEqual((result['precision_lower_bound'], result['precision_upper_bound']), (.2, .4))
        self.assertEqual((result['returned_precision_lower_bound'], result['returned_precision_upper_bound']), (.5, 1))
        self.assertEqual(result['judgment_coverage'], .5)

    def test_fixed_k_and_actual_size_are_never_conflated(self):
        result = retrieval_metrics(['a', 'b'], ['a'], judgments={'a': mark('direct'), 'b': mark('supporting')})
        self.assertEqual(result['precision_at_5'], .4); self.assertEqual(result['returned_precision'], 1)
        self.assertEqual(result['recall_at_5'], 1); self.assertTrue(result['fully_user_reviewed'])

    def test_no_relevant_empty_results_duplicates_and_top_k(self):
        result = retrieval_metrics(['a', 'a', 'b', 'c', 'd', 'e', 'f'], [], judgments={sid: mark('irrelevant') for sid in 'abcdef'})
        self.assertEqual(result['retrieved'], 5); self.assertEqual(result['returned_precision'], 0)
        self.assertIsNone(result['recall_at_5']); self.assertIsNone(result['reciprocal_rank'])
        empty = retrieval_metrics([], ['a'])
        self.assertEqual(empty['precision_at_5'], 0); self.assertEqual(empty['recall_at_5'], 0)
        self.assertIsNone(empty['returned_precision'])

    def test_marking_back_to_unjudged_restores_missing_score(self):
        result = retrieval_metrics(['a'], ['a'], judgments={'a': mark('unjudged')})
        self.assertIsNone(result['precision_at_5']); self.assertEqual(result['unjudged'], 1)
        self.assertIsNone(result['recall_at_5'])

    def test_nonreturned_positives_expand_recall_denominator(self):
        result = retrieval_metrics(['a'], ['a'], judgments={'a': mark('direct'), 'b': mark('supporting')})
        self.assertEqual(result['recall_at_5'], .5); self.assertEqual(result['returned_precision'], 1)

    def test_aggregate_uses_each_metric_sample_count_and_excludes_unjudged(self):
        rows = [{'metrics': retrieval_metrics(['a'], ['a'])}, {'metrics': retrieval_metrics(['a', 'b'], ['a'])},
                {'metrics': retrieval_metrics(['b'], [], judgments={'b': mark('irrelevant')})}]
        result = aggregate(rows)
        self.assertEqual(result['metric_counts']['precision_at_5'], 2)
        self.assertEqual(result['precision_at_5'], .1); self.assertEqual(result['returned_precision'], .5)
        self.assertEqual(result['judgment_coverage'], .75)
        self.assertEqual(result['user_reviewed_rows'], 1)

    def test_auxiliary_and_user_reviews_and_answer_dimensions_remain_separate(self):
        result = aggregate([{'grade': 'correct', 'grade_origin': 'assistant', 'answer_rubric': {'correctness': 'correct', 'faithfulness': 'partial'}},
                            {'grade': 'partial', 'grade_origin': 'user', 'answer_rubric': {'conditions': 'not_applicable'}}])
        self.assertEqual(result['assistant_graded_answers'], 1); self.assertEqual(result['user_graded_answers'], 1)
        self.assertEqual(result['answer_dimensions']['faithfulness'], {'count': 1, 'pass_rate': 0})
        self.assertEqual(result['answer_dimensions']['conditions']['count'], 0)

    def test_historical_read_is_nonmutating_and_old_scores_survive(self):
        old = {'summary': {'precision_at_5': .2}, 'rows': [{'case': {'relevant_section_ids': ['a']},
               'evidence': [{'section_id': 'a'}, {'section_id': 'b'}], 'metrics': {'precision_at_5': .2}}]}
        before = copy.deepcopy(old); fresh = normalize_run(old)
        self.assertEqual(old, before); self.assertEqual(fresh['legacy_summary'], before['summary'])
        self.assertEqual(fresh['rows'][0]['legacy_metrics'], {'precision_at_5': .2})
        self.assertIsNone(fresh['summary']['precision_at_5'])
        self.assertEqual(normalize_run(fresh), fresh)

    def test_clarification_is_not_scored_as_failed_retrieval(self):
        run = normalize_run({'protocol': 'relevance-pool-v2', 'rows': [{'case': {'id': 'q', 'category': 'clarification'}, 'evidence': []}]})
        self.assertIsNone(run['summary']['precision_at_5']); self.assertEqual(run['summary']['metric_counts']['precision_at_5'], 0)

    def test_default_run_excludes_challenges_and_cross_split_selection_rejected(self):
        data = [{'id': 'dev'}, {'id': 'hold', 'split': 'challenge'}]
        self.assertEqual(evaluation.RunInput().split, 'development')
        self.assertEqual([r['id'] for r in evaluation.select_cases(data, None, 'development')], ['dev'])
        with self.assertRaises(HTTPException): evaluation.select_cases(data, ['hold'], 'development')

    def test_challenge_set_has_frozen_provenance_and_global_scope(self):
        data = json.loads((evaluation.config.DATA_ROOT / 'evaluation-challenge-cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data), 8); self.assertTrue(all(c['model_id'] is None for c in data))
        self.assertTrue(all(c['reference_review'] == 'pending' and c['frozen_at'] for c in data))
        self.assertTrue({'comparison', 'multihop', 'followup', 'clarification', 'missing_evidence'} <= {c['category'] for c in data})

    def test_qrels_reuse_requires_same_question_answer_and_corpus(self):
        case = {'id': 'q', 'question': '原问题', 'reference_answer': '原答案', 'relevant_section_ids': ['a']}
        pools = {qrels_key(case, 'v1'): {'judgments': {'a': mark('direct'), 'b': mark('irrelevant')}}}
        self.assertIn('b', reuse_judgments(case, 'v1', pools))
        for changed, corpus in [(case, 'v2'), ({**case, 'question': '新问题'}, 'v1'), ({**case, 'reference_answer': '新答案'}, 'v1')]:
            self.assertNotIn('b', reuse_judgments(changed, corpus, pools))
        current = reuse_judgments(case, 'v1', pools); current['b']['label'] = 'supporting'
        self.assertEqual(pools[qrels_key(case, 'v1')]['judgments']['b']['label'], 'irrelevant')


class JudgmentApiTests(unittest.TestCase):
    def setUp(self):
        self.record = normalize_run({'id': 'run', 'corpus_sha256': 'current', 'rows': [{'case': {'id': 'q', 'relevant_section_ids': ['a']},
                         'evidence': [{'section_id': 'a'}, {'section_id': 'b'}]}]})
        self.active = SimpleNamespace(sections=[{'id': 'a'}, {'id': 'b'}, {'id': 'other-model'}])

    def judge(self, **fields):
        value = evaluation.JudgmentInput(section_id='b', note='此段说明相关条件', **fields)
        with patch.object(evaluation, 'get_run', return_value=copy.deepcopy(self.record)), patch.object(evaluation, 'save') as save, \
             patch.object(main, 'retriever', self.active), patch.object(evaluation, 'corpus_fingerprint', return_value='current'):
            result = evaluation.judge_evidence('run', 'q', value)
            save.assert_called_once(); return result

    def test_relevance_updates_metrics_and_keeps_annotator_and_history(self):
        result = self.judge(label='supporting', origin='assistant'); row = result['rows'][0]
        self.assertEqual(row['metrics']['returned_precision'], 1); self.assertEqual(row['metrics']['precision_at_5'], .4)
        self.assertEqual(row['judgment_history'][0]['after']['origin'], 'assistant')
        self.assertFalse(row['metrics']['fully_user_reviewed'])

    def test_unknown_section_stale_corpus_and_blank_note_are_rejected(self):
        with patch.object(evaluation, 'get_run', return_value=self.record), patch.object(main, 'retriever', self.active), \
             patch.object(evaluation, 'corpus_fingerprint', return_value='stale'):
            with self.assertRaises(HTTPException) as stale:
                evaluation.judge_evidence('run', 'q', evaluation.JudgmentInput(section_id='b', label='direct', note='依据'))
            self.assertEqual(stale.exception.status_code, 409)
        for sid, note in [('missing', '依据'), ('b', ' ')]:
            with patch.object(evaluation, 'get_run', return_value=self.record), patch.object(main, 'retriever', self.active), \
                 patch.object(evaluation, 'corpus_fingerprint', return_value='current'):
                with self.assertRaises(HTTPException) as invalid:
                    evaluation.judge_evidence('run', 'q', evaluation.JudgmentInput(section_id=sid, label='direct', note=note))
                self.assertEqual(invalid.exception.status_code, 422)


class FollowupEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_prelude_is_executed_before_final_query(self):
        planner = MagicMock(side_effect=lambda q, *a, **kw: {'model_id': 'resnet-18', 'question': q})
        execute = AsyncMock(return_value={'evidence': [{'section_id': 'a'}]})
        with patch.object(evaluation, 'rule_plan', planner), patch.object(evaluation, 'execute_plan', execute):
            _, _, prelude = await evaluation.retrieve_case({'question': '那去掉fc呢？', 'prior_turns': [{'question': 'ResNet-18输出？'}]}, object(), False)
        self.assertEqual(execute.await_count, 2); self.assertEqual(prelude[0]['evidence_ids'], ['a'])
        self.assertEqual(planner.call_args.kwargs['previous'][1]['metadata']['model_id'], 'resnet-18')

    async def test_real_generation_isolated_session_cleanup_even_when_provider_fails(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                record = {'id': 'r', 'corpus_sha256': 'current', 'rows': [{'case': {'id': 'q', 'question': '它呢？',
                          'prior_turns': [{'question': 'ResNet-18输出？'}], 'cloud_allowed': True}}]}
                reply = {'answer': '回答', 'generated': True, 'evidence': [], 'plan': {}}
                chat = AsyncMock(side_effect=[reply, RuntimeError('failed') if fail else reply])
                with patch.object(evaluation, 'get_run', side_effect=lambda _: copy.deepcopy(record)), patch.object(evaluation, 'save'), \
                     patch.object(evaluation, 'corpus_fingerprint', return_value='current'), patch.object(main, 'chat', chat), \
                     patch.object(evaluation, 'Session') as session:
                    if fail:
                        with self.assertRaises(RuntimeError): await evaluation.generate_answer('r', 'q')
                    else:
                        result = await evaluation.generate_answer('r', 'q')
                        self.assertEqual(len(result['rows'][0]['answer_prelude']), 1)
                session.return_value.__enter__.return_value.execute.assert_called()
                session.return_value.__enter__.return_value.commit.assert_called_once()
                ids = [call.args[0].session_id for call in chat.await_args_list]
                self.assertEqual(len(set(ids)), 1); self.assertTrue(ids[0])


if __name__ == '__main__': unittest.main()
