import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.generation.grounding_contract import evidence_boundary, SCOPED_EXPLANATION, qualify_absence_scope
from backend.generation.answer_review import review_trigger
from backend.rag import document_report as report
from backend.rag.report_contract import metadata_hint, profile_for
from backend.rag.research import ResearchRunner, table_cell
from test_research import ResearchFixture, fake_model


class ReportScopeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proof = {'section_id': 's', 'quote': 'The corrupted input is used to reconstruct the clean input.'}

    def card(self, identity, role, subject, text):
        return {'card_id': identity, 'role': role, 'subject': subject, 'field': 'methods', 'text': text, 'proofs': [self.proof]}

    def test_mixed_baseline_and_proposed_cards_are_separated(self):
        cards = [self.card('C1', 'baseline', '基础 AE', '基础方法以原始输入为目标。'),
                 self.card('C2', 'proposed', '去噪 AE', '本文方法接收受破坏输入。'),
                 {'card_id': 'S', 'field': 'summary', 'text': '概述', 'proofs': [self.proof]}]
        reduced = {'summary': [{'text': '文章概述', 'card_ids': ['S']}],
                   'methods': [{'text': '所有方法都采用同一配置。', 'card_ids': ['C1', 'C2']}]}
        result = report.normalize_report(reduced, cards)
        self.assertEqual([x['role'] for x in result['methods']], ['baseline', 'proposed'])
        self.assertNotIn('所有方法', str(result))

    def test_exact_duplicate_claims_merge_proofs(self):
        card = self.card('C1', 'baseline', 'AE', '基础方法')
        summary = {'card_id': 'S', 'field': 'summary', 'text': '概述', 'proofs': [self.proof]}
        row = {'text': '基础方法。', 'card_ids': ['C1']}
        result = report.normalize_report({'summary': [{'text': '概述', 'card_ids': ['S']}], 'methods': [row, row]}, [card, summary])
        self.assertEqual(len(result['methods']), 1)

    def test_same_role_synonyms_can_merge_but_research_focus_has_one_core_contribution(self):
        cards = [self.card('C1', 'proposed', '去噪自编码器', '输入被破坏。'),
                 self.card('C2', 'proposed', '本文方法', '目标为干净输入。')]
        for c in cards: c['field'] = 'contributions'
        cards.append({'card_id': 'S', 'field': 'summary', 'text': '概述', 'proofs': [self.proof]})
        value = {'summary': [{'text': '概述', 'card_ids': ['S']}], 'contributions': [
            {'text': '合并为一个去噪训练原则。', 'card_ids': ['C1', 'C2']}, {'text': '重复的第二种表述。', 'card_ids': ['C2']}]}
        result = report.normalize_report(value, cards, profile_for('research'))
        self.assertEqual(len(result['contributions']), 1)
        self.assertEqual(result['contributions'][0]['text'], '合并为一个去噪训练原则。')

    def test_numbered_proofs_extract_exact_original_and_reject_unknown_or_changed_quotes(self):
        unit = report.units_for([{'id': 's', 'content': self.proof['quote']}])[0]
        selected = report.proofs_for([{'unit_id': 'U1', 'passage_id': 1}], {'U1': unit})
        self.assertEqual(selected[0]['quote'], self.proof['quote'])
        for item in ({'unit_id': 'U1', 'passage_id': 99},
                     {'unit_id': 'U1', 'passage_id': 1, 'quote': 'An invented quotation that changes the source.'}):
            self.assertEqual(report.proofs_for([item], {'U1': unit}), [])
        self.assertNotIn('content', report.map_unit(unit))

    def test_conditional_math_bar_is_not_changed_to_division_in_comparison_table(self):
        result = table_cell(r'条件分布 $q_D(\tilde{x}|x)$，以及 $||x||$，代码 `a|b`')
        self.assertIn(r'\mid ', result)
        self.assertIn(r'\Vert ', result)
        self.assertNotIn('／', result)

    def test_review_mention_in_body_does_not_override_research_type(self):
        sections = [{'content': 'We compare our method to a recent survey of related work.'}]
        self.assertEqual(metadata_hint({'title': 'A New Learning Method'}, sections), 'unknown')
        self.assertEqual(metadata_hint({'title': 'Paper'}, [{'content': 'Review Article\nSummary'}]), 'review')

    def test_type_templates_have_distinct_contribution_meanings(self):
        labels = [profile_for(k)['fields']['contributions'] for k in ('research', 'review', 'tutorial', 'reference')]
        self.assertEqual(len(set(labels)), 4)
        self.assertEqual(profile_for('invalid')['kind'], 'unknown')

    async def test_profile_classification_reuses_first_map_call_and_requires_quote(self):
        quote = self.proof['quote']
        section = {'id': 's', 'content': quote, 'title': 'Paper', 'cloud_allowed': True}
        raw = {'document_type': {'kind': 'research', 'proofs': [{'unit_id': 'U1', 'quote': quote}]},
               'findings': [{'field': 'summary', 'text': '本文研究去噪原则', 'role': 'document', 'subject': '本文',
                             'proofs': [{'unit_id': 'U1', 'quote': quote}]}]}
        with patch.object(report, '_chat', new_callable=AsyncMock, side_effect=[raw, {'summary': [{'text': '本文研究去噪原则', 'card_ids': ['C1']}]}]) as chat:
            result = await report.analyze_document({'id': 'p', 'title': 'Paper', 'cloud_allowed': True}, [section], [])
        self.assertEqual(chat.await_count, 2)
        self.assertEqual(result['report_profile']['kind'], 'research')
        self.assertEqual(result['report_profile']['origin'], 'model_with_matched_source')
        raw['document_type']['proofs'][0]['quote'] = 'An invented type classification quotation.'
        with patch.object(report, '_chat', new_callable=AsyncMock, side_effect=[raw, {'summary': [{'text': '概述', 'card_ids': ['C1']}]}]):
            result = await report.analyze_document({'id': 'p', 'title': 'Paper', 'cloud_allowed': True}, [section], [])
        self.assertEqual(result['report_profile']['origin'], 'unconfirmed')

    async def test_research_synthesis_receives_actual_source_scope_without_an_extra_call(self):
        payloads = []
        async def model(purpose, payload):
            if purpose == 'synthesize': payloads.append(payload)
            return await fake_model(purpose, payload)
        runner = ResearchRunner(ResearchFixture(), model_call=model)
        result = await runner.run('比较输入和目标')
        self.assertTrue(result['generated'])
        self.assertTrue(payloads[0]['source_contexts'])
        self.assertIn('claim_scope', payloads[0]['source_contexts'][0])
        self.assertEqual(result['model_calls'], 4)

    def test_source_metadata_distinguishes_notes_and_fixed_code(self):
        active = SimpleNamespace(document_kinds={'code': 'official_code'})
        self.assertIn('教学配置', evidence_boundary({'title': 'AE'}, active)['role'])
        self.assertIn('固定版本', evidence_boundary({'document_id': 'code'}, active)['role'])
        self.assertIn('本轮资料未提供', SCOPED_EXPLANATION)

    def test_document_wide_absence_gets_targeted_review_even_in_learning_mode(self):
        profile = {'level': 'concise'}
        self.assertEqual(review_trigger(profile, {}, enabled=False, draft='全文[1][2]均未提及该任务。'), 'scope_overreach')
        self.assertEqual(review_trigger(profile, {}, enabled=False, draft='本文未涉及该任务。'), 'scope_overreach')
        self.assertIsNone(review_trigger(profile, {}, enabled=False, draft='本轮资料不足以支持该任务的性能。'))
        self.assertIsNone(review_trigger(profile, {}, enabled=False, draft='`全文没有` $\\text{全文没有}$'))

    def test_absence_qualification_narrows_only_asserted_prose_and_keeps_citations(self):
        source = '本文（Vincent 2008）未涉及轴承任务。全文[1][3]均未提及它。`本文没有`，$\\text{全文没有}$，“本文没有”，不能断言全文没有。'
        result, audit = qualify_absence_scope(source)
        self.assertEqual(audit['qualified_claims'], 2)
        self.assertIn('本轮所给正文节选未涉及轴承任务', result)
        self.assertIn('本轮所给正文节选[1][3]未提及它', result)
        self.assertIn('“本文没有”，不能断言全文没有。', result)


if __name__ == '__main__': unittest.main()
