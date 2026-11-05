import unittest
from test_research import ResearchFixture, fake_model
from backend.rag.research import ResearchRunner


class ResearchMultiDocumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_stategraph_reads_two_documents_and_preserves_pool_in_report(self):
        fixture = ResearchFixture()
        fixture.documents += [{'id': 'clip-paper', 'cloud_allowed': True}, {'id': 'llava-paper', 'cloud_allowed': True}]
        for section in fixture.sections:
            if section['model_id'] != 'private': section['document_id'] = section['model_id'] + '-paper'
        runner = ResearchRunner(fixture, ['clip-paper', 'llava-paper'], model_call=fake_model)
        result = await runner.run('比较两篇论文的输入和训练目标')
        self.assertEqual(result['document_ids'], ['clip-paper', 'llava-paper'])
        self.assertEqual(set(runner.catalog), {'clip', 'llava'})
        self.assertEqual({e['document_id'] for e in result['evidence']}, {'clip-paper', 'llava-paper'})
        self.assertTrue(result['generated'])
        self.assertGreater(result['tool_calls'], 1)
        self.assertTrue(all('secret' not in call[2]['allowed_section_ids'] for call in fixture.search_calls))


if __name__ == '__main__': unittest.main()
