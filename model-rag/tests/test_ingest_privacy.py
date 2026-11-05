"""文档报告替代逐片段审核：许可不足不发送云端，也不创建占位审核项。"""
import unittest
from unittest.mock import AsyncMock, patch
from backend.api import ingest
from backend.infra.ingest_job import IngestJob


class IngestPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def check_permission(self, document_permission, section_permission):
        document = {'id': 'private', 'title': 'Private', 'cloud_allowed': document_permission}
        sections = [{'id': 's', 'content': 'Private content', 'cloud_allowed': section_permission}]
        job = IngestJob(id='job', document_id='private', owner_user_id='local-user')
        with patch('backend.rag.document_report.analyze_document', new_callable=AsyncMock) as analyze, \
             patch('backend.infra.document_reports.retire_placeholders'), \
             patch.object(ingest.ingest_job, 'publish', new_callable=AsyncMock), \
             patch.object(ingest.ingest_job, 'finish', new_callable=AsyncMock) as finish, \
             patch.object(ingest, 'upsert_records') as save:
            await ingest._run(job, document, sections, [])
            analyze.assert_not_called(); save.assert_not_called()
            self.assertEqual(finish.call_args.args[1], 'skipped')

    async def test_private_document_blocks_even_public_section(self):
        await self.check_permission(False, True)

    async def test_private_section_blocks_whole_analysis(self):
        await self.check_permission(True, False)

    async def test_public_document_saves_one_report_without_section_review(self):
        document = {'id': 'public', 'title': 'Public', 'status': 'imported', 'cloud_allowed': True}
        sections = [{'id': 's', 'content': 'Public text', 'cloud_allowed': True}]
        report = {'id': 'dr-test', 'type': 'document_report', 'coverage': {'llm_calls': 2}}
        job = IngestJob(id='job', document_id='public', owner_user_id='local-user')
        with patch('backend.rag.document_report.analyze_document', new_callable=AsyncMock, return_value=report) as analyze, \
             patch('backend.infra.document_reports.retire_placeholders'), \
             patch('backend.infra.ingestion.document_record', return_value=document), \
             patch.object(ingest.ingest_job, 'publish', new_callable=AsyncMock), \
             patch.object(ingest.ingest_job, 'finish', new_callable=AsyncMock) as finish, \
             patch.object(ingest, 'upsert_records') as save:
            await ingest._run(job, document, sections, [])
            analyze.assert_awaited_once()
            self.assertEqual(save.call_args.args[1], [report])
            self.assertEqual(finish.call_args.args[1], 'done')


if __name__ == '__main__': unittest.main()
