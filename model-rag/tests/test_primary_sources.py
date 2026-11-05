import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from backend.infra.ingestion import page_numbers, parse_document
from backend.infra import public_sources
from backend.infra.database import KnowledgeDocument, KnowledgeSection


class PrimarySourceTests(unittest.TestCase):
    def pdf(self):
        buffer = io.BytesIO()
        pdf = PdfWriter()
        for i in range(1, 4):
            page = pdf.add_blank_page(width=300, height=150)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): pdf._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 10 100 Td (ORIGINAL PAGE {i}) Tj ET'.encode())
            page[NameObject('/Contents')] = pdf._add_object(stream)
        pdf.write(buffer)
        return buffer.getvalue()

    def test_page_selection_preserves_original_page_numbers_and_bytes(self):
        raw = self.pdf()
        selected = parse_document('source.pdf', raw, '3,1')
        self.assertEqual(selected['page_count'], 3)
        self.assertEqual(selected['selected_pages'], [1, 3])
        # 章节可跨页，不要求一页一个块；校验页范围和正文均完整。
        self.assertEqual(selected['parts'][0]['page'], 1)
        self.assertEqual({p for part in selected['parts'] for p in part['chapter_pages']}, {1, 3})
        self.assertTrue(all(f'ORIGINAL PAGE {p}' in '\n'.join(part['content'] for part in selected['parts']) for p in (1, 3)))
        self.assertTrue(all('ORIGINAL PAGE 2' not in part['content'] for part in selected['parts']))
        self.assertTrue(raw.startswith(b'%PDF-'))

    def test_invalid_ranges_are_rejected_before_extracting_or_building_range(self):
        for value in ['0', '4', '3-1', '1-999999999', '1,,2', '1;2', '-1', '1.5']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                page_numbers(value, 3)
        self.assertEqual(page_numbers('1-3, 2,3'), [1, 2, 3])
        self.assertIsNone(page_numbers(''))

    def test_page_ranges_cannot_be_applied_to_markdown(self):
        with self.assertRaises(ValueError):
            parse_document('sample.md', b'# Example\ncontent', '1')

    def test_full_pdf_remains_available_by_default(self):
        result = parse_document('source.pdf', self.pdf())
        self.assertIsNone(result['selected_pages'])
        self.assertEqual({p for part in result['parts'] for p in part['chapter_pages']}, {1, 2, 3})
        self.assertEqual(result['parts'][0]['page'], 1)
        self.assertTrue(all(f'ORIGINAL PAGE {p}' in '\n'.join(part['content'] for part in result['parts']) for p in (1, 2, 3)))

    def test_primary_graph_support_requires_registered_hash_owner_and_source(self):
        entry = {'key': 'clip-code', 'document_id': 'sample', 'import_sha256': 'checked', 'source_url': 'https://github.com/openai/CLIP'}
        document = {'id': 'sample', 'status': 'imported', 'sha256': 'checked', 'model_id': 'clip', 'source_url': entry['source_url']}
        section = {'id': 'doc:sample:0', 'document_id': 'sample', 'content': 'self.visual = VisionTransformer(', 'model_id': 'clip'}
        def source_records(table):
            return [document] if table is KnowledgeDocument else [section] if table is KnowledgeSection else []
        def edge():
            return {'id': 'clip-vision', 'owner_model_id': 'clip', 'source_ids': ['S26'], 'evidence': [{'section_id': 'clip:overview', 'quote': 'existing'}]}
        with tempfile.TemporaryDirectory() as folder, patch.object(public_sources, 'DATA_ROOT', Path(folder)), patch.object(public_sources, 'records', source_records):
            Path(folder, 'public-source-manifest.json').write_text(json.dumps({'entries': [entry]}), encoding='utf-8')
            result = public_sources.enrich_relations([edge()])[0]
            self.assertEqual(result['evidence'][0]['section_id'], section['id'])
            self.assertIn('Dsample', result['source_ids'])
            for key, bad in [('sha256', 'different'), ('model_id', 'vit'), ('source_url', 'https://example.test')]:
                original = document[key]
                document[key] = bad
                self.assertEqual(len(public_sources.enrich_relations([edge()])[0]['evidence']), 1)
                document[key] = original
