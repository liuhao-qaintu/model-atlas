"""真实本地上传/数据库链路；外部模型全部用替身，不上传测试文件。"""
import asyncio
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
from auth_fixtures import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.infra import config

from backend import main
from backend.infra.database import ChatMessage, ChatSession, KnowledgeDocument, KnowledgeSection, KnowledgeSource, engine
from backend.infra.ingestion import MAX_BYTES, UPLOAD_ROOT, parse_document
from backend.rag.retrieval import Retriever


def text_pdf():
    writer = PdfWriter()
    for text in (b"CNN input tensor on page one.", b"CNN output logits on page two."):
        page = writer.add_blank_page(width=300, height=150)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (" + text + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class ParsingTests(unittest.TestCase):
    def test_pdf_extracts_actual_text_and_page_numbers(self):
        parsed = parse_document("paper.pdf", text_pdf())
        self.assertEqual(parsed["page_count"], 2)
        self.assertEqual({page for part in parsed['parts'] for page in part['chapter_pages']},{1,2})
        content='\n'.join(part['content'] for part in parsed['parts'])
        self.assertIn('input tensor',content)
        self.assertIn('output logits',content)

    def test_pdf_warns_when_page_looks_like_formula_soup(self):
        with patch('backend.infra.ingestion.formula_soup', return_value=True):
            parsed = parse_document("paper.pdf", text_pdf())
        self.assertTrue(any('公式版式' in warning for warning in parsed["warnings"]))
        self.assertEqual({page for part in parsed['parts'] for page in part['chapter_pages']},{1,2})
        self.assertIn('output logits','\n'.join(part['content'] for part in parsed['parts']))

    def test_markdown_protects_headings_inside_code_and_bounds_prose(self):
        text = "# 模型\n## 输入\n[B,C,L]\n\n```python\n## 这是代码\nx = 1\n```\n## 输出\n" + "长段落" * 1000
        parsed = parse_document("card.md", text.encode())
        self.assertFalse(any(part["title"] == "这是代码" for part in parsed["parts"]))
        self.assertTrue(any("```python\n## 这是代码\nx = 1\n```" in part["content"] for part in parsed["parts"]))
        self.assertTrue(all(len(part["content"]) <= 1800 for part in parsed["parts"]))
        code = "~~~python\n# code\n\nx = 1\n~~~"
        self.assertIn(code, parse_document("fenced.md", ("## 代码\n" + code).encode())["parts"][0]["content"])

    def test_invalid_empty_scanned_and_oversized_files_are_explicit(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        blank = io.BytesIO(); writer.write(blank)
        for filename, raw in [("bad.exe", b"hello"), ("empty.txt", b""), ("gbk.txt", b"\xff"), ("scan.pdf", blank.getvalue()), ("bad.pdf", b"not a pdf"), ("huge.txt", b"a" * (MAX_BYTES + 1))]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                parse_document(filename, raw)


class ImportFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.document_ids, self.sessions = [], []

    def tearDown(self):
        with Session(engine) as db:
            db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(self.sessions)))
            db.execute(delete(ChatSession).where(ChatSession.id.in_(self.sessions)))
            for document_id in self.document_ids:
                db.execute(delete(KnowledgeSection).where(KnowledgeSection.id.like(f"doc:{document_id}:%")))
                db.execute(delete(KnowledgeSource).where(KnowledgeSource.id == f"D{document_id}"))
                db.execute(delete(KnowledgeDocument).where(KnowledgeDocument.id == document_id))
                for suffix in ("md", "txt", "pdf"):
                    path = (UPLOAD_ROOT / f"{document_id}.{suffix}").resolve()
                    self.assertEqual(path.parent, UPLOAD_ROOT.resolve())
                    path.unlink(missing_ok=True)
            db.commit()
        main.retriever = Retriever()

    def upload(self, model="clip", filename="test.md"):
        marker = f"fixture{uuid4().hex}"
        raw = f"# 本地测试资料\n## 输入输出\n{marker} 输入为图像和文本。输出为两个表示。".encode()
        result = self.client.post("/api/documents/preview", files={"file": (filename, raw)}, data={"model_id": model})
        self.assertEqual(result.status_code, 200, result.text)
        self.document_ids.append(result.json()['id'])
        return result.json(), marker, raw

    def commit(self, document, cloud_allowed=False):
        result = self.client.post(f"/api/documents/{document['id']}/import", json={"title": document["title"], "cloud_allowed": cloud_allowed})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def test_preview_import_dedup_and_original_download(self):
        document, marker, raw = self.upload(filename="../中文资料.md")
        self.assertEqual(document["filename"], "中文资料.md")
        self.assertFalse(any(c.get("document_id") == document["id"] for c in main.retriever.chunks))
        self.commit(document)
        second = self.commit(document)
        self.assertTrue(second["document"]["duplicate"])
        listing = self.client.get("/api/documents").json()
        self.assertEqual(sum(d["id"] == document["id"] for d in listing), 1)
        detail = self.client.get(f"/api/documents/{document['id']}").json()
        self.assertEqual(sum(marker in section['content'] for section in detail['sections']),1)
        self.assertTrue(all(section['document_id']==document['id'] for section in detail['sections']))
        self.assertEqual(self.client.get(f"/api/documents/{document['id']}/file").content, raw)
        duplicate = self.client.post("/api/documents/preview", files={"file": ("renamed.md", raw)}, data={"model_id": "clip"}).json()
        self.assertTrue(duplicate["duplicate"])

    def test_local_only_document_retrieval_never_sends_text_to_provider(self):
        document, marker, _ = self.upload()
        self.commit(document)
        session = str(uuid4()); self.sessions.append(session)
        retrieval_provider, generation_provider = AsyncMock(), AsyncMock()
        with patch("backend.rag.retrieval.provider_post", retrieval_provider), patch("backend.main.provider_post", generation_provider):
            result = self.client.post("/api/chat", json={"question": f"{marker} 输入输出是什么？", "model_id": "clip", "session_id": session}).json()
        self.assertTrue(result["evidence"])
        self.assertFalse(result["generated"])
        self.assertIn("原文摘录", result["answer"])
        self.assertEqual(result["evidence"][0]["document_id"], document["id"])
        self.assertIn(f"/documents/{document['id']}?part=", result["evidence"][0]["reader_url"])
        retrieval_provider.assert_not_awaited(); generation_provider.assert_not_awaited()
        # 后续改问公开模型时，也不能把上轮本地摘录夹进云端历史。
        generation_provider = AsyncMock(return_value={"choices": [{"message": {"content": "公开模型的解释。[1]"}}]})
        with patch.object(main.retriever, "vectors", None), patch("backend.infra.config.RERANK_KEY", ""), patch("backend.main.provider_post", generation_provider):
            self.client.post("/api/chat", json={"question": "1D CNN 输入是什么？", "model_id": "cnn-1d", "session_id": session})
        self.assertNotIn(marker, str(generation_provider.await_args.args[3]["messages"]))

    def test_incremental_index_reuses_old_vectors_and_excludes_local_only(self):
        approved, _, _ = self.upload(model="transformer")
        self.commit(approved, cloud_allowed=True)
        private, private_marker, _ = self.upload(model="clip")
        self.commit(private)
        with tempfile.TemporaryDirectory(dir=config.DATA_ROOT) as folder, patch.object(config, "DATA_ROOT", Path(folder)):
            retriever = Retriever()
            # 已有公开文档也是旧知识；仅这次新增的测试文档需要嵌入。
            originals = [i for i, c in enumerate(retriever.chunks) if c['cloud_allowed'] and c.get('document_id') not in {approved['id'], private['id']}]
            retriever.vector_cache = {retriever.chunk_keys[i]: np.array([1., 0., 0.], dtype=np.float32) for i in originals}
            approved_texts={retriever.texts[i] for i,chunk in enumerate(retriever.chunks) if chunk.get('document_id')==approved['id'] and chunk['cloud_allowed']}
            private_count=sum(not chunk['cloud_allowed'] for chunk in retriever.chunks)

            async def embed(texts, query=False):
                self.assertEqual(set(texts),approved_texts)
                self.assertFalse(any(private_marker in text for text in texts))
                return np.tile(np.array([0., 1., 0.], dtype=np.float32), (len(texts), 1))

            fake = AsyncMock(side_effect=embed)
            with patch.object(retriever, "embed", fake):
                result = asyncio.run(retriever.build_index())
            self.assertEqual(result["added_chunks"],len(approved_texts))
            self.assertEqual(result["local_only_chunks"],private_count)
            self.assertEqual(fake.await_count, 1)
            reloaded = Retriever()
            with patch.object(reloaded, "embed", AsyncMock(side_effect=AssertionError("不应重新嵌入"))):
                result = asyncio.run(reloaded.build_index())
            self.assertTrue(result["cached"])
            self.assertEqual(result["added_chunks"], 0)

    def test_invalid_model_and_source_url_do_not_import(self):
        response = self.client.post("/api/documents/preview", files={"file": ("test.md", b"hello")}, data={"model_id": "missing"})
        self.assertEqual(response.status_code, 422)
        document, _, _ = self.upload()
        response = self.client.post(f"/api/documents/{document['id']}/import", json={"title": "test", "source_url": "javascript:alert(1)"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get(f"/api/documents/{document['id']}").json()["status"], "preview")
