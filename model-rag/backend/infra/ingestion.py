"""本地解析、预览后入库；原始文件保留，未经允许的正文不会送往云端。"""
import hashlib
import io
import math
import re
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from pypdf import PdfReader
from sqlalchemy.orm import Session

from backend.infra import config
from backend.infra.database import KnowledgeDocument, KnowledgeSection, KnowledgeSource, engine
from backend.infra.knowledge import split_blocks
from backend.infra.text_quality import formula_soup

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_TEXT = 1_000_000
UPLOAD_ROOT = config.DATA_ROOT / "uploads"


def safe_filename(filename):
    return re.split(r"[/\\]", filename or "document")[-1][:180]


def split_text(text, limit=1800):
    """保留代码/块公式；长普通段落再切分，避免整篇论文成为一个父块。"""
    result = []
    for block in split_blocks(text, max_chars=limit):
        if len(block) <= limit or "```" in block or "~~~" in block or "$$" in block:
            result.append(block)
        else:
            remaining = block
            while len(remaining) > limit:
                target = len(remaining) // 2 if len(remaining) - limit < 120 else limit
                cut = max(remaining.rfind(' ', max(1, target // 2), target + 1), remaining.rfind('\n', max(1, target // 2), target + 1))
                cut = cut + 1 if cut > 0 else target
                result.append(remaining[:cut].strip())
                remaining = remaining[cut:]
            if remaining.strip():
                result.append(remaining.strip())
    return result


def markdown_sections(text):
    sections, lines, title, fence = [], [], "正文", None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1][0]
            fence = None if fence == token else token if fence is None else fence
        heading = re.match(r"^#{1,6}\s+(.+)$", line) if fence is None else None
        if heading:
            if "\n".join(lines).strip():
                sections.append((title, "\n".join(lines).strip()))
            title, lines = heading[1].strip(), []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((title, "\n".join(lines).strip()))
    return sections


SENTENCE_END = set("。！？；.!?;")
HEADING_NAMES = re.compile(
    r"^(摘要|abstract|引言|introduction|背景|background|相关工作|related work|"
    r"方法|method|methods|approach|实验|experiment|experiments|结果|results|"
    r"讨论|discussion|结论|conclusion|conclusions|参考文献|references|致谢|acknowledg)", re.I)
NUM_HEADING = re.compile(r"^\d+(\.\d+)*[\s、.]+[^\d\s]")
PAGE_HEADER = re.compile(
    r"(第\s*\d+\s*[卷期]|Vol\.|No\.|文章编号|中图分类号|文献标志码|DOI\s*[:：]|控\s*制\s*与\s*决\s*策|"
    r"Shock\s*and\s*Vibration|Hindawi|Article\s*ID|https?://)", re.I)
PAGE_MARKER = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")


def _is_heading(line):
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if line[-1] in "。！？；.!?;，,":
        return False
    if PAGE_HEADER.search(line):
        return False
    compact = re.sub(r"\s+", "", line)
    if re.match(
        r"^(摘要|abstract|引言|introduction|背景|background|相关工作|relatedwork|"
        r"方法|method|methods|approach|实验|experiment|experiments|结果|results|"
        r"讨论|discussion|结论|conclusion|conclusions|参考文献|references|致谢|acknowledg)"
        r"(?:[:：（(]|$)",
        compact, re.I):
        return True
    # 只认单级章节号（1. Introduction / 2 Method），不把 3.2 子标题和公式编号切出来。
    numbered = re.match(r"^(\d+)[\.\s、]+([A-Z一-鿿])", line)
    if numbered and 1 <= int(numbered[1]) <= 30 and not re.search(r"[()（），,]", line):
        return True
    # IEEE 常见罗马数字章节：I. INTRODUCTION / II. THE PROPOSED METHOD
    if re.match(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*[A-Z][A-Za-z]", line):
        return True
    return False


def _heading_role(title):
    value = re.sub(r"^\d+(\.\d+)*[\s、.]*", "", re.sub(r"\s+", "", title)).lower()
    if re.match(r"^(摘要|abstract)", value): return "abstract"
    if re.match(r"^(参考文献|references)", value): return "references"
    if re.match(r"^(引言|introduction|背景|background)", value): return "introduction"
    if re.match(r"^(方法|method|methods|approach|模型|framework|算法)", value): return "method"
    if re.match(r"^(实验|experiment|experiments|结果|results|evaluation|评估)", value): return "experiment"
    if re.match(r"^(结论|conclusion|conclusions|讨论|discussion)", value): return "conclusion"
    return "other"


def _join_line(prev, line):
    if not prev:
        return line
    if prev.endswith(("-", "—")):
        return prev[:-1] + line
    if re.search(r"[一-鿿]$", prev) or re.match(r"^[一-鿿]", line):
        return prev + line
    return prev + " " + line


def normalize_pdf_text(text):
    """把 PDF 的硬换行接回段落；标题行保持独立。"""
    text = text.replace("\x00", "").replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out, para = [], []

    def flush():
        nonlocal para
        if para:
            joined = para[0]
            for line in para[1:]:
                joined = _join_line(joined, line)
            out.append(joined.strip())
            para = []

    for line in lines:
        if not line:
            flush()
            continue
        inline = re.match(r"^(摘\s*要|Abstract)\s*[—\-–:：]\s*(.+)$", line, re.I)
        if inline:
            flush()
            out.append(re.sub(r"\s+", "", inline[1]))
            para.append(inline[2].strip())
            continue
        if _is_heading(line):
            flush()
            out.append(line)
        else:
            para.append(line)
    flush()
    return "\n".join(out)


def pdf_to_markdown(page_texts, title_hint=None):
    """PDF 文本 -> 带标题层级的 Markdown；保留页码注释供定位。"""
    out, title_seen, in_references = [], False, False
    for number, text in page_texts:
        body = normalize_pdf_text(text)
        if not body.strip():
            continue
        out.append(f"<!-- page: {number} -->")
        for line in body.splitlines():
            line = line.strip()
            if not line:
                out.append("")
                continue
            if in_references:
                out.append(line)
                continue
            if _is_heading(line):
                out.append(f"## {line}")
                title_seen = True
                in_references = _heading_role(line) == "references"
            elif not title_seen and number == 1:
                out.append(f"# {title_hint or line}")
                if title_hint and line != title_hint:
                    out.append(line)
                title_seen = True
            else:
                out.append(line)
        out.append("")
    return "\n".join(out).strip() + "\n"


def markdown_sections_roles(markdown):
    """按 #/## 标题切章节，返回 title/role/content/page。"""
    sections, current = [], {"title": "正文", "role": "other", "lines": [], "pages": []}
    active_page, fence = None, None
    def finish():
        content = '\n'.join(current['lines']).strip()
        if not content:
            return
        paragraphs = []
        offset = 0
        line_pages = []
        for line, page in zip(current['lines'], current['line_pages']):
            line_pages.append((offset, offset + len(line), page))
            offset += len(line) + 1
        trim = len('\n'.join(current['lines'])) - len('\n'.join(current['lines']).lstrip())
        for match in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', content):
            pages = sorted({p for start, end, p in line_pages if p and start < match.end() + trim and end > match.start() + trim})
            paragraphs.append({'number': len(paragraphs) + 1, 'start': match.start(), 'end': match.end(), 'pages': pages})
        sections.append({'title': current['title'], 'role': current['role'], 'content': content,
                         'page': current['pages'][0] if current['pages'] else None,
                         'pages': list(current['pages']), 'paragraphs': paragraphs})
    current['line_pages'] = []
    for line in markdown.splitlines():
        marker = PAGE_MARKER.match(line.strip())
        if marker:
            active_page = int(marker[1])
            continue
        token = re.match(r'^\s*(`{3,}|~{3,})', line)
        if token:
            fence = None if fence == token[1][0] else token[1][0] if fence is None else fence
        heading = re.match(r"^(#{1,6})\s+(.+)$", line) if fence is None and not token else None
        if heading:
            finish()
            level, title = heading[1], heading[2].strip()
            role = "title" if len(level) == 1 else _heading_role(title)
            display_title = "标题" if role == "title" else title
            current = {"title": display_title, "role": role,
                       "lines": [line] if role == "title" else [], "pages": [active_page] if active_page else [],
                       'line_pages': [active_page] if role == 'title' else []}
        else:
            current["lines"].append(line)
            current['line_pages'].append(active_page)
            if line.strip() and active_page and active_page not in current['pages']:
                current['pages'].append(active_page)
    finish()
    return [section for section in sections if section["content"].strip()]


def _cut_point(text, start, size):
    target = min(len(text), start + size)
    low, high = max(start + 1, target - 80), min(len(text), target + 80)
    for index in range(low, high):
        if text[index] in SENTENCE_END:
            return index + 1
    for index in range(high, low, -1):
        if text[index - 1].isspace():
            return index
    return target


def adaptive_chunks(text, role):
    """章节级优先：每章最多 5 段，目标约 1200 字；摘要 1 段，参考文献 2 段。

    450–550 太小、2000×3 太大，都会影响上下文；这里按章节长度自适应，
    短章不切，超长章节最多 5 段，单段约 1–2k 字。
    """
    text = text.strip()
    if not text:
        return []
    if role in {"title", "abstract"}:
        return [text]
    max_chunks = 2 if role == "references" else 5
    if len(text) <= 1500:
        return [text]
    count = min(max_chunks, max(1, math.ceil(len(text) / 1200)))
    size = math.ceil(len(text) / count)
    chunks, position = [], 0
    while position < len(text):
        if len(chunks) >= max_chunks - 1:
            chunks.append(text[position:].strip())
            break
        if len(text) - position <= size + 200:
            chunks.append(text[position:].strip())
            break
        cut = _cut_point(text, position, size)
        chunks.append(text[position:cut].strip())
        position = cut
    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) > 1 and len(chunks[-1]) < 200:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()
    return chunks


def markdown_parts(markdown):
    parts = []
    for section in markdown_sections_roles(markdown):
        content = section["content"]
        blocks = [(section["title"], section["role"], content)]
        # 没有识别到章节标题时，标题/摘要段可能吞掉全文；超过 3000 字就拆出正文。
        if section["role"] == "title" and len(content) > 3000:
            blocks = [("标题", "title", content[:1500]), ("正文", "other", content[1500:])]
        elif section["role"] == "abstract" and len(content) > 3000:
            blocks = [("摘要", "abstract", content[:2500]), ("正文", "other", content[2500:])]
        for title, role, body in blocks:
            if role == "other" and len(body) > 10000:
                chunks = split_text(body, limit=2000)
            else:
                chunks = adaptive_chunks(body, role)
            cursor = section['content'].find(body)
            for index, chunk in enumerate(chunks, 1):
                total = len(chunks)
                locator = f"{title} · 片段 {index}/{total}" if total > 1 else title
                start = section['content'].find(chunk, max(0, cursor))
                end = start + len(chunk)
                spans = [p for p in section.get('paragraphs', []) if p['start'] < end and p['end'] > start]
                pages = sorted({page for p in spans for page in p['pages']})
                paragraph_start = spans[0]['number'] if spans else None
                paragraph_end = spans[-1]['number'] if spans else None
                if paragraph_start:
                    locator = f"{title} · 解析第 {paragraph_start}" + (f"–{paragraph_end}" if paragraph_end != paragraph_start else '') + ' 段'
                cursor = max(cursor, end)
                parts.append({
                    "title": title, "chapter": title,
                    "structure_role": role, "content": chunk,
                    "page": pages[0] if pages else section.get("page"), "part": len(parts) + 1,
                    'pages': pages, 'paragraph_start': paragraph_start, 'paragraph_end': paragraph_end,
                    'paragraph_basis': 'parser_inferred',
                    'chapter_pages': section.get('pages', []),
                    "chapter_part": index, "chapter_size": total, "locator": locator,
                })
    return parts


def page_numbers(spec, page_count=MAX_PAGES):
    if not spec or not spec.strip():
        return None
    selected = set()
    for token in spec.replace('，', ',').split(','):
        match = re.fullmatch(r'\s*(\d+)(?:\s*-\s*(\d+))?\s*', token)
        if not match:
            raise ValueError('页码请使用 3-6,8 这样的格式。')
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= page_count:
            raise ValueError(f'页码范围必须在 1 到 {page_count} 之间，且起始页不大于结束页。')
        selected.update(range(start, end + 1))
    return sorted(selected)


def extract_pdf_pages(raw, pages_spec=''):
    """优先 pypdfium2（空格/标题恢复更好），失败回退 pypdf。"""
    try:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(io.BytesIO(raw))
        try:
            page_count = len(document)
            if page_count > MAX_PAGES:
                raise ValueError("单份 PDF 上限为 200 页，请分册导入。")
            selected = page_numbers(pages_spec, page_count)
            texts = []
            for index in range(page_count):
                number = index + 1
                if selected and number not in selected:
                    continue
                text = document[index].get_textpage().get_text_range() or ""
                texts.append((number, text))
            return page_count, selected, texts
        finally:
            document.close()
    except ValueError:
        raise
    except Exception:
        pass

    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，请先提供可读取的副本。")
    page_count = len(reader.pages)
    if page_count > MAX_PAGES:
        raise ValueError("单份 PDF 上限为 200 页，请分册导入。")
    selected = page_numbers(pages_spec, page_count)
    texts = []
    for number, page in enumerate(reader.pages, 1):
        if selected and number not in selected:
            continue
        texts.append((number, page.extract_text() or ""))
    return page_count, selected, texts


def parse_document(filename, raw, pages=''):
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".md", ".txt"}:
        raise ValueError("目前支持 PDF、Markdown（.md）和 UTF-8 TXT。")
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("文件不能为空，大小上限为 20 MB。")
    if pages.strip() and extension != '.pdf':
        raise ValueError('页码选择仅适用于 PDF。')
    parts, warnings, total_chars = [], [], 0
    markdown = ""

    def append(title, text, page=None):
        nonlocal total_chars
        total_chars += len(text)
        if total_chars > MAX_TEXT:
            raise ValueError("解析文字超过 100 万字符，请按章节拆分文件。")
        for index, content in enumerate(split_text(text), 1):
            if content.strip():
                parts.append({"title": title, "content": content, "page": page, "part": index,
                              "locator": f"第 {page} 页 · 片段 {index}" if page else f"{title} · 片段 {index}"})

    if extension == ".pdf":
        try:
            page_count, selected_pages, raw_pages = extract_pdf_pages(raw, pages)
            empty_pages, soup_pages, page_texts = [], [], []
            for number, text in raw_pages:
                text = (text or "").replace("\x00", "").strip()
                if not text:
                    empty_pages.append(number)
                elif formula_soup(text):
                    soup_pages.append(number)
                page_texts.append((number, text))
            try:
                from backend.infra.pdf_structure import layout_markdown
                markdown = layout_markdown(raw, selected_pages)
            except Exception:
                markdown = pdf_to_markdown(page_texts, Path(filename).stem)
            parts = markdown_parts(markdown)
            total_chars = sum(len(part["content"]) for part in parts)
            if total_chars > MAX_TEXT:
                raise ValueError("解析文字超过 100 万字符，请按章节拆分文件。")
            if len(re.findall(r"^## ", markdown, re.M)) <= 3:
                warnings.append("章节标题识别较少，部分正文已按长度分块；公式与双栏排版请查看原页。")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 解析失败，请检查文件是否损坏，或先转换为 Markdown。") from exc
        if empty_pages:
            warnings.append(f"{len(empty_pages)} 页未提取到文字；当前未接入扫描件 OCR。")
        if soup_pages:
            warnings.append(f"{len(soup_pages)} 页为公式版式，文字提取可能不可靠"
                            f"（{'、'.join(f'第 {n} 页' for n in soup_pages)}）；这些片段不会送入生成。")
        warnings.append("已提取文本供检索与分析；公式、表格和图片请核对原文件。段落编号由解析器生成，非出版社段落编号。")
    else:
        try:
            text = raw.decode("utf-8-sig").replace("\x00", "")
        except UnicodeDecodeError as exc:
            raise ValueError("文本编码不是 UTF-8，请转换编码后再上传。") from exc
        page_count = None
        if extension == ".md":
            markdown = text
            parts = markdown_parts(markdown)
            total_chars = sum(len(part["content"]) for part in parts)
            if total_chars > MAX_TEXT:
                raise ValueError("解析文字超过 100 万字符，请按章节拆分文件。")
        else:
            for title, content in [("正文", text)]:
                append(title, content)
    if not parts:
        raise ValueError("未解析到可检索文字。扫描 PDF 请先做 OCR，再导入文字版。")
    return {"format": extension[1:], "page_count": page_count, "selected_pages": selected_pages if extension == '.pdf' else None, "char_count": total_chars, "markdown": markdown, "parts": parts, "warnings": warnings}


def document_record(document_id):
    with Session(engine) as db:
        row = db.get(KnowledgeDocument, document_id)
        return dict(row.payload) if row else None


def preview_document(filename, raw, model_id, pages=''):
    from backend.infra.access import current_actor
    filename = safe_filename(filename)
    selected = page_numbers(pages)
    selection_key = (b'\0pages:' + ','.join(map(str, selected)).encode()) if selected else b''
    actor = current_actor.get()
    scope = actor.id.encode() + b'\0' if actor is not None and actor.id else b''
    document_id = hashlib.sha256(scope + model_id.encode() + b"\0" + raw + selection_key).hexdigest()[:32]
    existing = document_record(document_id)
    if existing:
        if existing.get('status') == 'archived':
            from backend.infra.document_versions import active_document
            existing = active_document(document_id)
        return {**existing, "duplicate": existing["status"] == "imported"}
    parsed = parse_document(filename, raw, pages)
    document = {**parsed, "id": document_id, "filename": filename, "model_id": model_id,
                "title": Path(filename).stem, "status": "preview", "cloud_allowed": False,
                "sha256": hashlib.sha256(raw).hexdigest(), "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    # 路径只来自服务端摘要和白名单后缀，客户端文件名不能指定存储位置。
    (UPLOAD_ROOT / f"{document_id}.{parsed['format']}").write_bytes(raw)
    if parsed.get("markdown") and parsed['format'] != 'md':
        (UPLOAD_ROOT / f"{document_id}.md").write_text(parsed["markdown"], encoding="utf-8")
    with Session(engine) as db:
        db.merge(KnowledgeDocument(id=document_id, payload=document))
        db.commit()
    return document


def commit_document(document_id, title, source_url="", cloud_allowed=False):
    if source_url and (urlparse(source_url).scheme not in {"http", "https"} or not urlparse(source_url).hostname):
        raise ValueError("原始来源地址需为有效的 http 或 https 链接。")
    with Session(engine) as db:
        row = db.get(KnowledgeDocument, document_id)
        if row is None:
            raise LookupError("预览记录不存在，请重新上传。")
        document = dict(row.payload)
        if document["status"] == "imported":
            return {**document, "duplicate": True}
        document.update(title=title.strip() or document["title"], source_url=source_url, cloud_allowed=cloud_allowed,
                        status="imported", imported_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        from backend.infra.document_versions import activate_revision, PROCESSING_POLICY
        activate_revision(db, document, document['imported_at'])
        document.setdefault('retrieval_added_at', document['imported_at'])
        document.setdefault('retrieval_priority', 1.0)
        document.setdefault('revision_number', 1)
        document.setdefault('processing_policy', PROCESSING_POLICY)
        sid = f"D{document_id}"
        source = {"id": sid, "title": document["title"], "topic": document["model_id"],
                  "url": f"/documents/{document_id}", "original_url": source_url, "kind": "document",
                  "scope": "已导入原文文字，保留页码/章节定位；内容未自动核验。", "status": "原文文字已入库",
                  "checked_at": document["imported_at"][:10]}
        db.merge(KnowledgeSource(id=sid, payload=source))
        for order, part in enumerate(document["parts"]):
            slug = f"doc-{document_id}-{order}"
            section = {**part, "id": f"doc:{document_id}:{order}", "slug": slug, "model_id": document["model_id"],
                       "title": f"{document['title']} · {part['locator']}", "order": 1000 + order,
                       "document_id": document_id, "origin": "document", "cloud_allowed": cloud_allowed,
                       "source_ids": [sid], "reader_url": f"/documents/{document_id}?part={slug}",
                       'original_file_url': f"/api/documents/{document_id}/file?inline=true" + (f"#page={part['page']}" if part.get('page') else '')}
            db.merge(KnowledgeSection(id=section["id"], model_id=document["model_id"], payload=section))
        row.payload = document
        db.commit()
    return document
