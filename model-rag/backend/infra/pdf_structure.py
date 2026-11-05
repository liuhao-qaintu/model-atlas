"""用已有 PDFium 的字体、缩进恢复标题和阅读段落；不重建公式或插图。"""
import ctypes
import io
import re

NUMBERED = re.compile(r'^(\d+(?:\.\d+)*\.)\s+(.+)$')
ROMAN = re.compile(r'^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z]')
FOOTER = re.compile(r'^(?:\d+\s+)?(?:Shock and Vibration|Hindawi)(?:\s+\d+)?$', re.I)


def _font(pdfium, textpage, position):
    index = pdfium.raw.FPDFText_GetCharIndexFromTextIndex(textpage, position)
    if index < 0:
        return '', (0, 0, 0, 0)
    buffer, flags = ctypes.create_string_buffer(256), ctypes.c_int()
    pdfium.raw.FPDFText_GetFontInfo(textpage, index, buffer, 256, ctypes.byref(flags))
    return buffer.value.decode('utf-8', 'replace'), textpage.get_charbox(index)


def _lines(pdfium, page):
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_range()
        result = []
        for match in re.finditer(r'[^\r\n]+', text):
            value = match[0].strip()
            if not value or FOOTER.fullmatch(value):
                continue
            offset = match.start() + len(match[0]) - len(match[0].lstrip())
            font, box = _font(pdfium, textpage, offset)
            row = {'text': value, 'font': font, 'x': box[0], 'y': box[1], 'height': box[3] - box[1]}
            # 期刊常把三级标题和正文排在同一行，用字体变化分开，保留原文字符。
            if NUMBERED.match(value) and re.search(r'It|Italic|Bold|Black', font, re.I):
                for word in re.finditer(r'\S+', value):
                    if word.start() < 10:
                        continue
                    next_font, _ = _font(pdfium, textpage, offset + word.start())
                    if next_font and next_font != font and value[:word.start()].rstrip().endswith('.'):
                        row['text'] = value[:word.start()].rstrip()
                        row['tail'] = value[word.start():]
                        break
            result.append(row)
        return result
    finally:
        textpage.close()


def layout_markdown(raw, selected_pages=None):
    """标题位置依赖可读取字体；失败让调用方使用纯文本回退。"""
    import pypdfium2 as pdfium
    output, in_references = [], False
    with pdfium.PdfDocument(io.BytesIO(raw)) as document:
        for index in range(len(document)):
            number = index + 1
            if selected_pages and number not in selected_pages:
                continue
            page = document[index]
            try:
                rows = _lines(pdfium, page)
            finally:
                page.close()
            output.append(f'<!-- page: {number} -->')
            previous = None
            position = 0
            while position < len(rows):
                row = rows[position]
                value, font = row['text'], row['font']
                numbered = NUMBERED.match(value)
                styled = bool(re.search(r'Bold|Black|Italic|(?:-|\b)It\b', font, re.I))
                heading = not in_references and bool((numbered and int(numbered[1].split('.')[0]) <= 30 and (styled or len(value) < 90))
                               or ROMAN.match(value)
                               or (value.lower() in {'references', 'acknowledgements', '结论', '参考文献'})
                               or (number == 1 and value.lower() in {'abstract', '摘要'}))
                if heading:
                    in_references = value.lower() in {'references', '参考文献'}
                    level = min(5, len(numbered[1].rstrip('.').split('.')) + 1) if numbered else 2
                    # 只合并相同标题字体的续行，不靠“大写开头”吞掉正文。
                    if not row.get('tail') and styled:
                        while position + 1 < len(rows):
                            following = rows[position + 1]
                            if (following['font'] != font or NUMBERED.match(following['text'])
                                    or value.endswith('.') or re.match(r'^\(\d+\)', following['text'])
                                    or abs(following['y'] - rows[position]['y']) > 25):
                                break
                            value += ' ' + following['text']
                            position += 1
                    if not row.get('tail') and not value.endswith('.') and position + 1 < len(rows):
                        following = rows[position + 1]
                        continuation = re.match(r'^([A-Z][A-Za-z -]{0,65}\.)\s+(.+)$', following['text'])
                        if numbered and continuation and re.search(r'It|Italic', following['font']):
                            value += ' ' + continuation[1]
                            row['tail'] = continuation[2]
                            position += 1
                    inline = re.search(r'\.\s+(?=\(e\b|The\b|In\b|Based\b|Researchers\b|CNN\b|RNN\b|Using\b|Autoencoder\b|Convolutional\b|Recurrent\b)', value)
                    if numbered and inline and inline.start() > len(numbered[1]) + 5:
                        row['tail'] = value[inline.end():]
                        value = value[:inline.start() + 1]
                    output.extend(['', '#' * level + ' ' + value, ''])
                    if row.get('tail'):
                        output.extend([row['tail'], ''])
                    previous = None
                else:
                    # PDFium 的 CR/LF 不是原始段落；缩进/较大行距给出可复查的解析段落。
                    if previous:
                        gap = previous['y'] - row['y']
                        indent = row['x'] - previous['x']
                        if gap < -20 or gap > 19 or 5 < indent < 45:
                            output.append('')
                    output.append(value)
                    previous = row
                position += 1
            output.extend(['', ''])
    # 普通硬换行在段内合并，空行/标题和页码保留。
    normalized, paragraph = [], []
    def flush():
        if paragraph:
            normalized.append(' '.join(paragraph).replace('\ufffe', '').strip())
            paragraph.clear()
    for line in output:
        if not line or line.startswith(('#', '<!-- page:')):
            flush()
            normalized.append(line)
        else:
            paragraph.append(line)
    flush()
    return '\n'.join(normalized).strip() + '\n'
