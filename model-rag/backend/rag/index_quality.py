"""只整理检索索引；原始章节、下载文件和既有引用位置保持可读。"""
import ast
import re
import textwrap

POLICY = 'source-quality-v2'


def exclusion_reason(section, kind, previous=None, is_last=False):
    text = section['content'].strip()
    if kind == 'official_code' and '```' not in text and '~~~' not in text:
        paragraphs = re.split(r'\n\s*\n', text)
        if paragraphs and all(re.match(r'^(?:来源[：:]|源码定位[：:]|固定版本[：:]|版本[：:]|原始文件[：:]|本文件为|版权与许可|以下签名和执行代码来自所列原文范围)', p) for p in paragraphs):
            return 'source_metadata_only'
    # 旧解析器在1800字符处硬切长段落；只识别有前块可追溯的短尾，不泛化为短文过滤。
    if (section.get('page') and section.get('part', 0) > 1 and previous and is_last
            and len(previous['content']) == 1800 and len(text) < 100
            and re.search(r'[A-Za-z]$', previous['content']) and re.match(r'[a-z]', text)
            and re.search(r'\n\d+\s*$', text)):
        return 'legacy_pdf_split_tail'
    return None


def bounded_windows(text, limit=1800):
    """按行索引长函数；窗口不宣称是完整函数，生成阶段仍引用所属完整正文。"""
    lines = text.splitlines(keepends=True)
    chunks, current = [], ''
    for line in lines:
        if len(line) > limit:
            if current:
                chunks.append(current.rstrip()); current = ''
            chunks.extend(line[i:i + limit].rstrip() for i in range(0, len(line), limit))
        elif current and len(current) + len(line) > limit:
            chunks.append(current.rstrip()); current = line
        else:
            current += line
    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def python_units(text, limit=1800):
    """静态读取Python围栏中的定义；不执行源文件，不重写函数内容。"""
    output, cursor = [], 0
    for match in re.finditer(r'(?ms)^```python[^\n]*\n(.*?)^```[ \t]*$', text):
        outside = text[cursor:match.start()].strip()
        if outside:
            output.append({'content': outside, 'symbol': None, 'kind': 'prose'})
        code = textwrap.dedent(match[1]).rstrip()
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            output.extend({'content': piece, 'symbol': None, 'kind': 'code_window'} for piece in bounded_windows(code, limit))
            cursor = match.end(); continue
        lines = code.splitlines()
        units = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if methods:
                    # 类头、文档与类属性也保留为单独索引，避免只保留方法而丢失接口条件。
                    method_lines = {i for method in methods for i in range(min([method.lineno, *[d.lineno for d in method.decorator_list]]), method.end_lineno + 1)}
                    header = '\n'.join(line for i, line in enumerate(lines[node.lineno - 1:node.end_lineno], node.lineno) if i not in method_lines)
                    units.append((node.name, header))
                    units.extend((f'{node.name}.{m.name}', '\n'.join(lines[min([m.lineno, *[d.lineno for d in m.decorator_list]]) - 1:m.end_lineno])) for m in methods)
                else:
                    units.append((node.name, '\n'.join(lines[node.lineno - 1:node.end_lineno])))
            else:
                units.append((getattr(node, 'name', None), '\n'.join(lines[node.lineno - 1:node.end_lineno])))
        for symbol, body in units:
            output.extend({'content': piece, 'symbol': symbol, 'kind': 'function_window' if symbol else 'code_window'} for piece in bounded_windows(body, limit) if piece.strip())
        cursor = match.end()
    if cursor == 0:
        return None
    if text[cursor:].strip():
        output.append({'content': text[cursor:].strip(), 'symbol': None, 'kind': 'prose'})
    return output
