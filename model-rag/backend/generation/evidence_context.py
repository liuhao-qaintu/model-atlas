"""选择真实正文节选；长Python源码按完整函数定位，不执行代码、不推断函数体。"""
import ast
import re

from backend.infra.text_quality import formula_soup
from backend.rag.source_scope import source_type
from backend.generation.grounding_contract import evidence_boundary


def python_units(content):
    units = []
    for block in re.finditer(r'(?m)^```(?:python|py)\s*\n([\s\S]*?)^```\s*$', content):
        code = block[1]
        try: tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError): continue
        offsets = [0]
        for line in code.splitlines(keepends=True): offsets.append(offsets[-1] + len(line))
        def collect(nodes, owner=''):
            for node in nodes:
                if isinstance(node, ast.ClassDef): collect(node.body, node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                    start, end = block.start(1) + offsets[first - 1], block.start(1) + offsets[node.end_lineno]
                    units.append({'start': start, 'end': end, 'name': f'{owner}.{node.name}' if owner else node.name, 'owner': owner, 'method': node.name})
        collect(tree.body)
    return units


def select_excerpt(content, query, kind=None, budget=4500):
    if len(content) <= budget: return {'strategy': 'full', 'ranges': [{'start': 0, 'end': len(content)}], 'text': content}
    if kind == 'official_code':
        words = set(re.findall(r'[a-z_][a-z0-9_]*', query.lower()))
        compact = re.sub(r'[^a-z0-9]', '', query.lower())
        io_intent = bool(re.search(r'输入|输出|返回|特征|分类头|\bforward\b|\bfc\b', query, re.I))
        candidates = []
        for unit in python_units(content):
            name = re.sub(r'[^a-z0-9]', '', unit['method'].lower())
            owner = re.sub(r'[^a-z0-9]', '', unit['owner'].lower())
            score = 24 * bool(name and name in compact) + 12 * bool(owner and owner in compact)
            score += 20 * bool(io_intent and unit['method'] in {'forward', '_forward_impl', 'forward_features', '__call__'})
            body_words = set(re.findall(r'[a-z_][a-z0-9_]*', content[unit['start']:unit['end']].lower()))
            score += 2 * len(words & body_words)
            candidates.append((score, unit))
        chosen, size = [], 0
        for score, unit in sorted(candidates, key=lambda pair: (-pair[0], pair[1]['start'])):
            cost = unit['end'] - unit['start'] + len(unit['name']) + 60
            if score > 0 and size + cost <= budget:
                chosen.append(unit); size += cost
        if chosen:
            chosen.sort(key=lambda unit: unit['start'])
            text = '\n\n'.join(f"原文代码单元：{u['name']}\n```python\n{content[u['start']:u['end']]}```" for u in chosen)
            return {'strategy': 'python_units', 'ranges': [{k: u[k] for k in ('start', 'end', 'name')} for u in chosen], 'text': text}
    # 非源码、无法解析或函数过长时保留旧的有界行为，不伪造完整定义。
    return {'strategy': 'head', 'ranges': [{'start': 0, 'end': budget}], 'text': content[:budget]}


def prepare_evidence(evidence, retriever, query):
    prepared = []
    kinds = getattr(retriever, 'document_kinds', {})
    for item in evidence:
        excerpt = select_excerpt(item['content'], query, kinds.get(item.get('document_id')))
        if item.get('document_id') and formula_soup(item['content']):
            # 公式版式的 PDF 文字不可靠：保留原文供查看，但不把符号堆送给生成模型。
            locator = item.get('locator') or item['title']
            text = (f'（该片段位于{locator}，是公式版式；PDF 提取的文字不可靠，已拦截，未提供给生成。'
                    f'回答时只能说明公式细节需查看原文对应页，不得复述本片段中的符号。）')
            selection = {'strategy': 'formula_gate', 'ranges': []}
        else:
            text, selection = excerpt['text'], {key: excerpt[key] for key in ('strategy', 'ranges')}
        prepared.append({**item, 'source_type': source_type(item, retriever), 'generation_excerpt': text,
                         'generation_selection': selection, 'claim_scope': evidence_boundary(item, retriever)})
    return prepared


def context_text(evidence):
    return '\n\n'.join(f"[证据 {i}] {e['title']}\n来源类型：{e['source_type']}\n表述范围：{e.get('claim_scope', {})}\n范围角色：{e.get('scope_role', '当前检索资料')}\n定位：{e.get('locator') or '整理章节'}\n正文节选策略：{e['generation_selection']['strategy']}\n{e['generation_excerpt']}" for i, e in enumerate(evidence, 1))
