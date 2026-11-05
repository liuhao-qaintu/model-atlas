"""引用和链接检查只处理正文，保留代码索引、调用及公式。"""
import re

PROTECTED = re.compile(r'(?P<fence>`{3,}|~{3,})[\s\S]*?(?P=fence)|(?P<ticks>`{1,2})[^\n]*?(?P=ticks)|\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])*\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]')
CITATION = re.compile(r'(?<![A-Za-z0-9_.])\[(\d+)\]')


def segments(text):
    start = 0
    for match in PROTECTED.finditer(text):
        yield False, text[start:match.start()]
        yield True, match[0]
        start = match.end()
    yield False, text[start:]


def citation_numbers(text):
    return [int(n) for protected, part in segments(text) if not protected for n in CITATION.findall(part)]


def canonical_citations(text):
    """将中文和合并编号规范为逐项引用，保护代码、公式和明确的数值形状。"""
    pattern = r'\[证据\s*(\d+(?:\s*[,，、]\s*(?:证据\s*)?\d+)*)\s*\]'
    def normalize(part):
        part = re.sub(pattern, lambda match: ''.join(f'[{n}]' for n in re.findall(r'\d+', match[1])), part)
        part = re.sub(r'证据\s*(\d+)', lambda match: f'[{match[1]}]', part)
        group = r'(?<![A-Za-z0-9_.])\[(\d+(?:\s*[,，、]\s*\d+)+)\](?!\s*\()'
        def split_group(match):
            numbers = [int(n) for n in re.findall(r'\d+', match[1])]
            # 引用从1起；含0的数值区间/数组不能拆成引用再删去0。
            if 0 in numbers:
                return match[0]
            rest = part[match.end():]
            if rest.startswith('[') and not re.match(r'\[\d+(?:\s*[,，、]\s*\d+)*\]', rest):
                return match[0]
            before = part[max(0, match.start() - 32):match.start()]
            if re.search(r'(?:形状|尺寸|维度|坐标|向量|数组|列表|区间|范围|映射到|取值|shape|size|tensor)\s*(?:为|是|在|[:：=])?\s*$', before, re.I):
                return match[0]
            return ''.join(f'[{n}]' for n in dict.fromkeys(re.findall(r'\d+', match[1])))
        return re.sub(group, split_group, part)
    return ''.join(part if protected else normalize(part) for protected, part in segments(text))


def validate_citations(answer, evidence_count):
    answer = canonical_citations(answer)
    invalid = any(n < 1 or n > evidence_count for n in citation_numbers(answer))
    result = ''.join(part if protected else CITATION.sub(lambda m: m[0] if 1 <= int(m[1]) <= evidence_count else '', part) for protected, part in segments(answer))
    return result, invalid


def keep_known_links(answer, allowed_urls):
    dropped = False
    def replace(match):
        nonlocal dropped
        if match[2] in allowed_urls:
            return match[0]
        dropped = True
        return match[1]
    result = ''.join(part if protected else re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', replace, part) for protected, part in segments(answer))
    return result, dropped
