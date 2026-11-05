"""判断 PDF 公式版式被抽取后是否已变成不可读的符号堆；只做拦截依据，不改写原文。"""


def _math_symbol(ch):
    code = ord(ch)
    return 0x1D400 <= code <= 0x1D7FF or 0x0370 <= code <= 0x03FF


def formula_soup(text):
    """公式版式的文字提取通常同时出现：数学字母足够多，且制表符多或数学字母占比高。

    中文资料和代码不会命中：前者 CJK 占比高，后者没有数学字母。
    阈值来自 2026-10-31 对 26 份文档、304 个片段的扫描（命中 9 个，均为英文论文公式页）。
    """
    if not text:
        return False
    math = sum(1 for ch in text if _math_symbol(ch))
    if math < 20:
        return False
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    if cjk / len(text) >= 0.2:
        return False
    alpha = sum(1 for ch in text if ch.isalpha()) or 1
    return text.count('\t') >= 4 or math / alpha >= 0.05
