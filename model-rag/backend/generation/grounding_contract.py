"""跨聊天、研究与报告共用的表述边界；不内置模型题目的标准答案。"""
import re
from backend.generation.answer_validation import segments

SCOPED_EXPLANATION = '''按引文的实际对象和条件解释：方法家族、本文改动、基础/相关方法、教学配置、固定版本接口分别归属。示例的损失、维数、激活、约束只写“本例/本文采用”，不能推广为家族必需或唯一选择。变量沿用各自来源的定义，不能把同名字母当成同一量；解释公式须先说明对象、输入、目标及假设。重构输入是任务目标，不等于有用表示必然是恒等复制；经验动机不写成必然保证。只读到节选时用“本轮资料未提供”，不能断言全文/所有实验完全没有。引用必须支持紧邻句子的全部事实。'''


def qualify_absence_scope(answer):
    """节选问答不冒称全文排查；仅收窄否定句主语，不补写知识或改动引文。"""
    count = 0
    pattern = re.compile(r'(?:本文|该论文|该文章|全文|整篇论文)(?:[（(][^）)\n]{0,60}[）)])?'
                         r'(?P<cites>(?:\[\d+\])*)\s*(?:均|都)?(?=(?:未涉及|未提及|没有(?:任何)?|不存在))')
    def narrow(part):
        def apply_piece(piece):
            def replace(match):
                nonlocal count
                before = piece[max(0, match.start()-16):match.start()]
                if re.search(r'(?:如果|假如|假设|不能断言|无法断定|不代表)\s*$', before):
                    return match[0]
                count += 1
                return '本轮所给正文节选' + match['cites']
            return pattern.sub(replace, piece)
        # 保留显式引语，避免改变作者原话或讨论中的示例句。
        pieces = re.split(r'(“[^”]*”|‘[^’]*’|"[^"\n]*")', part)
        return ''.join(piece if i % 2 else apply_piece(piece) for i, piece in enumerate(pieces))
    result = ''.join(part if protected else narrow(part) for protected, part in segments(answer))
    return result, {'rule': 'excerpt_absence_scope_v1', 'qualified_claims': count}


def evidence_boundary(item, retriever):
    kind = getattr(retriever, 'document_kinds', {}).get(item.get('document_id'))
    if not item.get('document_id'):
        role = '整理笔记与教学配置；本例参数不代表模型家族统一规格'
    elif kind == 'official_code':
        role = '固定版本官方实现；只说明已提供的定义与调用'
    elif kind in {'official_tutorial', 'official_document'}:
        role = '官方教程或接口说明；区分示例参数与接口约束'
    else:
        role = '文献正文；区分本文方法与所回顾的基础或相关方法'
    return {'role': role, 'coverage': '本轮实际正文节选；不代表全文已经查尽',
            'locator': item.get('locator') or item.get('title', '')}
