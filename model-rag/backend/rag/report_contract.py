"""按资料类型组织报告；类型判断的来源和证据随报告保存。"""
import re

ROLES = {'baseline', 'proposed', 'background', 'reviewed', 'teaching', 'interface', 'document', 'unknown'}
COMMON_FIELDS = {'summary': '文章概述', 'contributions': '资料要点', 'methods': '方法与数据流',
                 'formula_explanations': '公式与条件', 'limitations': '适用范围与待核实内容'}
TYPES = {
    'research': ('研究论文', {'contributions': '核心贡献', 'methods': '基础方法与本文改动'},
                 '先说明基础方法，再说明本文改变什么；贡献只写原文明示的新增原则、方法或证据，不把回顾内容归为首创。'),
    'review': ('综述', {'contributions': '综述组织与归纳价值', 'methods': '方法类别与比较'},
               '解释综述覆盖的问题、分类和比较；所述模型是被回顾对象，不写成本文提出或实际部署。'),
    'tutorial': ('教程', {'summary': '学习目标', 'contributions': '学习重点', 'methods': '实现步骤与数据流'},
                 '解释学习目标、步骤、必要条件及示例配置；不制造学术创新，示例参数不代表接口统一约束。'),
    'reference': ('接口与源码资料', {'summary': '对象与版本', 'contributions': '接口职责', 'methods': '调用、输入与返回'},
                  '说明正文给出的版本、对象、调用及返回；不根据调用猜未提供的模块内部实现，不制造论文贡献。'),
    'unknown': ('资料', {}, '类型未确认，按正文陈述主题和要点，不预设存在原创贡献。'),
}


def profile_for(kind='unknown', origin='unconfirmed', proofs=()):
    kind = kind if kind in TYPES else 'unknown'
    label, fields, instruction = TYPES[kind]
    return {'kind': kind, 'label': label, 'origin': origin, 'proofs': list(proofs),
            'fields': {**COMMON_FIELDS, **fields}, 'instruction': instruction}


def metadata_hint(document, sections):
    title = document.get('title', '')
    front = '\n'.join(s.get('content', '') for s in sections[:2])[:2500]
    if re.search(r'\breview\b|\bsurvey\b|综述', title, re.I) or re.search(r'^\s*(?:Review Article|综述论文)\s*$', front, re.I | re.M):
        return 'review'
    if re.search(r'教程|\btutorial\b', title, re.I):
        return 'tutorial'
    if re.search(r'源码|源代码|官方实现|接口', title):
        return 'reference'
    return 'unknown'


def finding_scope(item):
    role = item.get('role', 'unknown')
    return {'role': role if role in ROLES else 'unknown', 'subject': str(item.get('subject') or '').strip()[:80]}


def scope_key(item):
    value = finding_scope(item)
    return value['role'], value['subject'].casefold()
