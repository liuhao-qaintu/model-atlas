"""从用户问法继承来源范围；与LLM查询改写分离，不生成模型知识。"""
import re
from urllib.parse import urlparse

VERSION = 'official-scope-parent-v4'

IMPLEMENTATION = r'torchvision|pytorch|(?a:\btorch\.|\bnn\.|\bforward\b|\bidentity\b)|官方(?:实现|源码|接口|模块)|源码'
PAPER = r'原论文|论文|\bpaper\b'
STRICT = (r'(?:只|仅)(?:使用|用|看|采用|沿用|依据|参考|根据)?\s*'
          r'(?:(?:已入库|当前库中|本知识库|提供|固定版本)的?)?\s*'
          r'(?:(?:PyTorch|Torchvision)\s*(?:v?\d+(?:\.\d+){0,2})?\s*的?)?\s*官方')


def resolve_source_scope(question, model_ids, previous=(), followup=False):
    last = next((m.get('metadata', {}).get('plan', {}) for m in reversed(previous) if m.get('role') == 'assistant'), {})
    same_context = followup and bool(model_ids) and set(model_ids) == set(last.get('model_ids', []))
    inherited = last.get('source_scope') if same_context else None
    if same_context and inherited is None:
        # 升级前的会话从真实用户原话恢复，模型旧答案不参与范围推断。
        for message in reversed(previous[-6:]):
            if message.get('role') == 'user':
                inferred = resolve_source_scope(message['content'], model_ids)
                if inferred['kind'] != 'general':
                    inherited = inferred; break
    implementation = bool(re.search(IMPLEMENTATION, question, re.I))
    paper_text = re.sub(r'(?:不看|不用|无需|不要|不需要|不参考|排除)(?:原)?论文', '', question)
    paper = bool(re.search(PAPER, paper_text, re.I))
    strict = bool(re.search(STRICT, question))
    if implementation and paper:
        return {'kind': 'mixed', 'label': '论文与实现分别核对', 'origin': 'question', 'strict': False}
    if paper and not implementation:
        return {'kind': 'paper', 'label': '论文表述范围', 'origin': 'question', 'strict': False}
    if implementation or strict:
        framework = 'Torchvision' if re.search('torchvision', question, re.I) else 'PyTorch' if re.search(r'pytorch|(?a:\btorch\.|\bnn\.)', question, re.I) else None
        origin = 'question'
        if inherited and inherited['kind'] == 'implementation':
            strict = strict or inherited.get('strict', False)
            if not framework:
                framework = inherited.get('framework'); origin = 'question_and_last_turn'
        scope = {'kind': 'implementation', 'label': (framework or '实现与教程') + '接口范围', 'framework': framework,
                 'origin': origin, 'strict': strict}
        if strict:
            targets = list(dict.fromkeys(re.findall(r'(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*)', question)))
            version = re.search(r'(?:PyTorch|Torchvision)\s*v?(\d+\.\d+\.\d+)(?![\d.])', question, re.I)
            if targets:
                scope['api_targets'] = targets
                scope['api_version'] = version[1] if version else (inherited or {}).get('api_version')
            elif inherited and inherited.get('api_targets'):
                scope.update(api_targets=inherited['api_targets'], api_version=version[1] if version else inherited.get('api_version'))
        return scope
    if inherited and inherited['kind'] != 'general': return {**inherited, 'origin': 'last_turn'}
    return {'kind': 'general', 'label': '综合资料范围', 'origin': 'question', 'strict': False}


def strict_method_sections(scope, retriever):
    """明确限定类方法时按已核验来源的章节定位筛选；空集合表示缺资料，不能放宽。"""
    if scope.get('cloud_only'):
        documents = {d['id']: d for d in retriever.documents}
        return {s['id'] for s in retriever.sections if s.get('cloud_allowed', not bool(s.get('document_id')))
                and (not s.get('document_id') or documents.get(s['document_id'], {}).get('cloud_allowed', False))}
    if scope.get('document_ids'):
        return {s['id'] for s in retriever.sections if s.get('document_id') in scope['document_ids']}
    if scope.get('document_id'):
        return {s['id'] for s in retriever.sections if s.get('document_id') == scope['document_id']}
    targets = scope.get('api_targets') if scope.get('strict') else None
    if not targets:
        return None
    documents = {doc['id']: doc for doc in retriever.documents}
    version = scope.get('api_version')
    allowed = set()
    for section in retriever.sections:
        did = section.get('document_id')
        if retriever.document_kinds.get(did) != 'official_code':
            continue
        # 不能用正文中恰好提到某方法，冒充该方法的实际章节。
        if section.get('locator', '').split(' · ', 1)[0] not in targets:
            continue
        source_path = urlparse(documents.get(did, {}).get('source_url') or '').path
        if version and not re.search(r'/(?:v)?' + re.escape(version) + r'(?:/|$)', source_path):
            continue
        allowed.add(section['id'])
    return allowed


def source_filters(scope):
    if scope['kind'] == 'implementation':
        return {'source_kinds': ['official_code'] if scope.get('strict') else [], 'excluded_source_kinds': ['paper']}
    return {'source_kinds': [], 'excluded_source_kinds': []}


def source_type(item, retriever):
    kind = getattr(retriever, 'document_kinds', {}).get(item.get('document_id'))
    labels = {'paper': '论文原文', 'official_code': '已核对版本的官方源码', 'official_tutorial': '官方教程', 'official_document': '官方文档'}
    if kind in labels: return labels[kind]
    return '整理模型卡（含教学示例）' if item.get('origin', 'model_card') == 'model_card' and not item.get('document_id') else '导入资料（类型未核验）'
