"""严格来源下的保守越界检查；只拦截明确模式，不冒充语义证明。"""
import re

from backend.generation.answer_validation import segments

VERSION = 'strict-source-guard-v2'
HEAD_DEFINITION = re.compile(r'self\.(?:fc|classifier|head|heads|lm_head)\s*\(|\b(?:num_classes|vocab_size|logits)\b', re.I)
HEAD_TERM = re.compile(r'分类头|全连接|线性层|投影头|词表投影|\b(?:nn\.)?Linear\b', re.I)
PROPOSE = re.compile(r'(?:需要|必须|应当|需用户|要将|若要|要获得|要得到)[^。！？\n]{0,160}(?:添加|连接|接入|接一个|接上|应用|使用|映射|经过|线性层|投影头|分类头)|(?:额外|再|之后|后续)[^。！？\n]{0,60}(?:添加|连接|接入|接一个|应用)', re.I)
NEGATED_PROPOSAL = re.compile(r'不能|不应|不要|不得|不可|不代表|不意味着|不表示|不一定|(?:不|无|没有)\s*$')
FEATURE_LOGITS = re.compile(r'(?:输出(?:的)?(?:是|为)|称为|叫作)[^。\n]{0,70}(?<![A-Za-z0-9_])logits(?![A-Za-z0-9_])', re.I)


def inspect_strict_answer(answer, evidence, scope):
    result = {'version': VERSION, 'status': 'not_required', 'rules': []}
    if not scope.get('strict'):
        return result
    body = '\n'.join(item.get('generation_excerpt', item['content']) for item in evidence)
    result['status'] = 'no_pattern_detected'
    # Linear 也可能只是隐藏层，不能单凭出现它就认定已提供任务头。
    if HEAD_DEFINITION.search(body):
        return result
    prose = '\n'.join(part for protected, part in segments(answer) if not protected)
    for paragraph in re.split(r'\n\s*\n', answer):
        proposals = [match for match in PROPOSE.finditer(paragraph)
                     if not NEGATED_PROPOSAL.search(paragraph[max(0, match.start() - 8):match.start()])]
        if HEAD_TERM.search(paragraph) and proposals:
            result['rules'].append('unprovided_task_head_proposal')
            break
    for match in FEATURE_LOGITS.finditer(prose):
        context = prose[max(0, match.start() - 6):match.end()]
        if not re.search(r'不是|并非|不能|不应|不宜|不等于', context):
            result['rules'].append('unsupported_feature_logits_equivalence')
            break
    if result['rules']:
        result['status'] = 'blocked'
    return result


def official_excerpt_answer(evidence):
    lines = ['生成解读包含本轮官方正文未支持的延伸，未采用该解读。以下为实际检索到的官方原文节选；可从回答依据打开完整位置。']
    fence = '`' * 8
    for number, item in enumerate(evidence, 1):
        text = item.get('generation_excerpt', item['content'])
        excerpt = text[:600]
        lines.append(f"[{number}] {item['title']}\n\n{fence}text\n{excerpt}\n{fence}" + ('\n（节选，后续内容请打开原文。）' if len(text) > 600 else ''))
    return '\n\n'.join(lines)
