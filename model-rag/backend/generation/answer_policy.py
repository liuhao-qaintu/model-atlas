"""按用户的问题选择解释深度；不把增加字数当作正确性校验。"""
import re

VERSION = 'learning-depth-v3'
BRIEF = re.compile(r'一句话|请简短|简短回答|简单回答|只说结论|不展开|\bbriefly\b', re.I)
DEEP = re.compile(r'详细|逐层|逐步|推导|手算|为什么|如何|怎么改|怎样改|排查|排错|反例|梯度|感受野|参数量|teacher.forcing|交叉熵|padding.*损失|计算.*(?:长度|形状)|explain why|derive|debug', re.I)
CALCULATION = re.compile(r'手算|参数量|感受野|逐层|计算|推导.*(?:形状|尺寸|长度|参数)', re.I)


def presentation_instruction(question):
    if CALCULATION.search(question):
        return '计算题按本题条件、公式代入、中间步骤、最终结果的顺序回答；结果集中写一次，不在推导前另写数值摘要，也不在结尾重复整套步骤。'
    return '按实际子问题组织；每个问题将结论、条件与原因连起来解释，不另设重复摘要或末尾总结。'


def answer_profile(question, mode='learning'):
    requested = re.sub(r'(?:不要|不必|无需|别)\s*(?:再|只用|只|用)?\s*(?:' + BRIEF.pattern + ')', '', question, flags=re.I)
    if BRIEF.search(requested):
        return {'level': 'concise', 'max_tokens': 800, 'reason': '用户要求简短'}
    if DEEP.search(question) or question.count('？') + question.count('?') >= 2:
        return {'level': 'detailed', 'max_tokens': 3200, 'reason': '需要机制、推导、实现或多个子问题'}
    return {'level': 'standard', 'max_tokens': 1800 if mode == 'research' else 1600, 'reason': '直接解释当前问题'}


def depth_instruction(profile):
    if profile['level'] == 'concise':
        return '用户要求简短：先给直接结论和不可省略的条件，不扩展教学章节；简短不能省略依据或把特征说成概率。'
    if profile['level'] == 'detailed':
        return ('深入指推导充分，不是栏目和字数多：覆盖实际子问题、解释必要术语，保留成立所需的条件与中间步骤。'
                '机制题写具体运算和因果；计算题写公式代入；代码题定位本轮代码并解释输入输出；排错题给可执行核对办法。只采用本题需要的内容。'
                '问“保证、一定、至少、只有”等命题时，写清量词和假设；已有反例就据反例否定保证，不补未经证明的概率判断。'
                '教学推导、官方实现与经验结论各自标明依据，不能将简化式冒充含归一化或激活的完整算子。缺少依据的部分明确说尚不能判断。')
    return '先回答核心问题，再解释必要原因、术语和适用条件；避免只有定义或罗列名词，也不加入无关的长篇示例。'
