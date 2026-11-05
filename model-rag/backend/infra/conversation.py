"""会话事实从已提交的用户消息读取，不送入知识库或让模型猜测。"""
import re


def is_conversation_query(question):
    text = re.sub(r"\s+", "", question)
    return bool(re.search(
        r"(?:我|我们).*(?:问过|问了|问你的|提过|提了).*(?:什么|哪些|原话)|"
        r"(?:上(?:上)?(?:一个|个|一条|条|次)|刚才|之前|前面|第[一二三四五六七八九十\d]+(?:个|条)).*(?:问题|提问).*(?:是什么|是啥|内容|原话)|"
        r"(?:回顾|回看|列出|重复|复述|总结).*(?:我|我们|本次|这次|当前|刚才|之前|前面).*(?:问题|提问)|"
        r"(?:本次|这次|当前).*(?:对话|会话).*(?:问了什么|问过什么|哪些问题)", text))


def conversation_reply(question, questions):
    if not is_conversation_query(question):
        return None
    if not questions:
        return "当前是一个新会话，还没有上一条已完成的提问。示例问题只有在你点击发送后才会计入会话。"
    text = re.sub(r"\s+", "", question)
    if re.search(r"哪些|问过什么|问了什么|列出|回顾|回看|总结|全部|所有", text) and not re.search(r"上(?:上)?(?:一个|个|一条|条|次)|第.+(?:个|条)", text):
        start = max(0, len(questions) - 10)
        return f"本会话已完成 {len(questions)} 次提问，以下是最近 {len(questions) - start} 条用户原话：\n\n" + "\n\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions[start:], start))
    ordinal = re.search(r"第([一二三四五六七八九十\d]+)(?:个|条|次)", text)
    if ordinal:
        value = ordinal[1]
        number = int(value) if value.isdigit() else {c: i for i, c in enumerate("一二三四五六七八九十", 1)}.get(value)
        if number is None or number < 1 or number > len(questions):
            return f"本会话目前已完成 {len(questions)} 次提问，没有可定位的第 {value} 条。"
        label, answer = f"第 {number} 条提问", questions[number - 1]
    else:
        offset = 2 if re.search(r"上上|前两个|前两条", text) else 1
        if len(questions) < offset:
            return f"本会话目前只有 {len(questions)} 条已完成的提问，没有上上条。"
        label, answer = ("上上条提问" if offset == 2 else "上一条提问"), questions[-offset]
    return f"你在本会话中的{label}是：\n\n{answer}"
