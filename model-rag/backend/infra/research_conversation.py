"""把已完成的研究报告原样接入聊天，不重新生成答案或杜撰用户历史。"""
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.infra.database import ChatMessage, ChatSession, KnowledgeAnalysis
from backend.rag.research import allowed_scope


def conversation_metadata(report, retriever, session_id):
    if report.get('type') != 'agentic_research_run' or report.get('generated') is not True or not report.get('answer'):
        raise ValueError('只能继续追问已完整完成的研究报告')
    permitted = allowed_scope(retriever)
    evidence = []
    for index, item in enumerate(report.get('evidence', []), 1):
        section = permitted.get(item['section_id'])
        if not section:
            raise ValueError('报告引用已失效或不再允许云端处理，请重新核对资料后研究')
        evidence.append({**section, **item, 'id': item['section_id'], 'reference': index,
                         'section_slug': section.get('slug'), 'generation_excerpt': item['excerpt'],
                         'cloud_allowed': True, 'sources': []})
    if not evidence:
        raise ValueError('研究报告没有可继续核对的引用')
    model_ids = [mid for mid in report.get('outline', {}).get('model_ids', []) if mid in retriever.models]
    selected = list(dict.fromkeys(report.get('document_ids', [])))
    # 已保存报告来自不同文献时沿用原研究范围，不把未选文献悄悄改成强制筛选。
    if selected:
        allowed_scope(retriever, selected)
    focus = model_ids[0] if len(model_ids) == 1 else None
    return {'session_id': session_id, 'turn_id': report['id'], 'research_run_id': report['id'],
            'generated': True, 'cloud_allowed': True, 'response_kind': 'research', 'mode': 'research',
            'model_id': focus, 'model_ids': model_ids, 'section_id': None,
            'document_ids': selected, 'document_policy': 'auto', 'evidence': evidence,
            'plan': {'model_ids': model_ids, 'model_id': focus, 'section_id': None,
                     'original_query': report['question'], 'normalized_query': report['question'],
                     'mode': 'comparison' if len(model_ids) > 1 else 'single',
                     'source_scope': {'kind': 'general', 'origin': 'research_report'}},
            'links': [{'label': '查看原比较结果', 'url': f"/research?run={report['id']}"}],
            'warnings': [], 'paths': []}


def open_research_conversation(identity, retriever, engine):
    with Session(engine) as db:
        row = db.scalar(select(KnowledgeAnalysis).where(KnowledgeAnalysis.id == identity).with_for_update())
        if row is None:
            raise LookupError('研究记录不存在')
        report = row.payload
        session_id = report.get('conversation_id') or str(uuid4())
        metadata = conversation_metadata(report, retriever, session_id)
        if report.get('conversation_id') and db.get(ChatSession, session_id):
            return {'session_id': session_id, 'research_run_id': identity, 'reused': True}
        db.add(ChatSession(id=session_id, title=report['question'][:100]))
        db.flush()
        db.add_all([
            ChatMessage(id=str(uuid4()), session_id=session_id, role='user', content=report['question'],
                        payload={'model_id': metadata['model_id'], 'section_id': None, 'research_run_id': identity}),
            ChatMessage(id=str(uuid4()), session_id=session_id, role='assistant', content=report['answer'], payload=metadata),
        ])
        row.payload = {**report, 'conversation_id': session_id}
        db.commit()
        return {'session_id': session_id, 'research_run_id': identity, 'reused': False}
