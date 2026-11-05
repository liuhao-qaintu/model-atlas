"""按已保存消息顺序浏览会话；标题编辑不改变消息或模型上下文。"""
from datetime import timezone

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session, aliased

from backend.infra import database
from backend.infra.database import ChatMessage, ChatSession


def utc_text(value):
    return value.replace(tzinfo=timezone.utc).isoformat() if value else None


def list_sessions(query='', before=None, limit=20):
    activity = select(
        ChatMessage.session_id.label('session_id'),
        func.max(ChatMessage.sequence).label('last_sequence'),
        func.sum(case((ChatMessage.role == 'user', 1), else_=0)).label('question_count'),
    ).group_by(ChatMessage.session_id).subquery()
    latest_question = select(
        ChatMessage.session_id.label('session_id'),
        func.max(ChatMessage.sequence).label('sequence'),
    ).where(ChatMessage.role == 'user').group_by(ChatMessage.session_id).subquery()
    last, question = aliased(ChatMessage), aliased(ChatMessage)
    statement = select(ChatSession, last, question.content, activity.c.question_count).join(
        activity, activity.c.session_id == ChatSession.id,
    ).join(last, last.sequence == activity.c.last_sequence).join(
        latest_question, latest_question.c.session_id == ChatSession.id,
    ).join(question, question.sequence == latest_question.c.sequence)
    if query.strip():
        statement = statement.where(or_(ChatSession.title.contains(query.strip(), autoescape=True),
                                       question.content.contains(query.strip(), autoescape=True)))
    if before is not None:
        statement = statement.where(activity.c.last_sequence < before)
    with Session(database.engine) as db:
        rows = db.execute(statement.order_by(activity.c.last_sequence.desc()).limit(limit + 1)).all()
        items = [{
            'id': session.id, 'title': session.title, 'question_count': int(count),
            'last_question': preview[:180], 'updated_at': utc_text(last_message.created_at),
            'model_id': last_message.payload.get('model_id'),
            'mode': last_message.payload.get('mode', 'learning'), 'last_sequence': last_message.sequence,
        } for session, last_message, preview, count in rows[:limit]]
    return {'items': items, 'has_more': len(rows) > limit,
            'next_cursor': items[-1]['last_sequence'] if len(rows) > limit else None}


def session_page(session_id, before=None, limit=100):
    with Session(database.engine) as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise LookupError('会话不存在')
        statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
        if before is not None:
            statement = statement.where(ChatMessage.sequence < before)
        rows = db.scalars(statement.order_by(ChatMessage.sequence.desc()).limit(limit + 1)).all()
        selected = list(reversed(rows[:limit]))
        count = db.scalar(select(func.count()).select_from(ChatMessage).where(
            ChatMessage.session_id == session_id, ChatMessage.role == 'user'))
        return {'session': {'id': session.id, 'title': session.title, 'question_count': count},
                'messages': [{'id': row.id, 'sequence': row.sequence, 'role': row.role,
                              'content': row.content, 'metadata': row.payload} for row in selected],
                'has_more': len(rows) > limit,
                'next_cursor': selected[0].sequence if len(rows) > limit else None}


def rename_session(session_id, title):
    title = title.strip()
    if not title or len(title) > 120:
        raise ValueError('会话标题须为1–120个字符')
    with Session(database.engine) as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise LookupError('会话不存在')
        session.title = title
        db.commit()
    return {'id': session_id, 'title': title}
