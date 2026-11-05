"""请求身份与数据库归属边界。无HTTP上下文仅用于启动、离线维护和测试。"""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from fastapi import HTTPException
from sqlalchemy import event,or_,select,inspect
from sqlalchemy.orm import Session,with_loader_criteria


@dataclass(frozen=True)
class Actor:
    id: str | None
    role: str = 'user'
    maintainer: bool = False


current_actor = ContextVar('atlas_current_actor', default=None)


@contextmanager
def acting_as(actor):
    token = current_actor.set(actor)
    try:
        yield
    finally:
        current_actor.reset(token)


def can_read(row):
    actor = current_actor.get()
    return actor is None or bool(row.is_public) or bool(actor.id and row.owner_id == actor.id)


def check_session(session_id):
    actor = current_actor.get()
    if actor is None or not session_id:
        return
    from backend.infra.database import engine,ChatSession
    table = ChatSession.__table__
    with engine.connect() as connection:
        owner = connection.execute(select(table.c.owner_id).where(table.c.id == session_id)).first()
    if owner is not None and owner.owner_id != actor.id:
        raise HTTPException(404,'会话不存在')


def public_payload(value):
    actor = current_actor.get()
    if actor is None or actor.maintainer:
        return value
    if isinstance(value,dict):
        private = {'generation_diagnostics','diagnostics','raw_result','raw_content','partial_json',
                   'rejected_answer','rejected_partial_answer','tool_errors','agent_error','owner_id','owner_user_id'}
        return {key:public_payload(item) for key,item in value.items() if key not in private}
    if isinstance(value,list):
        return [public_payload(item) for item in value]
    return value


def is_seed_public(row):
    table = row.__tablename__
    if table not in {'knowledge_models','knowledge_sections','knowledge_sources','knowledge_relations'}:
        return False
    payload = row.payload or {}
    return not (payload.get('document_id') or payload.get('origin') == 'user' or payload.get('kind') in {'文档','document'}
                or str(row.id).startswith(('doc:', 'D'))
                or any(str(item.get('section_id','')).startswith('doc:') for item in payload.get('evidence',[])))


def install_ownership_guards(models):
    @event.listens_for(Session,'do_orm_execute')
    def limit_reads(state):
        actor = current_actor.get()
        if actor is None:
            return
        if state.is_select:
            identity = actor.id
            for model in models:
                if identity:
                    rule = with_loader_criteria(model,lambda cls: or_(cls.is_public.is_(True),cls.owner_id == identity),include_aliases=True)
                else:
                    rule = with_loader_criteria(model,lambda cls: cls.is_public.is_(True),include_aliases=True)
                state.statement = state.statement.options(rule)
        elif (state.is_update or state.is_delete) and state.bind_mapper and state.bind_mapper.class_ in models:
            model = state.bind_mapper.class_
            allowed = model.owner_id == actor.id if actor.id else model.owner_id == '__no_user__'
            if actor.role == 'admin':
                allowed = or_(allowed,model.is_public.is_(True))
            state.statement = state.statement.where(allowed)

    @event.listens_for(Session,'before_flush')
    def limit_writes(db,flush_context,instances):
        actor = current_actor.get()
        for row in set(db.new) | set(db.dirty) | set(db.deleted):
            if type(row) not in models:
                continue
            if actor is None:
                if row in db.new or is_seed_public(row):
                    row.is_public = is_seed_public(row)
                continue
            if not actor.id:
                raise HTTPException(401,'请先登录')
            table = row.__table__
            primary = list(table.primary_key.columns)
            identity = [getattr(row,column.name) for column in primary]
            stored = None
            if all(value is not None for value in identity):
                statement = select(table.c.owner_id,table.c.is_public)
                for column,value in zip(primary,identity):
                    statement = statement.where(column == value)
                stored = db.connection().execute(statement).first()
            if stored is not None:
                if stored.owner_id != actor.id and not (stored.is_public and actor.role == 'admin'):
                    raise HTTPException(404,'记录不存在或不可修改')
                row.owner_id,row.is_public = stored.owner_id,stored.is_public
            else:
                row.owner_id,row.is_public = actor.id,False
