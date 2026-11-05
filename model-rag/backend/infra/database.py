"""MySQL 保存资料、章节、关系与会话；向量缓存单独保存。"""
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Integer, Boolean, DateTime, ForeignKey, String, Text, create_engine, inspect, select, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from backend.infra.config import DATABASE_URL

engine_options = {'pool_pre_ping': True, 'pool_recycle': 1800}
if DATABASE_URL.get_backend_name() == 'sqlite':
    engine_options['connect_args'] = {'check_same_thread': False}
    if not DATABASE_URL.database or DATABASE_URL.database == ':memory:':
        engine_options['poolclass'] = StaticPool
engine = create_engine(DATABASE_URL, **engine_options)
TIMESTAMP = DateTime().with_variant(DATETIME(fsp=6), 'mysql')


class Base(DeclarativeBase):
    pass


class Ownership:
    owner_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text('0'))


class Account(Base):
    __tablename__ = 'atlas_accounts'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    username: Mapped[str] = mapped_column(String(40), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(20), default='user')
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class KnowledgeModel(Ownership, Base):
    __tablename__ = "knowledge_models"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class KnowledgeSection(Ownership, Base):
    __tablename__ = "knowledge_sections"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("knowledge_models.id"), index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class KnowledgeSource(Ownership, Base):
    __tablename__ = "knowledge_sources"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class KnowledgeRelation(Ownership, Base):
    __tablename__ = "knowledge_relations"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class KnowledgeDocument(Ownership, Base):
    __tablename__ = "knowledge_documents"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class ChatSession(Ownership, Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class KnowledgeAnalysis(Ownership, Base):
    __tablename__ = 'knowledge_analyses'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class ChatMessage(Ownership, Base):
    __tablename__ = "chat_messages"
    sequence: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(40), unique=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


def initialize_tables():
    Base.metadata.create_all(engine)
    for model in OWNED_MODELS:
        columns = {column['name'] for column in inspect(engine).get_columns(model.__tablename__)}
        with engine.begin() as connection:
            if 'owner_id' not in columns:
                connection.execute(text(f'ALTER TABLE {model.__tablename__} ADD COLUMN owner_id VARCHAR(40) NULL'))
            if 'is_public' not in columns:
                connection.execute(text(f'ALTER TABLE {model.__tablename__} ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0'))
    # 早期本地表没有顺序列；同一批写入可能获得相同时间戳。
    if "sequence" not in {c["name"] for c in inspect(engine).get_columns("chat_messages")}:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE chat_messages DROP PRIMARY KEY, ADD COLUMN sequence BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, ADD UNIQUE KEY uq_chat_message_id (id)"))


def upsert_records(table, records):
    with Session(engine) as db:
        for record in records:
            values = {"id": record["id"], "payload": record}
            if table is KnowledgeSection:
                values["model_id"] = record["model_id"]
            db.merge(table(**values))
        db.commit()


def records(table):
    with Session(engine) as db:
        return [row.payload for row in db.scalars(select(table)).all()]


def save_turn(session_id, question, answer, metadata):
    from backend.infra.access import check_session
    check_session(session_id)
    from uuid import uuid4
    with Session(engine) as db:
        if not db.get(ChatSession, session_id):
            db.add(ChatSession(id=session_id, title=question[:100]))
            db.flush()
        db.add(ChatMessage(id=str(uuid4()), session_id=session_id, role="user", content=question, payload={"model_id": metadata.get("model_id"), "section_id": metadata.get("section_id")}))
        db.add(ChatMessage(id=str(uuid4()), session_id=session_id, role="assistant", content=answer, payload=metadata))
        db.commit()


def history(session_id, limit=8):
    from backend.infra.access import check_session
    check_session(session_id)
    with Session(engine) as db:
        stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.sequence.desc()).limit(limit)
        rows = list(reversed(db.scalars(stmt).all()))
        return [{"id": r.id, "role": r.role, "content": r.content, "metadata": r.payload} for r in rows]


def user_questions(session_id):
    """仅读取用户原话；完整回忆不受生成模型的历史窗口限制。"""
    from backend.infra.access import check_session
    check_session(session_id)
    with Session(engine) as db:
        stmt = select(ChatMessage.content).where(ChatMessage.session_id == session_id, ChatMessage.role == "user").order_by(ChatMessage.sequence)
        return list(db.scalars(stmt))


def session_is_cloud_safe(session_id):
    with Session(engine) as db:
        payloads = db.scalars(select(ChatMessage.payload).where(ChatMessage.session_id == session_id, ChatMessage.role == "assistant"))
        return all(payload.get("cloud_allowed", True) and all(e.get("cloud_allowed", True) for e in payload.get("evidence", [])) for payload in payloads)


OWNED_MODELS = (KnowledgeModel, KnowledgeSection, KnowledgeSource, KnowledgeRelation,
                KnowledgeDocument, KnowledgeAnalysis, ChatSession, ChatMessage)
from backend.infra.access import install_ownership_guards
install_ownership_guards(OWNED_MODELS)
