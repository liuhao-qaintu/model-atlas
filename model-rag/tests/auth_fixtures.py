"""原有单用户业务测试的登录夹具；真实多用户边界另由test_auth_access覆盖。

仅允许隔离SQLite；测试资料显式归属固定测试账号，使用真实JWT认证。
业务测试保留可替换的单用户检索实例，避免旧模型响应替身失效。
"""
from datetime import datetime,timedelta,timezone
import hashlib
from unittest.mock import patch
import bcrypt
import jwt
from fastapi.testclient import TestClient as BaseClient
from sqlalchemy import inspect,update
from sqlalchemy.orm import Session
from backend.infra import auth,config,database

TEST_ID='f'*32
TEST_SECRET='TEST_ONLY_business_fixture_key_not_for_live_services'
TEST_PASSWORD_HASH=bcrypt.hashpw(b'TEST_ONLY_fixture_password',bcrypt.gensalt()).decode()


class TestClient(BaseClient):
    def __init__(self,*args,**kwargs):
        if database.engine.dialect.name!='sqlite':
            raise RuntimeError('业务回归只允许显式隔离SQLite，拒绝真实MySQL')
        self._fixture_patches=[]
        super().__init__(*args,**kwargs)

    def prepare_identity(self):
        if not self._fixture_patches:
            for target,name,value in [(auth,'engine',database.engine),(config,'AUTH_SECRET_KEY',TEST_SECRET),(config,'MAINTAINER_USER_IDS',[TEST_ID])]:
                option=patch.object(target,name,value);option.start();self._fixture_patches.append(option)
        database.Account.__table__.create(database.engine,checkfirst=True)
        tables=set(inspect(database.engine).get_table_names())
        with Session(database.engine) as db:
            account=db.get(database.Account,TEST_ID)
            if not account:
                account=database.Account(id=TEST_ID,username='fixture_owner',display_name='业务测试账号',role='admin',active=True,password_hash=TEST_PASSWORD_HASH)
                db.add(account);db.flush()
            for model in database.OWNED_MODELS:
                if model.__tablename__ in tables:
                    db.execute(update(model).where(model.owner_id.is_(None),model.is_public.is_(False)).values(owner_id=TEST_ID))
            db.commit()
            token=jwt.encode({'sub':TEST_ID,'ver':auth.password_version(account.password_hash),'csrf':hashlib.sha256(b'test-fixture').hexdigest(),
                              'exp':datetime.now(timezone.utc)+timedelta(hours=1)},TEST_SECRET,algorithm='HS256')
        self.headers['Authorization']='Bearer '+token

    def request(self,*args,**kwargs):
        self.prepare_identity()
        self.prepare_retriever()
        return super().request(*args,**kwargs)

    def prepare_retriever(self):
        from backend import main
        from backend.infra.secure_retriever import RetrieverRegistry
        if isinstance(main.retriever,RetrieverRegistry):
            main.retriever=main.retriever.current()

    def __enter__(self):
        super().__enter__()
        self.prepare_identity()
        self.prepare_retriever()
        return self

    def close(self):
        try:
            super().close()
        finally:
            while self._fixture_patches:self._fixture_patches.pop().stop()

    def __exit__(self,*args):
        try:
            return super().__exit__(*args)
        finally:
            while self._fixture_patches:self._fixture_patches.pop().stop()
