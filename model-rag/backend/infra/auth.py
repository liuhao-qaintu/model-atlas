"""账户认证、请求范围及Cookie防跨站校验。"""
from collections import OrderedDict
from datetime import datetime,timedelta,timezone
import hashlib
import hmac
import secrets
import threading
import time
from uuid import uuid4
import bcrypt
import jwt
from fastapi import HTTPException,Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from backend.infra import config
from backend.infra.database import Account,engine
from backend.infra.access import Actor,current_actor,public_payload

COOKIE = 'atlas_session'
CSRF_COOKIE = 'atlas_csrf'
LOGIN_ATTEMPTS = OrderedDict()
LOGIN_LOCK = threading.Lock()


def require_secret():
    if len(config.AUTH_SECRET_KEY) < 32:
        raise HTTPException(503,'认证配置未完成，请由维护人员配置AUTH_SECRET_KEY')


def password_hash(value):
    if len(value) < 8 or len(value.encode('utf-8')) > 72:
        raise ValueError('密码至少8个字符，且不超过72个UTF-8字节')
    return bcrypt.hashpw(value.encode(),bcrypt.gensalt()).decode()


def password_matches(value,stored):
    try:
        return bcrypt.checkpw(value.encode(),stored.encode())
    except ValueError:
        return False


def password_version(value):
    return hmac.new(config.AUTH_SECRET_KEY.encode(),b'password-version\0'+value.encode(),hashlib.sha256).hexdigest()


def account_payload(account):
    return {'id':account.id,'username':account.username,'display_name':account.display_name,
            'role':account.role,'is_maintainer':account.role == 'admin' and account.id in config.MAINTAINER_USER_IDS}


def register_account(username,password,display_name=''):
    require_secret()
    with Session(engine) as db:
        if db.query(Account).filter(Account.username == username).first():
            raise HTTPException(409,'用户名已被使用')
        account = Account(id=uuid4().hex,username=username,password_hash=password_hash(password),display_name=display_name or username,role='user',active=True)
        db.add(account)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409,'用户名已被使用')
        return account_payload(account)


def login_account(username,password,ip):
    require_secret()
    key = hmac.new(config.AUTH_SECRET_KEY.encode(),(ip+'\0'+username).encode(),hashlib.sha256).hexdigest()
    now = time.monotonic()
    with LOGIN_LOCK:
        attempts = [at for at in LOGIN_ATTEMPTS.pop(key,[]) if now-at < 60]
        LOGIN_ATTEMPTS[key] = attempts
        if len(attempts) >= 5:
            raise HTTPException(429,'尝试过于频繁，请稍后重试')
        attempts.append(now)
        if len(LOGIN_ATTEMPTS) > 2048:
            LOGIN_ATTEMPTS.popitem(last=False)
    with Session(engine) as db:
        account = db.query(Account).filter(Account.username == username).first()
        if not account or not account.active or not password_matches(password,account.password_hash):
            raise HTTPException(401,'用户名或密码错误')
        csrf = secrets.token_urlsafe(32)
        token = jwt.encode({'sub':account.id,'ver':password_version(account.password_hash),
            'csrf':hashlib.sha256(csrf.encode()).hexdigest(),
            'exp':datetime.now(timezone.utc)+timedelta(hours=config.AUTH_EXPIRE_HOURS)},config.AUTH_SECRET_KEY,algorithm='HS256')
        with LOGIN_LOCK:
            LOGIN_ATTEMPTS.pop(key,None)
        return account_payload(account),token,csrf


def authenticate(token):
    require_secret()
    try:
        payload = jwt.decode(token,config.AUTH_SECRET_KEY,algorithms=['HS256'],options={'require':['sub','exp','ver','csrf']})
        if not isinstance(payload['sub'],str):
            raise ValueError('invalid subject')
    except (jwt.InvalidTokenError,ValueError):
        raise HTTPException(401,'登录已失效，请重新登录')
    with Session(engine) as db:
        account = db.get(Account,payload['sub'])
        if not account or not account.active or not isinstance(payload['ver'],str) or not hmac.compare_digest(payload['ver'],password_version(account.password_hash)):
            raise HTTPException(401,'登录已失效，请重新登录')
        return Actor(account.id,account.role,account.role == 'admin' and account.id in config.MAINTAINER_USER_IDS),payload


def change_password(actor,old_password,new_password):
    with Session(engine) as db:
        account = db.get(Account,actor.id)
        if not account or not account.active or not password_matches(old_password,account.password_hash):
            raise HTTPException(400,'原密码错误')
        if old_password == new_password:
            raise HTTPException(400,'新密码不能与原密码相同')
        account.password_hash = password_hash(new_password)
        db.commit()


class ScopedJSONResponse(JSONResponse):
    def render(self,content):
        return super().render(public_payload(content))


class AuthenticationMiddleware:
    def __init__(self,app):
        self.app = app

    async def __call__(self,scope,receive,send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/') or scope['method'] == 'OPTIONS':
            return await self.app(scope,receive,send)
        request = Request(scope,receive)
        actor = Actor(None,'anonymous')
        public = scope['path'] in {'/api/health','/api/auth/login','/api/auth/register'}
        try:
            if scope['method'] not in {'GET','HEAD'}:
                origin = request.headers.get('origin')
                if origin and origin not in {'http://127.0.0.1:5174','http://localhost:5174',str(request.base_url).rstrip('/')}:
                    raise HTTPException(403,'请求来源不被允许')
            if scope['path'] == '/api/health':
                header = request.headers.get('authorization','')
                token = header[7:] if header.startswith('Bearer ') else request.cookies.get(COOKIE)
                if token:
                    try:
                        actor,_ = await run_in_threadpool(authenticate,token)
                    except HTTPException:
                        pass
            if not public:
                header = request.headers.get('authorization','')
                bearer = header[7:] if header.startswith('Bearer ') else None
                token = bearer or request.cookies.get(COOKIE)
                if not token:
                    raise HTTPException(401,'请先登录')
                actor,payload = await run_in_threadpool(authenticate,token)
                if not bearer and scope['method'] not in {'GET','HEAD'}:
                    csrf = request.headers.get('x-csrf-token','')
                    if not csrf or not hmac.compare_digest(csrf,request.cookies.get(CSRF_COOKIE,'')) or not hmac.compare_digest(hashlib.sha256(csrf.encode()).hexdigest(),payload['csrf']):
                        raise HTTPException(403,'请求验证失败，请刷新页面后重试')
                maintenance = scope['path'].startswith(('/api/evaluation','/api/analysis','/api/review')) or scope['path'] in {'/api/index','/api/knowledge/refresh'}
                if maintenance and not actor.maintainer:
                    raise HTTPException(403,'此功能仅供维护人员使用')
        except HTTPException as error:
            return await JSONResponse({'detail':error.detail},status_code=error.status_code,headers={'Cache-Control':'no-store'})(scope,receive,send)
        identity = current_actor.set(actor)
        async def private_send(message):
            if message['type'] == 'http.response.start':
                headers = [(key,value) for key,value in message.get('headers',[]) if key.lower() not in {b'cache-control',b'pragma'}]
                message = {**message,'headers':headers+[(b'cache-control',b'no-store'),(b'pragma',b'no-cache')]}
            await send(message)
        try:
            await self.app(scope,receive,private_send)
        finally:
            current_actor.reset(identity)
