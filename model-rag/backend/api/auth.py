from fastapi import APIRouter,Request,Response,HTTPException
from pydantic import BaseModel,ConfigDict,Field,SecretStr,field_validator
from sqlalchemy.orm import Session
from backend.infra import auth,config
from backend.infra.access import current_actor
from backend.infra.database import Account,engine

router = APIRouter(prefix='/api/auth',tags=['账号'])


class LoginInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    username:str = Field(min_length=3,max_length=40,pattern=r'^[A-Za-z0-9_.-]+$')
    password:SecretStr

    @field_validator('username',mode='before')
    @classmethod
    def normalize_name(cls,value):
        return value.strip().lower() if isinstance(value,str) else value


class RegisterInput(LoginInput):
    display_name:str = Field(default='',max_length=80)

    @field_validator('password')
    @classmethod
    def validate_password(cls,value):
        password = value.get_secret_value()
        if len(password) < 8 or len(password.encode('utf-8')) > 72:
            raise ValueError('密码至少8个字符，且不超过72个UTF-8字节')
        return value


def set_login_cookies(response,token,csrf):
    options = {'secure':config.AUTH_COOKIE_SECURE,'samesite':'lax','path':'/','max_age':config.AUTH_EXPIRE_HOURS*3600}
    response.set_cookie(auth.COOKIE,token,httponly=True,**options)
    response.set_cookie(auth.CSRF_COOKIE,csrf,httponly=False,**options)
    response.headers['Cache-Control'] = 'no-store'


@router.post('/register')
def register(value:RegisterInput,response:Response):
    response.headers['Cache-Control'] = 'no-store'
    return auth.register_account(value.username,value.password.get_secret_value(),value.display_name)


@router.post('/login')
def login(value:LoginInput,request:Request,response:Response):
    account,token,csrf = auth.login_account(value.username,value.password.get_secret_value(),request.client.host if request.client else 'unknown')
    set_login_cookies(response,token,csrf)
    return account


@router.get('/me')
def me(response:Response):
    actor = current_actor.get()
    with Session(engine) as db:
        account = db.get(Account,actor.id)
        if not account or not account.active:
            raise HTTPException(401,'登录已失效，请重新登录')
        response.headers['Cache-Control'] = 'no-store'
        return auth.account_payload(account)


@router.post('/logout')
def logout(response:Response):
    response.delete_cookie(auth.COOKIE,path='/')
    response.delete_cookie(auth.CSRF_COOKIE,path='/')
    response.headers['Cache-Control'] = 'no-store'
    return {'message':'已退出登录'}


class PasswordInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    old_password:SecretStr
    new_password:SecretStr


@router.post('/password')
def password(value:PasswordInput,response:Response):
    try:
        auth.change_password(current_actor.get(),value.old_password.get_secret_value(),value.new_password.get_secret_value())
    except ValueError as error:
        raise HTTPException(422,str(error))
    return logout(response)
