"""本地维护账号；密码交互输入，不提供公开的管理员创建接口。"""
import argparse
from pathlib import Path
import re
import sys
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select,update
from sqlalchemy.orm import Session
from backend.infra.database import Account,engine,initialize_tables,OWNED_MODELS
from backend.infra.auth import password_hash,require_secret


def read_password():
    value = input('设置密码：')
    if value != input('再次输入密码：'):
        raise ValueError('两次输入的密码不一致')
    return password_hash(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['create-admin','reset-password','list'])
    parser.add_argument('--username')
    parser.add_argument('--claim-legacy',action='store_true',help='显式将尚无归属的非公共旧数据分配给此管理员')
    args = parser.parse_args()
    if args.action != 'list' and (not args.username or not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}',args.username)):
        parser.error('请提供3至40位英文、数字、点、下划线或连字符的用户名')
    if args.claim_legacy and args.action != 'create-admin':
        parser.error('--claim-legacy仅用于显式初始化管理员')
    require_secret()
    initialize_tables()
    with Session(engine) as db:
        if args.action == 'list':
            for account in db.scalars(select(Account)).all():
                print(account.id,account.username,account.role,'正常' if account.active else '停用')
            return
        username = args.username.lower()
        account = db.scalar(select(Account).where(Account.username == username))
        if args.action == 'create-admin':
            if account is not None:
                raise ValueError('账号已存在，不会更改其密码或角色')
            account = Account(id=uuid4().hex,username=username,display_name=username,
                              password_hash=read_password(),role='admin',active=True)
            db.add(account)
            db.flush()
            if args.claim_legacy:
                for model in OWNED_MODELS:
                    db.execute(update(model).where(model.owner_id.is_(None),model.is_public.is_(False)).values(owner_id=account.id))
        else:
            if account is None or not account.active:
                raise ValueError('账号不存在或已停用')
            account.password_hash = read_password()
        db.commit()
        print('操作完成。账号ID：'+account.id)
        print('维护接口还需在本地配置MAINTAINER_USER_IDS，业务管理员不会自动获得维护权限。')
        print('如接管了旧资料或修改了维护配置，请重启后端使缓存与配置同步。')


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error))
