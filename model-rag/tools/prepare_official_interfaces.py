"""下载固定官方源码并静态整理接口；不导入或执行所下载代码。"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import textwrap
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.api_client import app_client
ARCHIVE = ROOT.parent / '模型知识库资料/公开原始资料/接口与实现'
MANIFEST = ROOT / 'data/interface-source-manifest.json'
SPECS = [
    {'key': 'conv1d-interface', 'model': 'cnn-1d', 'title': 'PyTorch Conv1d 官方接口与实现 · v2.8.0', 'version': 'v2.8.0',
     'filename': 'conv_v2.8.0.py', 'path': 'torch/nn/modules/conv.py', 'classes': ['Conv1d'],
     'symbols': ['_ConvNd.__init__', 'Conv1d.__init__', 'Conv1d._conv_forward', 'Conv1d.forward']},
    {'key': 'transformer-interface', 'model': 'transformer', 'title': 'PyTorch Transformer 官方接口与掩码 · v2.6.0', 'version': 'v2.6.0',
     'filename': 'transformer_v2.6.0.py', 'path': 'torch/nn/modules/transformer.py', 'classes': ['Transformer'],
     'symbols': ['Transformer.__init__', 'Transformer.forward', 'Transformer.generate_square_subsequent_mask',
                 'TransformerEncoderLayer.forward', 'TransformerEncoderLayer._sa_block', 'TransformerDecoderLayer.forward']},
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def literal_text(expr, notes=None):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return literal_text(expr.left, notes) + literal_text(expr.right, notes)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == 'format':
        return literal_text(expr.func.value, notes).format_map(KeepUnknown(notes or {}))
    raise ValueError('仅接受固定源码中的静态字符串，不能执行表达式')


class KeepUnknown(dict):
    def __missing__(self, name):
        return '{' + name + '}'


def prepare(spec, raw):
    text = raw.decode('utf-8'); tree = ast.parse(text); lines = text.splitlines()
    owners = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    notes = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'convolution_notes' for t in node.targets):
            notes = ast.literal_eval(node.value)
    source = f"https://github.com/pytorch/pytorch/blob/{spec['version']}/{spec['path']}"
    output = [f"# {spec['title']}", '', f"版本：{spec['version']}。官方文档字符串分段排版，代码仅静态摘录，未执行。", '',
              f"原始文件：{source}。版权与许可沿用上游 LICENSE。未展开的上游硬件备注占位符不作为接口结论。", '']
    for name in spec['classes']:
        owner = owners[name]
        doc = ast.get_docstring(owner, clean=False)
        if doc is None:
            assigned = next(n for n in owner.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__doc__' for t in n.targets))
            doc = literal_text(assigned.value, notes)
        output += [f'## {name} · 官方接口文档', '', f'源码定位：{source}#L{owner.lineno}-L{owner.end_lineno}', '', textwrap.dedent(doc).strip(), '']
    for symbol in spec['symbols']:
        owner, name = symbol.split('.')
        method = next(n for n in owners[owner].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        doc = ast.get_docstring(method, clean=False)
        if doc:
            output += [f'## {symbol} · 参数、形状与返回值', '', f'源码定位：{source}#L{method.lineno}-L{method.end_lineno}', '', textwrap.dedent(doc).strip(), '']
            docnode = method.body[0]
            ranges = [(method.lineno, docnode.lineno - 1), (docnode.end_lineno + 1, method.end_lineno)]
        else:
            ranges = [(method.lineno, method.end_lineno)]
        ranges = [(a, b) for a, b in ranges if a <= b]
        code = '\n'.join('\n'.join(lines[a - 1:b]) for a, b in ranges)
        output += [f'## {symbol} · 执行代码', '', '源码定位：' + '；'.join(f'{source}#L{a}-L{b}' for a, b in ranges), '',
                   '以下签名和执行代码来自所列原文范围；文档字符串已在上节单列。不是独立可运行模块，未改变执行语句。', '', '```python', code, '```', '']
    return '\n'.join(output).encode('utf-8')


def download():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    prior = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'entries': []}
    saved = {e['key']: e for e in prior['entries']}
    entries = []
    with httpx.Client(timeout=35, follow_redirects=True, trust_env=False) as client:
        for spec in SPECS:
            url = f"https://raw.githubusercontent.com/pytorch/pytorch/{spec['version']}/{spec['path']}"
            path = ARCHIVE / spec['filename']; old = saved.get(spec['key'])
            if old and path.exists() and digest(path.read_bytes()) == old['source_sha256']:
                raw = path.read_bytes()
            else:
                try:
                    response = client.get(url); response.raise_for_status(); raw = response.content
                except httpx.HTTPError:
                    api = f"https://api.github.com/repos/pytorch/pytorch/contents/{spec['path']}"
                    response = client.get(api, params={'ref': spec['version']}); response.raise_for_status()
                    item = response.json()
                    if item.get('path') != spec['path'] or item.get('type') != 'file' or item.get('encoding') != 'base64':
                        raise ValueError('官方GitHub接口未返回指定文件')
                    raw = base64.b64decode(item['content'])
                    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                    if blob != item['sha']:
                        raise ValueError('GitHub文件对象摘要不匹配')
                ast.parse(raw.decode('utf-8'))
                if path.exists() and path.read_bytes() != raw:
                    raise ValueError('已有源码与官方响应不一致，拒绝覆盖')
                path.write_bytes(raw)
            prepared = prepare(spec, raw); imported = path.with_suffix('.md')
            if imported.exists() and imported.read_bytes() != prepared:
                raise ValueError('已有整理稿不同，须另建版本，拒绝覆盖')
            imported.write_bytes(prepared)
            entry = {**spec, 'url': url, 'source_url': url.replace('raw.githubusercontent.com/pytorch/pytorch/', 'github.com/pytorch/pytorch/blob/'),
                     'type': 'official_code', 'source_path': str(path), 'import_path': str(imported),
                     'source_sha256': digest(raw), 'import_sha256': digest(prepared), 'bytes': len(raw)}
            if old and old.get('import_sha256') == entry['import_sha256']:
                entry.update({k: old[k] for k in ('document_id', 'part_count') if k in old})
            entries.append(entry)
            print(json.dumps({'prepared': spec['key'], 'raw_bytes': len(raw), 'import_bytes': len(prepared)}), flush=True)
    MANIFEST.write_text(json.dumps({'schema': 'official-interfaces-v1', 'entries': entries}, ensure_ascii=False, indent=2), encoding='utf-8')


def import_documents():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with app_client(timeout=120) as client:
        for entry in manifest['entries']:
            path = Path(entry['import_path'])
            if not path.is_absolute(): path = ROOT / path
            raw = path.read_bytes()
            if digest(raw) != entry['import_sha256']:
                raise ValueError('整理稿摘要不匹配')
            response = client.post('/documents/preview', files={'file': (path.name, raw)}, data={'model_id': entry['model']}); response.raise_for_status()
            preview = response.json()
            response = client.post(f"/documents/{preview['id']}/import", json={'title': entry['title'], 'source_url': entry['source_url'], 'cloud_allowed': True}); response.raise_for_status()
            entry.update(document_id=preview['id'], part_count=response.json()['part_count'])
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'imported': entry['key'], 'id': entry['document_id'], 'parts': entry['part_count']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--import-only', action='store_true'); parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if not args.import_only:
        download()
    if not args.download_only:
        import_documents()
