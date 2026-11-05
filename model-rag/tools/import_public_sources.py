"""下载固定版本公开原文；按核对的页码和函数通过预览接口入库。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import httpx
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.api_client import app_client
from backend.infra.recent_knowledge import PAPER_RECIPES
ARCHIVE = ROOT.parent / '模型知识库资料' / '公开原始资料'
MANIFEST = ROOT / 'data/public-source-manifest.json'
CLIP_COMMIT = 'a1d071733d7111c9c014f024669f959182114e33'
ENTRIES = [
    {'key': 'transformer-paper', 'model': 'transformer', 'title': 'Transformer 原始论文 · v7 · 方法与位置编码', 'version': '1706.03762v7',
     'url': 'https://arxiv.org/pdf/1706.03762v7', 'filename': 'Transformer_1706.03762v7.pdf', 'pages': '3-6', 'type': 'paper'},
    {'key': 'vit-paper', 'model': 'vit', 'title': 'ViT 原始论文 · v2 · 结构与迁移接口', 'version': '2010.11929v2',
     'url': 'https://arxiv.org/pdf/2010.11929v2', 'filename': 'ViT_2010.11929v2.pdf', 'pages': '3-4', 'type': 'paper'},
    {'key': 'clip-paper', 'model': 'clip', 'title': 'CLIP 原始论文 · v1 · 图文学习与零样本接口', 'version': '2103.00020v1',
     'url': 'https://arxiv.org/pdf/2103.00020v1', 'filename': 'CLIP_2103.00020v1.pdf', 'pages': '3-6', 'type': 'paper'},
    {'key': 'vit-code', 'model': 'vit', 'title': 'Torchvision ViT 官方源码 · v0.21.0 · 关键函数', 'version': 'torchvision v0.21.0',
     'url': 'https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/models/vision_transformer.py',
     'source_url': 'https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py',
     'filename': 'vision_transformer_v0.21.0.py', 'type': 'official_code', 'git_blob_sha': 'f2983ef9db0a3adf232dd6fe1b90ce9417ce0853',
     'symbols': ['EncoderBlock.__init__', 'EncoderBlock.forward', 'VisionTransformer._process_input', 'VisionTransformer.forward']},
    {'key': 'clip-code', 'model': 'clip', 'title': 'CLIP 官方源码 · a1d0717 · 双编码器与匹配', 'version': CLIP_COMMIT,
     'url': f'https://raw.githubusercontent.com/openai/CLIP/{CLIP_COMMIT}/clip/model.py',
     'source_url': f'https://github.com/openai/CLIP/blob/{CLIP_COMMIT}/clip/model.py',
     'filename': 'clip_model_a1d0717.py', 'type': 'official_code', 'git_blob_sha': '232b7792eb97440642547bd462cf128df9243933',
     'symbols': ['VisionTransformer.__init__', 'VisionTransformer.forward', 'CLIP.__init__', 'CLIP.build_attention_mask', 'CLIP.encode_image', 'CLIP.encode_text', 'CLIP.forward']},
    {'key': 'vit-heads', 'model': 'vit', 'title': 'Torchvision ViT 官方定义 · v0.21.0 · 分类头与构造参数', 'version': 'torchvision v0.21.0',
     'url': 'https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/models/vision_transformer.py',
     'source_url': 'https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py',
     'filename': 'vision_transformer_v0.21.0.py', 'import_filename': 'vision_transformer_heads_v0.21.0.md',
     'type': 'official_code', 'git_blob_sha': 'f2983ef9db0a3adf232dd6fe1b90ce9417ce0853',
     'symbols': ['VisionTransformer.__init__'], 'blocks': ['signature', 'heads', 'head_initialization'],
     'definition_dependency': {'source_key': 'vit-code', 'call_quote': 'self.heads(x)', 'definition_quote': 'heads_layers:'}},
    {'key': 'detr-paper', 'model': 'detr', 'title': 'DETR 原始论文 · v3 · 检测结构与集合预测', 'version': '2005.12872v3',
     'url': 'https://arxiv.org/pdf/2005.12872v3', 'filename': 'DETR_2005.12872v3.pdf', 'pages': '5-8', 'type': 'paper'},
    {'key': 'llava-paper', 'model': 'llava', 'title': 'LLaVA 原始论文 · v2 · 视觉连接与两阶段微调', 'version': '2304.08485v2',
     'url': 'https://arxiv.org/pdf/2304.08485v2', 'filename': 'LLaVA_2304.08485v2.pdf', 'pages': '4-5', 'type': 'paper'},
    {'key': 'dinov2-paper', 'model': 'dinov2', 'title': 'DINOv2 原始论文 · v2 · 自监督特征与下游接口', 'version': '2304.07193v2',
     'url': 'https://arxiv.org/pdf/2304.07193v2', 'filename': 'DINOv2_2304.07193v2.pdf', 'pages': '5-7,10,13-15', 'type': 'paper'},
]


ENTRIES.extend(PAPER_RECIPES)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def constructor_ranges(node, lines, blocks):
    """在指定构造函数内摘录原样区域；定位缺失或不唯一时拒绝生成。"""
    def locate(prefix):
        found = [index + 1 for index in range(node.lineno - 1, node.end_lineno) if lines[index].strip().startswith(prefix)]
        if len(found) != 1:
            raise ValueError('源码区域定位不唯一：' + prefix)
        return found[0]
    ranges = []
    for block in blocks:
        if block == 'signature':
            ranges.append(('参数签名', node.lineno, node.body[0].lineno - 1))
        elif block == 'heads':
            ranges.append(('分类头构造', locate('heads_layers:'), locate('self.heads = nn.Sequential(')))
        elif block == 'head_initialization':
            ranges.append(('分类头权重初始化', locate('if hasattr(self.heads, "pre_logits")'), node.end_lineno))
        else:
            raise ValueError('未知源码摘录区域：' + block)
    return ranges


def code_excerpt(entry, raw):
    text = raw.decode('utf-8')
    lines = text.splitlines()
    symbols = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols[f'{node.name}.{method.name}'] = method
    output = [f"# {entry['title']}", '', f"版本：{entry['version']}。本文件为固定官方源码的原样函数摘录；未执行源码。", '',
              f"原始文件：{entry['source_url']}。版权与许可沿用上游仓库 LICENSE。", '']
    for name in entry['symbols']:
        node = symbols[name]
        ranges = constructor_ranges(node, lines, entry['blocks']) if entry.get('blocks') else [('', node.lineno, node.end_lineno)]
        for label, start, end in ranges:
            output += [f"## {name}{' · ' + label if label else ''}", '', f"源码定位：{entry['source_url']}#L{start}-L{end}", '']
            if label:
                output += ['以下为构造函数内部的原样区域，保留条件分支；不是独立可执行函数。', '']
            output += ['```python', '\n'.join(lines[start - 1:end]), '```', '']
    return '\n'.join(output).encode('utf-8')


def download():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    prior = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'entries': []}
    old = {entry['key']: entry for entry in prior['entries']}
    entries = []
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        for recipe in ENTRIES:
            path = ARCHIVE / recipe['filename']
            cached = old.get(recipe['key'])
            blob_valid = path.exists() and recipe.get('git_blob_sha') == hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0' + path.read_bytes()).hexdigest()
            if path.exists() and ((cached and digest(path.read_bytes()) == cached['source_sha256']) or blob_valid):
                raw = path.read_bytes()
            else:
                if path.exists():
                    raise RuntimeError('已有原始资料与下载记录不匹配，请先核查，不自动覆盖。')
                response = client.get(recipe['url'])
                response.raise_for_status()
                raw = response.content
                if recipe['type'] == 'paper' and not raw.startswith(b'%PDF-'):
                    raise RuntimeError('下载内容不是 PDF')
                path.write_bytes(raw)
            if recipe.get('git_blob_sha'):
                assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == recipe['git_blob_sha'], '固定版本源码哈希不匹配'
            prepared = raw if recipe['type'] == 'paper' else code_excerpt(recipe, raw)
            import_path = path if recipe['type'] == 'paper' else ARCHIVE / recipe.get('import_filename', path.with_suffix('.md').name)
            if import_path != path:
                import_path.write_bytes(prepared)
            entries.append({**recipe, 'source_path': str(path), 'import_path': str(import_path), 'source_sha256': digest(raw),
                            'import_sha256': digest(prepared), 'bytes': len(raw),
                            **({key: cached[key] for key in ('document_id', 'part_count') if key in cached}
                               if cached and cached.get('import_sha256') == digest(prepared) else {})})
            MANIFEST.write_text(json.dumps({'date': '2026-10-13', 'entries': entries}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'downloaded': recipe['key'], 'bytes': len(raw), 'cache': bool(cached)}, ensure_ascii=True), flush=True)


def import_documents():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with app_client(timeout=90) as client:
        for entry in manifest['entries']:
            path = Path(entry['import_path'])
            if not path.is_absolute(): path = ROOT / path
            raw = path.read_bytes()
            assert digest(raw) == entry['import_sha256'], '原文校验失败，不入库'
            data = {'model_id': entry['model'], 'pages': entry.get('pages', '')}
            response = client.post('/documents/preview', files={'file': (path.name, raw)}, data=data)
            response.raise_for_status()
            preview = response.json()
            if entry['type'] == 'paper':
                assert preview.get('selected_pages'), '运行后端未应用页码选择，停止入库'
                assert {page for part in preview['parts'] for page in (part.get('chapter_pages') or [part.get('page')]) if page is not None} == set(preview['selected_pages'])
            response = client.post(f"/documents/{preview['id']}/import", json={'title': entry['title'], 'source_url': entry.get('source_url', entry['url']), 'cloud_allowed': True})
            response.raise_for_status()
            entry['document_id'] = preview['id']
            entry['part_count'] = response.json()['part_count']
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'imported': entry['key'], 'id': preview['id'], 'parts': entry['part_count'], 'pages': preview.get('selected_pages')}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--import-only', action='store_true')
    options = parser.parse_args()
    if not options.import_only:
        download()
    if not options.download_only:
        import_documents()
