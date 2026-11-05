"""补充固定原论文和官方训练教程，原文保留，代码只解析不执行。"""
import ast
import hashlib
import json
import textwrap
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.api_client import app_client
FOLDER = ROOT.parent / '模型知识库资料/公开原始资料/使用与训练'
MANIFEST = ROOT / 'data/product-corpus-manifest.json'
TUTORIALS = [('optimization', 'beginner_source/basics/optimization_tutorial.py', '训练循环与优化器'),
             ('autograd', 'beginner_source/basics/autogradqs_tutorial.py', '自动梯度与反向传播'),
             ('transfer', 'beginner_source/transfer_learning_tutorial.py', '预训练与迁移学习'),
             ('saving', 'beginner_source/saving_loading_models.py', '模型保存、加载与推理模式')]


def digest(raw): return hashlib.sha256(raw).hexdigest()


def tutorial_markdown(raw, title, source, commit):
    text = raw.decode('utf-8'); lines = text.splitlines(); tree = ast.parse(text)
    output = [f'# {title}', '', f'来源：{source}', '', f'固定版本：{commit}。原文来自 PyTorch 官方教程，版权与许可沿用上游仓库 LICENSE；以下仅组织原文注释与代码，未执行或训练模型。', '']
    cursor = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            code = '\n'.join(lines[cursor:node.lineno - 1]).strip()
            if code: output += ['```python', code, '```', '']
            output += [node.value.value.strip(), '']; cursor = node.end_lineno
    tail = '\n'.join(lines[cursor:]).strip()
    if tail: output += ['```python', tail, '```', '']
    return '\n'.join(output).encode('utf-8')


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    prior = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
    with app_client(timeout=180) as app, httpx.Client(timeout=75, follow_redirects=True, trust_env=False) as web:
        old = {entry['key']: entry for entry in prior.get('entries', [])}
        commit = prior.get('tutorial_commit')
        if not commit:
            response = web.get('https://api.github.com/repos/pytorch/tutorials/commits/main'); response.raise_for_status()
            commit = response.json()['sha']
        response = app.get('/models'); response.raise_for_status(); models = response.json()
        training = next((m for m in models if m.get('corpus_role') == 'training' or m['name'] == '模型训练与使用'), None)
        if not training:
            response = app.post('/models', json={'name': '模型训练与使用', 'category': '训练与实践', 'description': '查阅梯度、优化器、迁移学习与模型保存的官方教程。', 'aliases': ['训练基础', 'PyTorch训练'], 'tags': ['训练', '迁移学习', '工程使用']}); response.raise_for_status(); training = response.json()
        recipes = [{'key': 'resnet-paper', 'url': 'https://arxiv.org/pdf/1512.03385v1', 'title': 'ResNet 原始论文 · v1 · 方法、训练与实验', 'filename': 'resnet_1512.03385v1.pdf', 'model': 'resnet-18', 'format': 'pdf'},
                   {'key': 'resnet-official', 'url': 'https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/models/resnet.py', 'title': 'Torchvision ResNet 官方实现 · v0.21.0', 'filename': 'torchvision_resnet_v0.21.0.py', 'model': 'resnet-18', 'format': 'code'}]
        recipes += [{'key': key, 'url': f'https://raw.githubusercontent.com/pytorch/tutorials/{commit}/{path}', 'title': 'PyTorch 官方教程 · ' + title, 'filename': key + '.py', 'model': training['id'], 'format': 'tutorial'} for key, path, title in TUTORIALS]
        recipes += [
            {'key': 'gradient-accumulation', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/docs/source/notes/amp_examples.rst', 'title': 'PyTorch 2.6 梯度累积 · 更新与清空顺序', 'filename': 'amp_examples_v2.6.0.rst', 'model': training['id'], 'format': 'accumulation'},
            {'key': 'zero-grad-api', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/optim/optimizer.py', 'title': 'PyTorch 2.6 Optimizer.zero_grad · 默认None与清空语义', 'filename': 'optimizer_v2.6.0.py', 'model': training['id'], 'format': 'zero_grad'},
        ]
        record = {'tutorial_commit': commit, 'training_model_id': training['id'], 'entries': []}
        MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        for recipe in recipes:
            source = FOLDER / recipe['filename']; cached = old.get(recipe['key'])
            if source.exists():
                raw = source.read_bytes()
                if cached and digest(raw) != cached['source_sha256']: raise RuntimeError('已存原文摘要变化，先核查，不覆盖')
            else:
                response = web.get(recipe['url']); response.raise_for_status(); raw = response.content
                if recipe['format'] == 'pdf' and not raw.startswith(b'%PDF'): raise RuntimeError('原文响应不是PDF')
                source.write_bytes(raw)
            if recipe['format'] == 'pdf': prepared, imported = raw, source
            elif recipe['format'] == 'accumulation':
                text = raw.decode('utf-8'); start = text.index('Gradient accumulation\n'); end = text.index('Gradient penalty\n', start)
                original = text[start:end].strip()
                intro = '以下是依据所附官方原文和代码整理的教学说明：每批次独立更新时，可在本次backward前清空梯度；也可在上次step后清空，为下一批次准备。梯度累积时，先按目标累积多个微批次的backward，在更新边界执行step（AMP场景为scaler.step并update），之后才zero_grad。不能在累积完成后、step之前清空，否则丢失本轮梯度。原示例还对loss除以累积次数，并讨论AMP缩放条件，不能只改变清空频率。'
                prepared = (f"# {recipe['title']}\n\n来源：{recipe['url']}\n\n## 教学读法与条件\n\n{intro}\n\n## 官方原文与代码摘录\n\n```text\n{original}\n```\n").encode('utf-8'); imported = source.with_suffix('.md'); imported.write_bytes(prepared)
            elif recipe['format'] == 'zero_grad':
                text = raw.decode('utf-8'); tree = ast.parse(text)
                owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Optimizer')
                method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'zero_grad')
                original = textwrap.dedent('\n'.join(text.splitlines()[method.lineno - 1:method.end_lineno]))
                intro = 'PyTorch 2.6的Optimizer.zero_grad默认set_to_none=True，清除已有梯度时通常设为None；set_to_none=False才清零已有梯度张量。None与全零张量不是所有优化器都按同一语义处理；例如step可以跳过None梯度的参数，而零梯度仍可能触发更新相关逻辑。应按本页实际接口与所用优化器判断。'
                prepared = (f"# {recipe['title']}\n\n来源：{recipe['url']}#L{method.lineno}-L{method.end_lineno}\n\n## 接口说明\n\n{intro}\n\n## 官方方法原样摘录\n\n以下为类方法摘录，不是可独立运行的完整模块。\n\n```python\n{original}\n```\n").encode('utf-8'); imported = source.with_suffix('.md'); imported.write_bytes(prepared)
            else:
                imported = source.with_suffix('.md')
                prepared = tutorial_markdown(raw, recipe['title'], recipe['url'], commit if recipe['format'] == 'tutorial' else 'v0.21.0')
                imported.write_bytes(prepared)
            response = app.post('/documents/preview', files={'file': (imported.name, prepared)}, data={'model_id': recipe['model']}); response.raise_for_status(); preview = response.json()
            response = app.post(f"/documents/{preview['id']}/import", json={'title': recipe['title'], 'source_url': recipe['url'], 'cloud_allowed': True}); response.raise_for_status()
            entry = {**recipe, 'source_path': str(source), 'import_path': str(imported), 'source_sha256': digest(raw), 'import_sha256': digest(prepared), 'document_id': preview['id'], 'part_count': response.json()['part_count']}
            record['entries'].append(entry); MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'key': recipe['key'], 'parts': entry['part_count'], 'duplicate': preview.get('duplicate', False)}), flush=True)
        response = app.post('/documents/index', json={}); response.raise_for_status()
        print(json.dumps({'index': response.json()}), flush=True)


if __name__ == '__main__': main()
