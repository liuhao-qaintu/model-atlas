"""按源码与登记公开资料白名单生成本地交付包，不读取数据库或配置值。"""
import argparse
import ast
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DOCS = []  # 内部修改记录和面试材料不进入公开交付包。
TOOLS = ['import_public_sources.py','export_project.py','expand_product_corpus.py','prepare_official_interfaces.py','manage_accounts.py','api_client.py','run_tests.py']
ENV_TEMPLATE = '''MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=model_rag
MYSQL_USER=root
MYSQL_PASSWORD=
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=
RERANK_MODEL_NAME=Qwen/Qwen3-Reranker-0.6B
RERANK_BASE_URL=https://api.siliconflow.cn/v1
RERANK_API_KEY=
LLM_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=
AUTH_SECRET_KEY=
AUTH_EXPIRE_HOURS=24
AUTH_COOKIE_SECURE=false
ENABLE_API_DOCS=false
MAINTAINER_USER_IDS=[]
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def within_file(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('导出文件必须位于指定资料或源码目录内')
    return path


def topic_files(root):
    files = []
    for relative, name in [('backend/infra/knowledge.py', 'TOPICS'), ('backend/infra/recent_knowledge.py', 'RECENT_TOPICS')]:
        source = root / relative
        if not source.exists(): continue
        for node in ast.parse(source.read_text(encoding='utf-8')).body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                files.extend(topic['file'] for topic in ast.literal_eval(node.value))
    if not files or any(Path(file).name != file or not file.endswith('.md') for file in files):
        raise ValueError('模型卡目录不完整或路径无效')
    return list(dict.fromkeys(files))


def build_archive(root, content, destination):
    root, content, destination = Path(root).resolve(), Path(content).resolve(), Path(destination).resolve()
    if destination.exists(): raise FileExistsError('保留已有交付包，请指定另一个输出文件名')
    files = {}; prefix = 'ModelAtlas/model-rag'; corpus = 'ModelAtlas/' + content.name

    def put(name, raw):
        if name in files and files[name] != raw: raise ValueError('同名导出内容不一致')
        if not name.endswith(('.pdf', '.png')) and re.search(rb'\bsk-[A-Za-z0-9_-]{24,}', raw):
            raise ValueError('文本文件疑似含服务密钥，停止导出，请先核对源文件')
        files[name] = raw

    def copy(path, base, archive_base, expected=None):
        path = within_file(path, base); raw = path.read_bytes()
        if expected and sha(raw) != expected: raise ValueError('登记资料摘要不匹配，停止导出')
        relative = path.relative_to(base).as_posix()
        if path.suffix == '.md':
            text = raw.decode('utf-8')
            if base == root: text = text.replace(root.as_posix() + '/', '../')
            else: text = text.replace(content.as_posix() + '/', '')
            raw = text.encode('utf-8')
        put(archive_base + '/' + relative, raw)
        return relative

    for name in ['requirements.txt', '.gitignore', 'start-backend.ps1', 'start-frontend.ps1',
                 'frontend/package.json', 'frontend/package-lock.json', 'frontend/index.html', 'frontend/vite.config.js']:
        copy(root / name, root, prefix)
    for path in (root / 'backend').rglob('*.py'): copy(path, root, prefix)
    for path in (root / 'frontend/src').rglob('*'):
        if path.is_file() and path.suffix in {'.vue', '.js', '.css'}: copy(path, root, prefix)
    for path in (root / 'tests').glob('*.py'): copy(path, root, prefix)
    fixture = root / 'tests/fixtures/research_position_plan.json'
    if fixture.exists(): copy(fixture, root, prefix)
    for path in (root / 'frontend/tests').glob('*.js'): copy(path, root, prefix)
    for name in TOOLS: copy(root / 'tools' / name, root, prefix)
    for name in DOCS: copy(root / 'docs' / name, root, prefix)
    copy(root / 'data/evaluation-cases.json', root, prefix)
    copy(root / 'data/evaluation-challenge-cases.json', root, prefix)
    if (root / 'data/answer-evaluation-cases.json').exists(): copy(root / 'data/answer-evaluation-cases.json', root, prefix)
    cards = topic_files(root)
    for name in cards: copy(content / '模型卡' / name, content, corpus)
    for name in ['首批资料清单.md', '公开原始资料清单.md']:
        copy(content / name, content, corpus)

    manifest_path = root / 'data/public-source-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'entries': []}
    for entry in manifest['entries']:
        for field, expected in [('source_path', 'source_sha256'), ('import_path', 'import_sha256')]:
            path = Path(entry[field])
            if not path.is_absolute(): path = root / path
            relative = copy(path, content, corpus, entry[expected])
            entry[field] = '../' + content.name + '/' + relative
    put(prefix + '/data/public-source-manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    extra_count = 0
    for manifest_name in ('product-corpus-manifest.json', 'interface-source-manifest.json'):
        additional = root / 'data' / manifest_name
        if not additional.exists():
            continue
        extra = json.loads(additional.read_text(encoding='utf-8'))
        for entry in extra['entries']:
            for field, expected in [('source_path', 'source_sha256'), ('import_path', 'import_sha256')]:
                path = Path(entry[field])
                if not path.is_absolute(): path = root / path
                relative = copy(path, content, corpus, entry[expected])
                entry[field] = '../' + content.name + '/' + relative
        extra_count += len(extra['entries'])
        put(prefix + '/data/' + manifest_name, (json.dumps(extra, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    figure_manifest = root / 'data/figure-manifest.json'
    figures = json.loads(figure_manifest.read_text(encoding='utf-8')) if figure_manifest.exists() else {'figures': []}
    for figure in figures['figures']:
        copy(root / 'data/figures' / figure['file'], root / 'data', prefix + '/data', figure['sha256'])
    put(prefix + '/data/figure-manifest.json', (json.dumps(figures, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    put(prefix + '/.env.example', ENV_TEMPLATE.encode('utf-8'))
    copy(root / 'README.md', root, prefix)
    summary = {'date': date.today().isoformat(), 'model_cards': len(cards), 'source_documents': len(manifest['entries']) + extra_count,
               'original_figures': len(figures['figures']),
               'files': [{'path': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(files.items())]}
    put('ModelAtlas/export-manifest.json', (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(files.items()): archive.writestr(name, raw)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None: raise ValueError('交付包完整性校验失败')
    result = {**summary, 'archive': str(destination), 'archive_bytes': destination.stat().st_size,
              'archive_sha256': sha(destination.read_bytes())}
    destination.with_suffix('.manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'exports/ModelAtlas_Source.zip')
    options = parser.parse_args()
    result = build_archive(ROOT, ROOT.parent / '模型知识库资料', options.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=True))


if __name__ == '__main__': main()
