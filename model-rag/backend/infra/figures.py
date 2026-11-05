"""仅提供登记的原论文插图，保存页码、版本和原文件摘要。"""
import hashlib
import json
from pathlib import Path

from backend.infra.config import DATA_ROOT


def figure_records(model_id=None):
    manifest = DATA_ROOT / 'figure-manifest.json'
    if not manifest.is_file():
        return []
    records = json.loads(manifest.read_text(encoding='utf-8'))['figures']
    return [{key: value for key, value in figure.items() if key not in {'file', 'pdf_rect'}} |
            {'image_url': f"/api/figures/{figure['id']}/image", 'reader_url': f"/models/{figure['model_id']}?section=figures"}
            for figure in records if model_id is None or figure['model_id'] == model_id]


def figure_path(identity):
    manifest = DATA_ROOT / 'figure-manifest.json'
    figures = json.loads(manifest.read_text(encoding='utf-8'))['figures'] if manifest.is_file() else []
    figure = next((f for f in figures if f['id'] == identity), None)
    if not figure:
        raise LookupError('插图不存在')
    folder = (DATA_ROOT / 'figures').resolve()
    path = (folder / figure['file']).resolve()
    if path.parent != folder or path.suffix != '.png' or not path.is_file():
        raise LookupError('插图文件不存在')
    if hashlib.sha256(path.read_bytes()).hexdigest() != figure['sha256']:
        raise ValueError('插图与登记摘要不一致')
    return path


def answer_figures(evidence):
    models = list(dict.fromkeys(e['model_id'] for e in evidence if e.get('model_id')))
    return [figure for mid in models for figure in figure_records(mid)][:3]
