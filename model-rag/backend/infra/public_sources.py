"""只给本次核对的固定来源补上原文摘录；不自动为用户上传推断关系。"""
import json

from backend.infra.config import DATA_ROOT
from backend.infra.database import KnowledgeDocument, KnowledgeSection, records

SUPPORTS = {
    'clip-vision': ('clip-code', 'self.visual = VisionTransformer('),
    'clip-vit-family': ('clip-code', 'self.visual = VisionTransformer('),
    'clip-class-token': ('clip-code', 'self.class_embedding'),
    'clip-patches': ('clip-code', 'kernel_size=patch_size, stride=patch_size'),
}


def document_kinds(documents):
    entries = {}
    for name in ('public-source-manifest.json', 'product-corpus-manifest.json', 'interface-source-manifest.json'):
        path = DATA_ROOT / name
        if path.exists(): entries.update({entry.get('document_id'): entry for entry in json.loads(path.read_text(encoding='utf-8'))['entries']})
    kinds = {}
    for doc in documents:
        entry = entries.get(doc.get('provenance_document_id', doc['id']))
        if entry and doc.get('sha256') == entry.get('import_sha256') and doc['model_id'] == entry['model']:
            kinds[doc['id']] = entry.get('type') or {'pdf': 'paper', 'code': 'official_code', 'zero_grad': 'official_code',
                'tutorial': 'official_tutorial', 'accumulation': 'official_document'}.get(entry.get('format'), 'document')
    return kinds


def expand_definitions(retriever, result, limit):
    """仅补已登记且摘要/归属匹配的调用定义，不扩展用户上传的任意源码。"""
    path = DATA_ROOT / 'public-source-manifest.json'
    if not path.exists() or not result['evidence']:
        return result
    entries = json.loads(path.read_text(encoding='utf-8'))['entries']
    by_key = {entry['key']: entry for entry in entries}
    kinds = document_kinds(retriever.documents)
    docs = {doc['id']: doc for doc in retriever.documents}
    active_ids = {doc.get('provenance_document_id', doc['id']): doc['id'] for doc in retriever.documents}
    expansions = []
    for entry in entries:
        dependency = entry.get('definition_dependency')
        target_id = active_ids.get(entry.get('document_id'), entry.get('document_id'))
        if not dependency or kinds.get(target_id) != 'official_code':
            continue
        source = by_key.get(dependency['source_key'], {})
        source_id = active_ids.get(source.get('document_id'), source.get('document_id'))
        if kinds.get(source_id) != 'official_code' or source.get('model') != entry['model']:
            continue
        if any(docs.get(active_ids.get(spec['document_id'], spec['document_id']), {}).get('source_url') != spec.get('source_url', spec.get('url')) for spec in (source, entry)):
            continue
        trigger = next((e for e in result['evidence'] if e.get('document_id') == source_id
                        and e['model_id'] == entry['model'] and dependency['call_quote'] in e['content']), None)
        chunk = next((c for c in retriever.chunks if c.get('document_id') == target_id
                      and c['model_id'] == entry['model'] and dependency['definition_quote'] in c['parent_content']), None)
        if not trigger or not chunk:
            continue
        definition = next((e for e in result['evidence'] if e['section_id'] == chunk['section_id']), None)
        if definition is None:
            definition = {**chunk, 'content': chunk['parent_content'], 'channels': ['定义补全'], 'score': 0,
                          'sources': [retriever.sources[s] for s in chunk['source_ids'] if s in retriever.sources]}
            definition.pop('parent_content', None)
            if len(result['evidence']) < limit:
                result['evidence'].append(definition)
            else:
                replace = next((i for i in range(len(result['evidence']) - 1, -1, -1)
                                if result['evidence'][i]['section_id'] != trigger['section_id']), None)
                if replace is None:
                    continue
                result['evidence'][replace] = definition
        expansions.append({'trigger_section_id': trigger['section_id'], 'trigger_url': trigger['reader_url'],
                           'definition_section_id': definition['section_id'], 'definition_url': definition['reader_url'],
                           'definition_title': definition['title'], 'call_quote': dependency['call_quote']})
    if expansions:
        result['definition_expansions'] = expansions
        result['channels']['定义补全'] = len(expansions)
    return result


def enrich_relations(relations):
    manifest_path = DATA_ROOT / 'public-source-manifest.json'
    if not manifest_path.exists():
        return relations
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    documents = {d['id']: d for d in records(KnowledgeDocument) if d['status'] == 'imported'}
    sections = records(KnowledgeSection)
    entries = {item['key']: item for item in manifest['entries']}
    for edge in relations:
        spec = SUPPORTS.get(edge['id'])
        if not spec:
            continue
        key, quote = spec
        entry = entries.get(key, {})
        document = documents.get(entry.get('document_id'))
        if not document or document['sha256'] != entry.get('import_sha256') or document['model_id'] != edge['owner_model_id']:
            continue
        if document.get('source_url') != entry.get('source_url', entry.get('url')):
            continue
        proof = next((s for s in sections if s.get('document_id') == document['id'] and quote in s['content']), None)
        if proof:
            edge['evidence'] = [{'section_id': proof['id'], 'quote': quote}, *edge['evidence']]
            edge['source_ids'] = list(dict.fromkeys([f"D{document['id']}", *edge['source_ids']]))
    return relations
