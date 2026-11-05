"""构造所选文献的首轮范围；允许补查时由scope_expansion有界扩展。"""


def selected_documents(request, retriever):
    identities = list(dict.fromkeys([*(getattr(request, 'document_ids', []) or []), *([request.document_id] if request.document_id else [])]))
    identity = identities[0] if identities else None
    if not identity and request.model_id and retriever.models[request.model_id].get('kind') == '文档':
        matches = [d for d in retriever.documents if d['model_id'] == request.model_id]
        if len(matches) == 1:
            identity = matches[0]['id']
            identities = [identity]
    return [d for identity in identities for d in retriever.documents if d['id'] == identity]


def selected_document(request, retriever):
    return next(iter(selected_documents(request, retriever)), None)


def pin_document_plan(plan, document):
    documents = document if isinstance(document, list) else [document]
    ids = [d['id'] for d in documents]
    owners = list(dict.fromkeys(d['model_id'] for d in documents))
    owner = owners[0] if len(owners) == 1 else None
    queries = [{**q, 'model_id': owner} for q in plan.get('queries', [])] or [{'model_id': owner, 'query': plan['normalized_query']}]
    if len(documents) > 1:
        combined = '；'.join(dict.fromkeys(q['query'] for q in queries))
        queries = [{'id': f'd{i+1}', 'model_id': d['model_id'], 'document_id': d['id'], 'query': combined}
                   for i, d in enumerate(documents)]
    return {**plan, 'document_id': ids[0] if len(ids) == 1 else None, 'document_ids': ids, 'model_id': owner, 'model_ids': owners,
            'section_id': plan.get('section_id') if len(ids) == 1 and str(plan.get('section_id', '')).startswith('doc:' + ids[0] + ':') else None,
            'mode': 'comparison' if len(queries) > 1 else 'single', 'queries': queries, 'relation_route': [],
            'clarification': None, 'context_origin': 'selected_document',
            'source_scope': {'kind': 'document', 'label': '所选文献', 'origin': 'selected_document', 'strict': False, 'document_id': ids[0] if len(ids) == 1 else None, 'document_ids': ids},
            'source_kinds': [], 'excluded_source_kinds': [], 'remote_allowed': all(d.get('cloud_allowed', False) for d in documents)}
