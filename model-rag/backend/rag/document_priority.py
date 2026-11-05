"""相关候选的小幅文献先验：首次入库时间衰减和人工优先级，不新增召回。"""
import math
from datetime import datetime, timezone

POLICY = 'document-priority-v1'
FRESHNESS_MAX, HALF_LIFE_DAYS, MAX_FACTOR = 0.12, 7, 1.30


def priority_day():
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def priority_factor(document, relevant, now):
    if not document or not relevant:
        return {'factor': 1.0, 'manual': 1.0, 'freshness': 0.0}
    try:
        manual = float(document.get('retrieval_priority', 1.0))
        manual = max(1.0, min(1.25, manual)) if math.isfinite(manual) else 1.0
    except (TypeError, ValueError):
        manual = 1.0
    freshness = 0.0
    raw = document.get('retrieval_added_at') or document.get('imported_at')
    try:
        added = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if added.tzinfo is None:
            added = added.replace(tzinfo=timezone.utc)
        age = max(0.0, (now - added).total_seconds() / 86400)
        freshness = FRESHNESS_MAX * 2 ** (-age / HALF_LIFE_DAYS)
    except (ValueError, TypeError, AttributeError):
        pass
    return {'factor': min(MAX_FACTOR, manual * (1 + freshness)), 'manual': manual, 'freshness': freshness}


def prioritize(candidates, documents, now, score_key='score', protected=()):
    docs = {d['id']: d for d in documents}
    output = []
    usable = all(type(c.get(score_key)) in (int, float) and math.isfinite(c[score_key]) for c in candidates)
    for item in candidates:
        adjustment = priority_factor(docs.get(item.get('document_id')), item.get('retrieval_relevant', False), now)
        base = item.get(score_key)
        effective = base * adjustment['factor'] if usable and base > 0 else base
        output.append({**item, 'retrieval_adjustment': {**adjustment, 'policy': POLICY, 'score_kind': score_key,
                       'applied': usable and base > 0 and adjustment['factor'] > 1, 'base_score': base,
                       'ranking_score': effective}})
    if usable:
        protected = set(protected)
        output.sort(key=lambda c: (bool(protected) and c['section_id'] not in protected,
                                  -c['retrieval_adjustment']['ranking_score']))
    return output
