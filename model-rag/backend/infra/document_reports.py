"""文档级报告保存及一次确认标签；正文可检索和图谱批准互不阻塞。"""
import hashlib
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.infra.database import KnowledgeAnalysis, KnowledgeDocument, KnowledgeModel, KnowledgeRelation, KnowledgeSection
from backend.infra.access import current_actor
from backend.rag.document_report import evidence_name, target_version_supported


def latest_report(document_id):
    from backend.infra.database import records
    rows = [r for r in records(KnowledgeAnalysis) if r.get('type') == 'document_report' and r.get('document_id') == document_id]
    return max(rows, key=lambda r: r.get('created_at', '')) if rows else None


def retire_placeholders(document_id):
    from backend.infra.database import engine
    with Session(engine) as db:
        for row in db.query(KnowledgeAnalysis).all():
            value = row.payload
            if (value.get('type') == 'review_item' and value.get('document_id') == document_id
                    and value.get('status') == 'pending'
                    and (value.get('proposed', {}).get('classification') or {}).get('skipped')):
                row.payload = {**value, 'status': 'superseded', 'superseded_reason': 'document_report_flow'}
        db.commit()


def confirm_report(report_id, selected_ids):
    from backend.infra.database import engine
    with Session(engine) as db:
        row = db.get(KnowledgeAnalysis, report_id, with_for_update=True)
        if not row or row.payload.get('type') != 'document_report':
            raise LookupError('分析报告不存在')
        report = dict(row.payload)
        document_row = db.get(KnowledgeDocument, report['document_id'], with_for_update=True)
        if not document_row or document_row.payload.get('status') != 'imported':
            raise ValueError('报告不属于当前文档版本，请重新分析')
        if report.get('status') == 'approved':
            return report
        tags = {tag['id']: tag for tag in report.get('tags', [])}
        if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - tags.keys():
            raise ValueError('选择的标签不存在或重复')
        document = dict(document_row.payload)
        owner = db.get(KnowledgeModel, document['model_id'])
        if owner is None:
            raise ValueError('文档对应的知识节点不存在')
        accepted, now = [], datetime.now(timezone.utc).isoformat()
        for tag_id in selected_ids:
            tag = tags[tag_id]
            verified = []
            for proof in tag.get('proofs', []):
                section = db.get(KnowledgeSection, proof['section_id'])
                quote = proof.get('quote', '')
                if (not section or section.payload.get('document_id') != document['id']
                        or not quote or quote not in section.payload.get('content', '')):
                    raise ValueError('标签依据已变化，请重新生成分析报告')
                verified.append(proof)
            if not verified or not evidence_name(tag['name'], tag.get('aliases', []), verified):
                raise ValueError('标签缺少原文命名依据')
            target_id = tag.get('model_id')
            if target_id:
                target = db.get(KnowledgeModel, target_id)
                if not target or not evidence_name(target.payload['name'], target.payload.get('aliases', []), verified) or not target_version_supported(target.payload, verified):
                    raise ValueError('已有模型链接缺少原文别名依据')
            else:
                actor = current_actor.get()
                namespace = actor.id + ':' if actor is not None else ''
                target_id = 'tag-' + hashlib.sha256((namespace + tag['kind'] + ':' + tag['name'].casefold()).encode()).hexdigest()[:24]
                target = db.get(KnowledgeModel, target_id)
                if not target:
                    payload = {'id': target_id, 'name': tag['name'], 'kind': '知识标签', 'category': tag['kind'],
                               'aliases': [tag['name'], *tag.get('aliases', [])], 'tags': [tag['kind']],
                               'description': '来自已确认文献分析的主题标签；不表示独立模型实现。',
                               'origin': 'document_report', 'order': 3000, 'ready': False, 'section_count': 0, 'source_ids': []}
                    target = KnowledgeModel(id=target_id, payload=payload)
                    db.add(target)
            if target_id != document['model_id']:
                relation_id = 'dr-' + hashlib.sha256((document['id'] + ':' + target_id).encode()).hexdigest()[:24]
                relation = {'id': relation_id, 'source': document['model_id'], 'target': target_id,
                            'source_name': owner.payload['name'], 'target_name': target.payload['name'],
                            'owner_model_id': document['model_id'], 'document_id': document['id'],
                            'relation': '提及' if tag.get('model_id') else '讨论主题', 'kind': '文献主题', 'origin': 'user',
                            'status': 'accepted', 'review_status': 'approved', 'source_ids': [f"D{document['id']}"],
                            'evidence': [{'section_id': p['section_id'], 'quote': p['quote']} for p in verified],
                            'locator': verified[0].get('locator', ''), 'report_id': report_id, 'updated_at': now}
                db.merge(KnowledgeRelation(id=relation_id, payload=relation))
            accepted.append({**tag, 'graph_node_id': target_id})
        # 重复分析后确认新的标签集：旧报告的关系保留历史但退出当前图谱。
        retained = {tag['graph_node_id'] for tag in accepted}
        for edge in db.query(KnowledgeRelation).all():
            payload = edge.payload
            if payload.get('document_id') == document['id'] and payload.get('report_id') and payload.get('target') not in retained:
                edge.payload = {**payload, 'status': 'rejected', 'updated_at': now}
        document_row.payload = {**document, 'analysis_report_id': report_id, 'analysis_tags': accepted}
        if owner.payload.get('kind') == '文档':
            owner.payload = {**owner.payload, 'name': (document.get('title') or owner.payload['name'])[:80], 'tags': [tag['name'] for tag in accepted], 'description': report['content']['summary'][0]['text'][:500]}
        report.update(status='approved', confirmed_at=now, selected_tag_ids=list(selected_ids), confirmed_tags=accepted)
        row.payload = report
        db.commit()
        return report
