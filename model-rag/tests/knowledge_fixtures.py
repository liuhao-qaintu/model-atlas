"""内存中模拟目录主题缺少资料，不删除真实 MySQL 记录或向量缓存。"""
import copy
from unittest.mock import patch

from backend.infra.database import KnowledgeModel, KnowledgeSection, KnowledgeRelation, KnowledgeDocument, KnowledgeSource
from backend.rag.retrieval import Retriever


def without_model_material(active, model_id):
    tables = {
        KnowledgeModel: [dict(copy.deepcopy(model), ready=False) if model['id'] == model_id else copy.deepcopy(model)
                         for model in active.models.values()],
        KnowledgeSection: [copy.deepcopy(section) for section in active.sections if section['model_id'] != model_id],
        KnowledgeRelation: [copy.deepcopy(edge) for edge in active.relations if edge.get('owner_model_id') != model_id],
        KnowledgeDocument: [copy.deepcopy(document) for document in active.documents if document['model_id'] != model_id],
        KnowledgeSource: list(copy.deepcopy(active.sources).values()),
    }

    def fixture_records(table):
        return tables[table]

    with patch('backend.rag.retrieval.records', side_effect=fixture_records):
        result = Retriever()
    result.vectors = None
    return result
