"""资料分析和原图接口；读取当前检索实例，避免使用过期目录。"""
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.rag.analysis import extract_relations, get_analysis, review_relation, code_catalog, compare_code
from backend.rag.analysis import EXTRACTION_VERSION
from backend.infra.figures import figure_path
from backend.rag.retrieval import Retriever

router = APIRouter(prefix='/api')
analysis_lock = asyncio.Lock()


def service():
    from backend import main
    return main


@router.get('/figures/{figure_id}/image')
def figure_image(figure_id: str):
    try:
        return FileResponse(figure_path(figure_id), media_type='image/png')
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/analysis/code')
def code_options():
    return code_catalog(service().retriever)


@router.get('/analysis/status')
def analysis_status():
    return {'relation_extraction': EXTRACTION_VERSION, 'code_comparison': 'python-ast-v1'}


class CodeComparisonRequest(BaseModel):
    left: str = Field(min_length=1, max_length=120)
    right: str = Field(min_length=1, max_length=120)


@router.post('/analysis/code/compare')
def code_comparison(request: CodeComparisonRequest):
    try:
        return compare_code(request.left, request.right, service().retriever)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post('/analysis/documents/{document_id}/relations')
async def relation_extraction(document_id: str):
    try:
        async with analysis_lock:
            return await extract_relations(document_id, service().retriever)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get('/analysis/relations/{analysis_id}')
def relation_analysis(analysis_id: str):
    result = get_analysis(analysis_id)
    if not result:
        raise HTTPException(404, '提取记录不存在')
    return result


class RelationReviewRequest(BaseModel):
    decision: str = Field(pattern='^(accepted|rejected)$')


@router.post('/analysis/relations/{analysis_id}/{candidate_id}/review')
async def relation_review(analysis_id: str, candidate_id: str, request: RelationReviewRequest):
    main = service()
    try:
        async with main.knowledge_lock:
            result = await run_in_threadpool(review_relation, analysis_id, candidate_id, request.decision, main.retriever)
            main.replace_retriever(await run_in_threadpool(Retriever))
        return result
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
