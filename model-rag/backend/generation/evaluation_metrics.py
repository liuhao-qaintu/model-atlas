"""评测口径 v2：待判断不等于无关，辅助标注不等于独立复核。"""
from copy import deepcopy
import hashlib
import json
from backend.generation.answer_assessment import assessment_summary

PROTOCOL = 'relevance-pool-v2'
POSITIVE = {'direct', 'supporting'}
DIMENSIONS = ('correctness', 'completeness', 'faithfulness', 'conditions')


def qrels_key(case, corpus):
    fields = ('id', 'question', 'model_id', 'reference_answer', 'relevant_section_ids', 'prior_turns', 'category')
    raw = json.dumps({'case': {k: case.get(k) for k in fields}, 'corpus': corpus}, ensure_ascii=False, sort_keys=True)
    return 'eq-' + hashlib.sha256(raw.encode()).hexdigest()[:28]


def reuse_judgments(case, corpus, pools):
    saved = pools.get(qrels_key(case, corpus))
    return deepcopy(saved['judgments']) if saved else seed_judgments(case)


def annotation_fingerprint(rows):
    raw = json.dumps([(r['case'].get('id'), r.get('judgments', {})) for r in rows], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def seed_judgments(case):
    return {sid: {'label': 'direct', 'origin': case.get('annotation_origin', 'assistant'),
                  'note': '题库预标注，未证明相关集合完整。'} for sid in case.get('relevant_section_ids', [])}


def retrieval_metrics(retrieved, relevant, k=5, judgments=None):
    if k < 1: raise ValueError('k必须为正数')
    result = list(dict.fromkeys(retrieved))[:k]
    pool = seed_judgments({'relevant_section_ids': relevant}) if judgments is None else judgments
    expected = {sid for sid, item in pool.items() if item['label'] in POSITIVE}
    hits = [i for i, sid in enumerate(result, 1) if sid in expected]
    unresolved = [sid for sid in result if pool.get(sid, {}).get('label', 'unjudged') == 'unjudged']
    count, unknown, size = len(hits), len(unresolved), len(result)
    complete = not unknown
    human = sum(pool.get(sid, {}).get('origin') == 'user' and sid not in unresolved for sid in result)
    return {'protocol': PROTOCOL, 'retrieved': size, 'matched': count, 'relevant': len(expected),
            'unjudged': unknown, 'judged': size - unknown, 'user_judged': human,
            'judgment_coverage': (size - unknown) / size if size else None,
            'fully_judged': complete, 'fully_user_reviewed': complete and size > 0 and human == size,
            'precision_at_5': count / k if complete else None,
            'returned_precision': count / size if complete and size else None,
            'precision_lower_bound': count / k, 'precision_upper_bound': (count + unknown) / k,
            'returned_precision_lower_bound': count / size if size else None,
            'returned_precision_upper_bound': (count + unknown) / size if size else None,
            # 这三项只针对已知相关集合，不能声称是整个知识库的命中/召回。
            'hit_at_5': int(bool(hits)) if expected else None,
            'recall_at_5': count / len(expected) if expected else None,
            'reciprocal_rank': 1 / hits[0] if hits else (0 if expected else None)}


def aggregate(rows):
    scored = [row for row in rows if row.get('metrics')]
    names = ('hit_at_5', 'precision_at_5', 'returned_precision', 'recall_at_5', 'reciprocal_rank',
             'precision_lower_bound', 'precision_upper_bound')
    values = {name: [row['metrics'][name] for row in scored if row['metrics'].get(name) is not None] for name in names}
    grades = [row['grade'] for row in rows if row.get('grade')]
    returned = sum(row['metrics'].get('retrieved', 0) for row in scored)
    judged = sum(row['metrics'].get('judged', 0) for row in scored)
    dimensions = {}
    for name in DIMENSIONS:
        marks = [row.get('answer_rubric', {}).get(name) for row in rows]
        marks = [mark for mark in marks if mark in {'correct', 'partial', 'incorrect'}]
        dimensions[name] = {'count': len(marks), 'pass_rate': marks.count('correct') / len(marks) if marks else None}
    reviewed = [row['metrics'] for row in scored if row['metrics'].get('fully_user_reviewed')]
    user_grades = [row['grade'] for row in rows if row.get('grade') and row.get('grade_origin') == 'user']
    return {'protocol': PROTOCOL, 'evaluated': len(scored), 'failed': sum(bool(row.get('error')) for row in rows),
            'unscorable': sum(row.get('unscorable', False) for row in rows),
            **{name: sum(items) / len(items) if items else None for name, items in values.items()},
            'metric_counts': {name: len(items) for name, items in values.items()},
            'retrieved_sections': returned, 'judged_sections': judged, 'unjudged_sections': returned - judged,
            'judgment_coverage': judged / returned if returned else None,
            'fully_judged_rows': sum(bool(row['metrics'].get('fully_judged')) for row in scored),
            'user_reviewed_rows': len(reviewed),
            'user_reviewed_returned_precision': sum(m['returned_precision'] for m in reviewed) / len(reviewed) if reviewed else None,
            'dataset_splits': sorted({row.get('case', {}).get('split', 'development') for row in rows}),
            'graded_answers': len(grades), 'answer_accuracy': grades.count('correct') / len(grades) if grades else None,
            'user_graded_answers': sum(bool(row.get('grade')) and row.get('grade_origin') == 'user' for row in rows),
            'assistant_graded_answers': sum(bool(row.get('grade')) and row.get('grade_origin') == 'assistant' for row in rows),
            'user_answer_accuracy': user_grades.count('correct') / len(user_grades) if user_grades else None,
            'answer_assessment': assessment_summary(rows),
            'partly_correct': grades.count('partial'), 'incorrect': grades.count('incorrect'), 'answer_dimensions': dimensions,
            'confidence': None, 'confidence_note': '检索得分未经概率校准，尚无可报告的答案置信度。'}


def rescore_row(row):
    row.setdefault('judgments', seed_judgments(row['case']))
    if 'evidence' in row and not row.get('error'):
        row['metrics'] = retrieval_metrics([e['section_id'] for e in row['evidence']],
                                           row['case'].get('relevant_section_ids', []), judgments=row['judgments'])
        if row['case'].get('category') == 'clarification' or row['case'].get('answer_contract', {}).get('evidence_required') is False:
            row['metrics']['retrieval_applicable'] = False
            row['metrics']['fully_judged'] = False
            for key in ('precision_at_5', 'returned_precision', 'precision_lower_bound', 'precision_upper_bound',
                        'returned_precision_lower_bound', 'returned_precision_upper_bound', 'hit_at_5', 'recall_at_5', 'reciprocal_rank'):
                row['metrics'][key] = None
    if row.get('answer_evidence_details') is not None:
        row['answer_retrieval_metrics'] = retrieval_metrics([e['section_id'] for e in row['answer_evidence_details']],
                                                           row['case'].get('relevant_section_ids', []), judgments=row['judgments'])


def normalize_run(record):
    """历史记录在读取时升级显示；原分数留存，不写回数据库。"""
    record = deepcopy(record)
    if record.get('protocol') != PROTOCOL:
        record.setdefault('legacy_summary', deepcopy(record.get('summary', {})))
        for row in record['rows']:
            row.setdefault('legacy_metrics', deepcopy(row.get('metrics')))
    for row in record['rows']:
        row.setdefault('judgments', seed_judgments(row['case']))
        row.setdefault('judgment_history', [])
        rescore_row(row)
    record['protocol'] = PROTOCOL
    record['qrels_sha256'] = annotation_fingerprint(record['rows'])
    record['summary'] = aggregate(record['rows'])
    return record
