"""真正运行StateGraph；模型替身只提供确定输出，不接触生产数据库。"""
import asyncio
import copy
import json
import unittest
import httpx
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.rag.research import ResearchRunner, allowed_scope, connection_diagnostics, render_report, validate_assessment, validate_outline
from backend.api import research


class ResearchFixture:
    def __init__(self):
        self.models = {mid: {'id': mid, 'name': mid.upper(), 'aliases': [mid]} for mid in ('clip', 'llava', 'private')}
        self.sections = [
            {'id': 'clip:io', 'model_id': 'clip', 'content': 'CLIP receives image and text.', 'cloud_allowed': True},
            {'id': 'clip:train', 'model_id': 'clip', 'content': 'CLIP uses contrastive training.', 'cloud_allowed': True},
            {'id': 'llava:io', 'model_id': 'llava', 'content': 'LLaVA accepts images and instructions. LLaVA uses visual instruction tuning.', 'cloud_allowed': True},
            {'id': 'secret', 'model_id': 'private', 'document_id': 'private-doc', 'content': 'PRIVATE SECRET', 'cloud_allowed': False},
        ]
        for s in self.sections:
            s.update(title=s['id'], source_ids=['source'], slug='io')
        self.documents = [{'id': 'private-doc', 'cloud_allowed': False}]
        self.document_kinds = {}
        self.sources = {'source': {'kind': 'internal', 'url': '/models/clip'}}
        self.relations = []
        self.search_calls = []

    async def search(self, query, focus, **kwargs):
        self.search_calls.append((query, focus, kwargs))
        identity = 'clip:train' if 'contrastive' in query else 'clip:io' if focus == 'clip' else 'llava:io'
        section = next(s for s in self.sections if s['id'] == identity)
        return {'evidence': [{**section, 'section_id': identity, 'retrieval_excerpt': section['content']}]}


def outline():
    return {'model_ids': ['clip', 'llava'], 'dimensions': [{'id': 'input', 'label': '输入'}, {'id': 'objective', 'label': '训练目标'}],
            'queries': [{'tool': 'search', 'query': 'CLIP input', 'model_id': 'clip', 'unit_ids': ['clip|input', 'clip|objective']},
                        {'tool': 'search', 'query': 'LLaVA input objective', 'model_id': 'llava', 'unit_ids': ['llava|input', 'llava|objective']}]}


async def fake_model(purpose, payload):
    if purpose == 'plan':
        return outline()
    if purpose == 'assess':
        available = {e['section_id']: e for e in payload['actual_evidence']}
        checks = []
        for unit in payload['units']:
            sid = 'clip:train' if unit['id'] == 'clip|objective' else 'clip:io' if unit['model_id'] == 'clip' else 'llava:io'
            checks.append({'unit_id': unit['id'], 'status': 'supported' if sid in available else 'missing',
                           'quotes': [{'section_id': sid, 'passage_id': available[sid]['passages'][0]['id']}] if sid in available else [],
                           'note': '本轮原文有直接依据' if sid in available else '需要补查训练目标'})
        actions = [] if 'clip:train' in available else [{'tool': 'search', 'query': 'CLIP contrastive training', 'model_id': 'clip', 'unit_ids': ['clip|objective']}]
        return {'checks': checks, 'decision': 'continue' if actions else 'finish', 'actions': actions}
    return {'rows': [[c['unit_id'], '依据资料解释输入或训练目标。'] for c in payload['checks']]}


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_failure_keeps_safe_diagnostic_codes(self):
        runner = ResearchRunner(ResearchFixture())
        failure = httpx.ConnectError('PRIVATE SECRET URL')
        failure.__cause__ = ExceptionGroup('connection attempts', [PermissionError(13, 'PRIVATE SECRET')])
        with patch('backend.rag.research.provider_post', new_callable=AsyncMock, side_effect=failure):
            with self.assertRaises(httpx.ConnectError):
                await runner.ask('plan', 'JSON', {}, 100)
        chain = runner.diagnostics[0]['connection']
        self.assertTrue(any(row['errno'] == 13 for row in chain))
        self.assertNotIn('PRIVATE SECRET', json.dumps(runner.diagnostics))

    async def test_stream_connection_error_is_actionable_and_does_not_save(self):
        with patch.object(ResearchRunner, 'run', new_callable=AsyncMock, side_effect=httpx.ConnectError('PRIVATE SECRET')), \
                patch.object(research, 'save_report') as save:
            events = research.research_events(research.ResearchInput(question='比较'), ResearchFixture())
            name, value = await anext(events)
            self.assertEqual(name, 'error')
            self.assertEqual(value['code'], 'model_connection_failed')
            self.assertIn('无法连接模型服务', value['detail'])
            self.assertNotIn('PRIVATE SECRET', json.dumps(value))
            with self.assertRaises(StopAsyncIteration): await anext(events)
            save.assert_not_called()

    async def test_graph_tool_keeps_public_proofs_and_excludes_private_proofs(self):
        fixture = ResearchFixture()
        fixture.relations = [dict(id=identity, source='clip', target='llava', owner_model_id=owner,
            source_ids=['source'], evidence=[{'section_id': sid, 'quote': quote}]) for identity, owner, sid, quote in
            [('public-edge', 'clip', 'clip:io', 'CLIP receives image and text.'),
             ('private-edge', 'private', 'secret', 'PRIVATE SECRET')]]
        runner = ResearchRunner(fixture, model_call=fake_model)
        state = {'actions': [{'tool': 'graph_lookup', 'model_id': 'clip', 'query': '', 'unit_ids': ['clip|input']}],
                 'rounds': 0, 'tool_calls': 0, 'seen_actions': [], 'evidence': [], 'trace': [], 'stop_reason': ''}
        result = await runner.tools(state)
        self.assertEqual([e['section_id'] for e in result['evidence']], ['clip:io'])
        self.assertNotIn('PRIVATE SECRET', json.dumps(result))

    async def test_read_tool_selects_query_window_instead_of_a_long_chapter_head(self):
        fixture = ResearchFixture()
        fixture.sections[0]['content'] = 'Background. ' * 600 + '\n\nTarget answer: image text input.'
        runner = ResearchRunner(fixture, model_call=fake_model)
        state = {'actions': [{'tool': 'read_section', 'section_id': 'clip:io', 'query': 'Target answer', 'unit_ids': ['clip|input']}],
                 'rounds': 1, 'tool_calls': 0, 'seen_actions': [], 'evidence': [], 'trace': [], 'stop_reason': ''}
        runner.retain(state, fixture.sections[0], 'CLIP')
        result = await runner.tools(state)
        self.assertIn('Target answer: image text input.', result['evidence'][0]['excerpt'])
        self.assertEqual(result['evidence'][0]['selection']['strategy'], 'matched_window')

    async def test_compact_model_protocol_runs_through_real_graph(self):
        async def compact(purpose, payload):
            result = await fake_model(purpose, payload)
            if purpose == 'assess':
                result['checks'] = [[c['unit_id'], c['status'], [[q['section_id'], q['passage_id']] for q in c['quotes']], c['note']] for c in result['checks']]
            return result
        result = await ResearchRunner(ResearchFixture(), model_call=compact).run('比较')
        self.assertTrue(result['generated'])
        self.assertEqual(result['rounds'], 2)
        self.assertIn('CLIP receives image and text.', result['checks'][0]['quotes'][0]['quote'])

    async def test_supported_provider_disables_extra_thinking_only_for_research(self):
        runner = ResearchRunner(ResearchFixture())
        response = {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
        with patch('backend.rag.research.provider_post', new_callable=AsyncMock, return_value=response) as call, \
                patch.object(research.config, 'LLM_BASE', 'https://api.siliconflow.cn/v1'), \
                patch.object(research.config, 'LLM_MODEL', 'Pro/deepseek-ai/DeepSeek-V3.2'):
            await runner.ask('assess', 'JSON', {}, 100)
            self.assertIs(call.call_args.args[3]['enable_thinking'], False)
        with patch('backend.rag.research.provider_post', new_callable=AsyncMock, return_value=response) as call, \
                patch.object(research.config, 'LLM_BASE', 'https://example.com/v1'):
            await runner.ask('assess', 'JSON', {}, 100)
            self.assertNotIn('enable_thinking', call.call_args.args[3])

    async def test_truncated_model_json_never_becomes_a_success(self):
        runner = ResearchRunner(ResearchFixture())
        response = {'choices': [{'message': {'content': '{"checks":['}, 'finish_reason': 'length'}]}
        with patch('backend.rag.research.provider_post', new_callable=AsyncMock, return_value=response):
            with self.assertRaisesRegex(ValueError, '完整'):
                await runner.ask('assess', 'JSON', {}, 100)
        self.assertEqual(runner.diagnostics[0]['finish_reason'], 'length')
        self.assertEqual(runner.diagnostics[0]['partial_json'], '{"checks":[')

    async def test_graph_retrieves_gap_then_stops_and_preserves_quotes(self):
        fixture = ResearchFixture()
        events = []
        async def emit(name, data): events.append((name, data))
        result = await ResearchRunner(fixture, model_call=fake_model, emit=emit).run('比较CLIP和LLaVA的输入与训练目标')
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(result['tool_calls'], 3)
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['stop_reason'], 'evidence_sufficient')
        self.assertTrue(result['generated'])
        self.assertIn('[3]', result['answer'])
        self.assertTrue(all('secret' not in call[2]['allowed_section_ids'] for call in fixture.search_calls))
        self.assertNotIn('PRIVATE SECRET', json.dumps(result))
        self.assertTrue(events)

    async def test_enough_evidence_has_no_fixed_second_round(self):
        fixture = ResearchFixture()
        real_search = fixture.search
        async def search(query, focus, **kwargs):
            result = await real_search(query, focus, **kwargs)
            if focus == 'clip':
                result['evidence'].append({**fixture.sections[1], 'section_id': 'clip:train'})
            return result
        fixture.search = search
        result = await ResearchRunner(fixture, model_call=fake_model).run('比较')
        self.assertEqual(result['rounds'], 1)
        self.assertEqual(result['model_calls'], 3)

    async def test_unknown_quote_fails_instead_of_claiming_success(self):
        async def invalid(purpose, payload):
            result = await fake_model(purpose, payload)
            if purpose == 'assess': result['checks'][0]['quotes'][0]['quote'] = '原文中不存在的事实'
            return result
        with self.assertRaisesRegex(ValueError, '引文'):
            await ResearchRunner(ResearchFixture(), model_call=invalid).run('比较')

    async def test_repeated_tool_request_stops_with_gap(self):
        async def repeated(purpose, payload):
            if purpose == 'synthesize':
                return {'rows': [{'unit_id': c['unit_id'], 'text': '资料描述如下。', 'citations': []} for c in payload['checks']]}
            result = await fake_model(purpose, payload)
            if purpose == 'assess':
                for c in result['checks']: c.update(status='missing', quotes=[])
                result['decision'] = 'continue'
                result['actions'] = [outline()['queries'][0]]
            return result
        result = await ResearchRunner(ResearchFixture(), model_call=repeated).run('比较')
        self.assertEqual(result['stop_reason'], 'repeated_actions')
        self.assertEqual(result['rounds'], 1)
        self.assertIn('本轮资料未覆盖', result['answer'])

    async def test_hard_round_budget_stops_new_queries(self):
        fixture = ResearchFixture()
        async def search(query, focus, **kwargs):
            result = await ResearchFixture.search(fixture, query, focus, **kwargs)
            # 确定的新证据用于测试预算路由，而非重复同一查询。
            identity = 'extra:' + str(len(fixture.search_calls))
            section = {**fixture.sections[0], 'id': identity, 'content': 'Fresh evidence ' + identity}
            runner.sections[identity] = section
            result['evidence'].append({**section, 'section_id': identity})
            return result
        fixture.search = search
        async def budget(purpose, payload):
            if purpose == 'synthesize':
                return {'rows': [{'unit_id': c['unit_id'], 'text': '未覆盖', 'citations': []} for c in payload['checks']]}
            result = await fake_model(purpose, payload)
            if purpose == 'assess':
                for c in result['checks']: c.update(status='missing', quotes=[])
                result.update(decision='continue', actions=[{'tool': 'search', 'query': 'new terms ' + str(len(fixture.search_calls)), 'model_id': 'clip', 'unit_ids': ['clip|objective']}])
            return result
        runner = ResearchRunner(fixture, model_call=budget)
        result = await runner.run('比较')
        self.assertEqual(result['rounds'], 3)
        self.assertEqual(result['stop_reason'], 'budget_reached')
        self.assertEqual(result['model_calls'], 5)

    async def test_cancellation_propagates(self):
        started = asyncio.Event()
        async def wait_model(purpose, payload):
            started.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(ResearchRunner(ResearchFixture(), model_call=wait_model).run('比较'))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task

    async def test_stream_disconnect_cancels_work_without_saving_success(self):
        cancelled = asyncio.Event()
        async def work(runner, question):
            await runner.event('正在查找资料')
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with patch.object(ResearchRunner, 'run', work), patch.object(research, 'save_report') as save:
            stream = research.research_events(research.ResearchInput(question='比较'), ResearchFixture())
            name, _ = await anext(stream)
            self.assertEqual(name, 'stage')
            await stream.aclose()
            self.assertTrue(cancelled.is_set())
            save.assert_not_called()


class ResearchContractTests(unittest.TestCase):
    def test_real_three_model_plan_merges_queries_without_losing_dimensions(self):
        raw = json.loads((Path(__file__).parent / 'fixtures/research_position_plan.json').read_text(encoding='utf-8'))
        parsed, units = validate_outline(raw, {mid: {'name': mid} for mid in ('transformer', 'vit', 'detr')})
        self.assertEqual(len(parsed['queries']), 3)
        self.assertEqual(len(units), 6)
        self.assertEqual({u['id'] for u in units}, {uid for a in parsed['queries'] for uid in a['unit_ids']})
        self.assertEqual(parsed['dimensions'][1]['id'], 'position_encoding_implementation')
        for action in parsed['queries']:
            self.assertEqual(len(action['unit_ids']), 2)
            for original in raw['queries']:
                if original['model_id'] == action['model_id']:
                    self.assertIn(original['query'], action['query'])

    def test_query_merge_does_not_hide_invalid_tool_or_exceed_scope_budget(self):
        raw = json.loads((Path(__file__).parent / 'fixtures/research_position_plan.json').read_text(encoding='utf-8'))
        catalog = {mid: {'name': mid} for mid in ('transformer', 'vit', 'detr')}
        bad = copy.deepcopy(raw); bad['queries'][0]['tool'] = 'read_section'
        with self.assertRaisesRegex(ValueError, '首轮'):
            validate_outline(bad, catalog)
        bad = copy.deepcopy(raw)
        bad['queries'].append({**bad['queries'][0], 'model_id': None})
        with self.assertRaisesRegex(ValueError, '预算'):
            validate_outline(bad, catalog)

    def test_connection_errors_and_timeouts_have_distinct_messages(self):
        for failure in (httpx.ConnectError('private'), httpx.ProxyError('private'), httpx.ConnectTimeout('private')):
            status, detail, code = research.research_error(failure)
            self.assertEqual((status, code), (503, 'model_connection_failed'))
            self.assertNotIn('private', detail)
        for failure in (TimeoutError(), httpx.ReadTimeout('private')):
            status, _, code = research.research_error(failure)
            self.assertEqual((status, code), (504, 'research_timeout'))

    def test_post_connection_error_returns_503_instead_of_internal_500(self):
        from backend import main
        app = FastAPI(); app.include_router(research.router)
        with patch.object(main, 'retriever', ResearchFixture()), patch.object(research.config, 'LLM_KEY', 'test'), \
                patch.object(ResearchRunner, 'run', new_callable=AsyncMock, side_effect=httpx.ConnectError('PRIVATE SECRET')), \
                patch.object(research, 'save_report') as save, TestClient(app) as client:
            response = client.post('/api/research', json={'question': '比较'})
            self.assertEqual(response.status_code, 503)
            self.assertIn('无法连接模型服务', response.json()['detail'])
            self.assertNotIn('PRIVATE SECRET', response.text)
            save.assert_not_called()

    def test_passage_reference_extracts_original_and_rejects_unknown_number(self):
        state = {'units': [{'id': 'clip|input'}], 'outline': {'model_ids': ['clip']},
                 'evidence': [{'section_id': 'clip:io', 'excerpt': 'CLIP receives image and text.'}]}
        raw = {'checks': [{'unit_id': 'clip|input', 'status': 'supported', 'quotes': [{'section_id': 'clip:io', 'passage_id': 1}], 'note': '输入依据'}],
               'decision': 'finish', 'actions': []}
        checked = validate_assessment(raw, state)
        self.assertEqual(checked['checks'][0]['quotes'][0]['quote'], state['evidence'][0]['excerpt'])
        raw['checks'][0]['quotes'][0]['passage_id'] = 99
        with self.assertRaisesRegex(ValueError, '未知原文编号'):
            validate_assessment(raw, state)

    def test_private_document_blocked_before_planning(self):
        with self.assertRaisesRegex(ValueError, '本地'):
            ResearchRunner(ResearchFixture(), ['private-doc'])

    def test_private_document_overrides_public_section_flag(self):
        fixture = ResearchFixture()
        fixture.sections[-1]['cloud_allowed'] = True
        self.assertNotIn('secret', allowed_scope(fixture))

    def test_outline_matrix_is_program_generated_and_must_be_fully_queried(self):
        raw = outline()
        parsed, units = validate_outline(raw, ResearchFixture().models)
        self.assertEqual(len(units), 4)
        raw['queries'][0]['unit_ids'].pop()
        with self.assertRaisesRegex(ValueError, '覆盖'): validate_outline(raw, ResearchFixture().models)

    def test_unknown_model_and_duplicate_dimensions_rejected(self):
        raw = outline(); raw['model_ids'][0] = 'invented'
        with self.assertRaises(ValueError): validate_outline(raw, ResearchFixture().models)
        raw = outline(); raw['dimensions'][1]['id'] = 'input'
        with self.assertRaises(ValueError): validate_outline(raw, ResearchFixture().models)

    def test_invalid_report_reference_is_not_silently_kept(self):
        state = {'units': [{'id': 'clip|input'}], 'checks': [{'unit_id': 'clip|input', 'status': 'supported', 'quotes': [{'section_id': 'clip:io'}]}],
                 'evidence': [{'section_id': 'clip:io'}], 'outline': {'model_ids': ['clip'], 'dimensions': [{'id': 'input', 'label': '输入'}]}}
        with self.assertRaisesRegex(ValueError, '引用'):
            render_report([{'unit_id': 'clip|input', 'text': '解释', 'citations': [99]}], state, {'clip': {'name': 'CLIP'}})

    def test_api_rejects_private_scope_without_calling_model(self):
        from backend import main
        app = FastAPI(); app.include_router(research.router)
        with patch.object(main, 'retriever', ResearchFixture()), patch.object(research.config, 'LLM_KEY', 'test'), \
                patch.object(ResearchRunner, 'run', new_callable=AsyncMock) as run, TestClient(app) as client:
            result = client.post('/api/research', json={'question': '解释', 'document_ids': ['private-doc']})
            self.assertEqual(result.status_code, 422)
            run.assert_not_called()

    def test_api_saves_only_a_completed_report(self):
        from backend import main
        app = FastAPI(); app.include_router(research.router)
        completed = {'id': 'research-test', 'type': 'agentic_research_run', 'generated': True, 'answer': '完成'}
        with patch.object(main, 'retriever', ResearchFixture()), patch.object(research.config, 'LLM_KEY', 'test'), \
                patch.object(ResearchRunner, 'run', new_callable=AsyncMock, return_value=completed), \
                patch.object(research, 'save_report') as save, TestClient(app) as client:
            response = client.post('/api/research', json={'question': '比较'})
            self.assertEqual(response.json(), completed)
            save.assert_called_once_with(completed)

    def test_api_model_failure_does_not_save_or_leak_details(self):
        from backend import main
        app = FastAPI(); app.include_router(research.router)
        with patch.object(main, 'retriever', ResearchFixture()), patch.object(research.config, 'LLM_KEY', 'test'), \
                patch.object(ResearchRunner, 'run', new_callable=AsyncMock, side_effect=ValueError('PRIVATE SECRET')), \
                patch.object(research, 'save_report') as save, TestClient(app) as client:
            response = client.post('/api/research', json={'question': '比较'})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn('PRIVATE SECRET', response.text)
            save.assert_not_called()


if __name__ == '__main__': unittest.main()
