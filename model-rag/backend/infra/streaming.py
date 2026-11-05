"""模型 SSE 正文与浏览器事件流；断连关闭请求，不转发推理字段或服务凭据。"""
import asyncio
import json
from contextlib import aclosing, suppress
from time import monotonic

import httpx
from backend.infra.access import public_payload


async def stream_completion(base, key, body):
    if not key:
        raise RuntimeError("服务密钥尚未配置")
    async with httpx.AsyncClient(timeout=65) as client:
        for attempt in range(3):
            async with client.stream("POST", f"{base.rstrip('/')}/chat/completions",
                                     headers={"Authorization": f"Bearer {key}"},
                                     json={**body, "stream": True}) as response:
                if response.status_code in (429, 502, 503) and attempt < 2:
                    retry = True
                else:
                    retry = False
                    if response.status_code >= 400:
                        raise RuntimeError(f"模型服务返回 HTTP {response.status_code}")
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise RuntimeError("模型服务未返回事件流")
                    data_lines, total, finished = [], 0, False
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                            if sum(map(len, data_lines)) > 262144:
                                raise RuntimeError("模型事件过长")
                        elif not line and data_lines:
                            payload = "\n".join(data_lines)
                            data_lines.clear()
                            if payload == "[DONE]":
                                return
                            try:
                                item = json.loads(payload)
                                if not isinstance(item, dict) or item.get("error"):
                                    raise ValueError("invalid event")
                                choices = item.get("choices", [])
                                if not choices:
                                    continue  # 部分服务另发 usage 事件。
                                choice = choices[0]
                                text = choice.get("delta", {}).get("content") or ""
                                reason = choice.get("finish_reason")
                                if not isinstance(text, str):
                                    raise ValueError("invalid content")
                            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                                raise RuntimeError("模型事件格式无效") from exc
                            total += len(text)
                            if total > 64000:
                                raise RuntimeError("模型正文超过长度限制")
                            if reason:
                                finished = True
                            if text or reason:
                                yield {"text": text, "finish_reason": reason}
                    if data_lines or not finished:
                        raise RuntimeError("模型事件流提前结束")
                    return
            if retry:
                await asyncio.sleep(1 + attempt)


async def relay_events(events, is_disconnected):
    """有界队列提供背压；等待期间检查断连，心跳仅保持连接。"""
    queue = asyncio.Queue(maxsize=16)

    async def produce():
        try:
            async with asyncio.timeout(180), aclosing(events):
                async for event in events:
                    await queue.put(event)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            await queue.put(("error", {"detail": "本轮处理超时，请缩短问题后重试。"}))
        except Exception:
            # 服务错误正文可能包含配置信息，不直接送到浏览器。
            await queue.put(("error", {"detail": "本轮处理未完成，请检查后端服务后重试。"}))

    task = asyncio.create_task(produce())
    last_output = monotonic()
    try:
        while True:
            if await is_disconnected():
                return
            try:
                name, data = await asyncio.wait_for(queue.get(), timeout=1)
            except TimeoutError:
                if task.done() and queue.empty():
                    return
                if monotonic() - last_output >= 12:
                    yield ": heartbeat\n\n"
                    last_output = monotonic()
                continue
            yield f"event: {name}\ndata: {json.dumps(public_payload(data), ensure_ascii=False)}\n\n"
            last_output = monotonic()
            if name in ("done", "error"):
                return
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
