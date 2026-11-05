"""按请求身份缓存检索视图；共享的是预置知识，不共享用户私有片段。"""
from collections import OrderedDict
import threading
from backend.infra.access import current_actor
from backend.rag.retrieval import Retriever


class RetrieverRegistry:
    def __init__(self):
        self._system = Retriever()
        self._views = OrderedDict()
        self._lock = threading.RLock()

    def current(self):
        actor = current_actor.get()
        with self._lock:
            if actor is None:
                if self._system is None:
                    self._system = Retriever()
                return self._system
            key = (actor.id,actor.role,actor.maintainer)
            if key not in self._views:
                self._views[key] = Retriever()
                if len(self._views) > 16:
                    self._views.popitem(last=False)
            self._views.move_to_end(key)
            return self._views[key]

    def install(self,value):
        actor = current_actor.get()
        with self._lock:
            self._views.clear()
            self._system = value if actor is None else None
            if actor is not None:
                self._views[(actor.id,actor.role,actor.maintainer)] = value

    def __getattr__(self,name):
        return getattr(self.current(),name)
