# Optional observability wrapper.
# If Langfuse is not enabled/configured, we no-op to allow system to run offline smoothly.
from typing import Optional
from functools import wraps

class TraceLogger:
    def __init__(self):
        self.enabled = False
        try:
            # Conditional import
            from langfuse import Langfuse
            # Assuming env vars LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST are set
            import os
            if os.environ.get("LANGFUSE_PUBLIC_KEY"):
                self.langfuse = Langfuse()
                self.enabled = True
        except ImportError:
            pass

    def trace(self, session_id: str, name: str, **kwargs):
        if not self.enabled:
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
                def span(self, *args, **kw): return DummyContext()
                def end(self, *args, **kw): pass
            return DummyContext()
            
        return self.langfuse.trace(
            name=name,
            session_id=session_id,
            metadata=kwargs
        )

tracer = TraceLogger()
