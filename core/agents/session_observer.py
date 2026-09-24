"""Live session tail for task observers; streamed into the runner's memory.

This copy never replaces or emits the host's native transcript. JSONL readers
wait for complete lines. Legacy Gemini JSON snapshots are forwarded only when
the document changes; the task observer handles message updates by ID.
"""
import json
import threading


class SessionObserver:
    def __init__(self, find_session, observer):
        self.find_session, self.observer = find_session, observer
        self.offsets = {}
        self.snapshots = {}
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def drain(self):
        path = self.find_session()
        if not path:
            return
        try:
            with open(path, "rb") as f:
                if path.endswith(".json"):
                    data = f.read()
                    try:
                        record = json.loads(data)
                    except (ValueError, UnicodeDecodeError):
                        return  # A snapshot is being rewritten.
                    if data != self.snapshots.get(path):
                        self.observer.observe(record)
                        self.snapshots[path] = data
                    return
                f.seek(self.offsets.get(path, 0))
                while True:
                    line = f.readline()
                    if not line.endswith(b"\n"):
                        return
                    if line.strip():
                        self.observer.observe(json.loads(line))
                    self.offsets[path] = f.tell()
        except FileNotFoundError:
            return  # The CLI has not created the file yet.

    def _loop(self):
        try:
            while not self.stop.is_set():
                self.drain()
                self.stop.wait(0.05)
        except Exception as e:
            self.error = f"live message observer failed: {type(e).__name__}: {e}"

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.error is None:
            try:
                self.drain()
            except Exception as e:
                self.error = f"live message observer failed: {type(e).__name__}: {e}"


def observe_session(call, find_session, observer):
    tail = SessionObserver(find_session, observer)
    tail.thread.start()
    try:
        rc, error = call()
    finally:
        tail.close()
    return rc, error or tail.error
