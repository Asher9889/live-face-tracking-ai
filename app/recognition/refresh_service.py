import threading

from app.config.config import envConfig
from app.recognition import embedding_store, unknown_embedding_store


class EmbeddingRefreshService:
    def __init__(self, interval_sec: float):
        self.interval_sec = max(float(interval_sec), 1.0)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="embedding-refresh-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.wait(self.interval_sec):
            known_error = None
            unknown_error = None

            print("[AI] Periodic embedding reload started")

            try:
                embedding_store.load_embeddings()
            except Exception as exc:
                known_error = exc

            try:
                unknown_embedding_store.load_unknown_embeddings()
            except Exception as exc:
                unknown_error = exc

            if known_error or unknown_error:
                print(
                    "[AI] Periodic embedding reload completed with errors: "
                    f"known={known_error}, unknown={unknown_error}"
                )
            else:
                print("[AI] Periodic embedding reload completed")


embedding_refresh_service = EmbeddingRefreshService(
    interval_sec=getattr(envConfig, "EMBEDDINGS_REFRESH_INTERVAL_SEC", 300),
)
