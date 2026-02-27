# app/metrics/runtime_metrics.py

import time
from collections import defaultdict

class RuntimeMetrics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(float)
        self.last_log = time.time()

    def inc(self, key: str, value: int = 1):
        self.counters[key] += value

    def add_time(self, key: str, duration: float):
        self.timers[key] += duration

    def log(self, interval: float = 5.0):
        now = time.time()
        if now - self.last_log < interval:
            return

        print("\n===== RUNTIME METRICS =====")
        for k, v in self.counters.items():
            print(f"{k}: {v / interval:.2f}/sec")

        for k, v in self.timers.items():
            print(f"{k}_avg: {(v / max(self.counters.get(k+'_count',1),1)):.4f}s")

        print("===========================\n")

        self.counters.clear()
        self.timers.clear()
        self.last_log = now


metrics = RuntimeMetrics()