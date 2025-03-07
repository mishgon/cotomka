from typing import List, Optional, Literal
import threading
import multiprocessing
import random
import time
from typing import Any

from cotomka.datasets.base import Dataset


class CloneJutsu:
    def __init__(
            self, 
            dataset: Dataset,
            max_originals: int,
            max_clones: int,
            num_workers: int,
            backend: Literal['threading', 'multiprocessing'] = 'threading',
            fields: Optional[List[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.fields = fields
        self.max_originals = max_originals
        self.max_clones = max_clones
        self.originals = []

        match backend:
            case 'threading':
                self.lock = threading.Lock()
                self.stop_event = threading.Event()
                worker_cls = threading.Thread
            case 'multiprocessing':
                self.lock = multiprocessing.Lock()
                self.stop_event = multiprocessing.Event()
                worker_cls = multiprocessing.Process
            case _:
                raise ValueError(backend)

        self.workers = []
        for _ in range(num_workers):
            worker = worker_cls(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            original = self.dataset.get(random.choice(self.dataset.ids), fields=self.fields)

            with self.lock:
                if len(self.originals) < self.max_originals:
                    self.originals.append((original, 0))
                else:
                    time.sleep(0.01)

    def get(self) -> Any:
        if not self.originals:
            return self.dataset.get(random.choice(self.dataset.ids), fields=self.fields)

        with self.lock:
            idx = random.randrange(len(self.originals))
            clone, clones_count = self.originals[idx]
            clones_count += 1
            if clones_count >= self.max_clones:
                del self.originals[idx]
            else:
                self.originals[idx] = (clone, clones_count)
            return clone

    def __del__(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
