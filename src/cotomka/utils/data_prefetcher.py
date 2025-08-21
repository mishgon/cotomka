from typing import List, Optional, Literal, Any
import threading
import multiprocessing
import random
import time
import itertools

from cotomka.datasets.base import Dataset


class DataPrefetcher:
    def __init__(
            self,
            dataset: Dataset,
            num_workers: int,
            buffer_size: int = 2,
            clone_factor: int = 1,
            backend: Literal['threading', 'multiprocessing'] = 'threading',
            fields: Optional[List[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.fields = fields
        self.buffer_size = buffer_size
        self.clone_factor = clone_factor

        match backend:
            case 'threading':
                worker_cls = threading.Thread
                lock_cls = threading.Lock
                self.stop_event = threading.Event()
            case 'multiprocessing':
                worker_cls = multiprocessing.Process
                lock_cls = multiprocessing.Lock
                self.stop_event = multiprocessing.Event()
            case _:
                raise ValueError(backend)

        self.buffers = [[] for _ in range(num_workers)]
        self.locks = [lock_cls() for _ in range(num_workers)]
        self.workers = []
        for i in range(num_workers):
            worker = worker_cls(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

    def _worker(self, i: int) -> None:
        while not self.stop_event.is_set():
            try:
                id_ = random.choice(self.dataset.ids[:-1])
                data = self.dataset.get(id_, fields=self.fields)
            except Exception as e:
                # print(e)
                # print(f"failed id: {id_}")
                # continue
                raise e
            with self.locks[i]:
                if len(self.buffers[i]) < self.buffer_size:
                    self.buffers[i].append((data, 0))
                else:
                    time.sleep(0.01)

    def sample(self, k: int) -> Any:
        indices, samples = [], []
        for i in range(len(self.workers)):
            with self.locks[i]:
                for j in range(len(self.buffers[i])):
                    indices.append((i, j))

        if len(indices) > k:
            indices = random.sample(indices, k=k)

        indices = sorted(indices, key=lambda ij: ij[0])
        for i, g in itertools.groupby(indices, key=lambda ij: ij[0]):
            with self.locks[i]:
                for _, j in g:
                    clone, clones_count = self.buffers[i][j]
                    clones_count += 1
                    if clones_count >= self.clone_factor:
                        del self.buffers[i][j]
                    else:
                        self.buffers[i][j] = (clone, clones_count)
                    
                    samples.append(clone)

        if len(samples) < k:
            samples.extend([
                self.dataset.get(random.choice(self.dataset.ids[:-1]), fields=self.fields)
                for _ in range(k - len(samples))
            ])

        return samples

    def __iter__(self):
        return self

    def __next__(self, schedule=None) -> Any:
        if schedule is None:
            schedule = random.sample(range(len(self.workers)), k=len(self.workers))
        for i in schedule:
            with self.locks[i]:
                if not self.buffers[i]:
                    continue

                j = random.randrange(len(self.buffers[i]))
                clone, clones_count = self.buffers[i][j]
                clones_count += 1
                if clones_count >= self.clone_factor:
                    del self.buffers[i][j]
                else:
                    self.buffers[i][j] = (clone, clones_count)
                return clone

        return self.dataset.get(random.choice(self.dataset.ids[:-1]), fields=self.fields)

    def destroy(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        self.buffers.clear()

    def __del__(self) -> None:
        self.destroy()
