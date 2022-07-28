import threading
import warnings

import torch
import torch.distributed as dist

from concurrent.futures import ThreadPoolExecutor, Future
from torchdata.datapipes.iter import IterDataPipe
from typing import Callable, Optional

_DEFAULT_TIMEOUT = 30 * 60

__all__ = ["FullSyncIterDataPipe", "OnlineReceiverIterDataPipe"]


class PrefetchExecutor:
    def __init__(
        self,
        datapipe_iterator,
        callback_fn: Optional[Callable[[Future], None]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.datapipe_iterator = datapipe_iterator
        self.callback_fn = callback_fn
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._get_data_future: Future = self._executor.submit(self.fetch_next)
        if self.callback_fn is not None:
            self._get_data_future.add_done_callback(self.callback_fn)

    def fetch_next(self):
        return next(self.datapipe_iterator)

    def return_next(self):
        data = self._get_data_future.result(timeout=self.timeout)
        self._get_data_future = self._executor.submit(self.fetch_next)
        if self.callback_fn is not None:
            self._get_data_future.add_done_callback(self.callback_fn)
        return data

    def shutdown(self):
        self._executor.shutdown(wait=True)


class FullSyncIterDataPipe(IterDataPipe):
    def __init__(self, datapipe, timeout=_DEFAULT_TIMEOUT):
        self.datapipe = datapipe
        self.timeout = timeout

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Torch Distributed is required to be initialized"
            )
        self._process_group = dist.new_group(backend="gloo")
        self._end = torch.tensor([0], dtype=torch.int32)

        self._lock = threading.RLock()
        self._cv = threading.Condition(lock=self._lock)
        self._executor = None
        self._error = None

    def _callback_fn(self, f: Future) -> None:
        with self._cv:
            if f.exception() and not isinstance(f.exception(), StopIteration):
                self._error = f.exception()
            else:
                if isinstance(f.exception(), StopIteration):
                    self._end = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(
                tensor=self._end,
                op=dist.ReduceOp.SUM,
                group=self._process_group,
            )
            self._cv.notify()

    def __iter__(self):
        assert self._executor is None
        self._executor = PrefetchExecutor(
            iter(self.datapipe),
            self._callback_fn,
            self.timeout
        )
        while True:
            with self._cv:
                try:
                    self._cv.wait(self.timeout)
                except RuntimeError:
                    warnings.warn("Timeout")
                    break
                if self._error is not None:
                    raise self._error
                if bool(self._end > 0):
                    break
                data = self._executor.return_next()
                yield data

    def reset(self):
        self._executor.shutdown()
        self._executor = None
        self._end = torch.tensor([0], dtype=torch.int32)
        self._error = None
