import time
import torch
import torch.distributed as dist

from datetime import timedelta
from torch.utils.data.graph import DataPipe
from torchdata.dataloader2 import ReadingServiceInterface

from datapipe import FullSyncIterDataPipe


SHARED_SEED = "_dl_shared_seed"
SHARED_SEED_COUNTER = "_dl_shared_seed_recv_cnt"
SHARED_SEED_CHECK_INTERVAL = 0.01
SHARED_SEED_DEFAULT_TIMEOUT = 30 * 60


class DistributedReadingService(ReadingServiceInterface):
    def __init__(self):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Torch Distributed is required to be initialized"
             )
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

    def initialize_iteration(self) -> None:
        seed = self._share_seed()
        self._seed_generator = torch.Generator()
        self._seed_generator.manual_seed(seed)
        self._datapipe = torch.utils.data.graph_settings.apply_shuffle_seed(
            self._datapipe,
            self._seed_generator,
        )

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        torch.utils.data.graph_settings.apply_sharding(
            datapipe,
            self._world_size,
            self._rank,
        )
        self._datapipe = datapipe
        return FullSyncIterDataPipe(datapipe)

    def _share_seed(self):
        _sd = torch.empty((), dtype=torch.int64).random_().item()
        store = dist.distributed_c10d._get_default_store()
        if self._rank == 0:
            _sd_str = str(_sd)
            store.set(SHARED_SEED, _sd_str)
            _sd_recv_cnt = store.add(SHARED_SEED_COUNTER, 1)
            start = time.time()
            while _sd_recv_cnt < self._world_size:
                time.sleep(SHARED_SEED_CHECK_INTERVAL)
                _sd_recv_cnt = store.add(SHARED_SEED_COUNTER, 0)
                if timedelta(seconds=(time.time() - start)) > \
                        timedelta(seconds=SHARED_SEED_DEFAULT_TIMEOUT):
                    raise RuntimeError(
                        "Timed out receiving the signal from the "
                        "distribtued store on Rank 0 that all other "
                        "Ranks have received the shared seed. "
                        f"(world_size={self._world_size}, "
                        f"received={_sd_recv_cnt}, "
                        f"timeout={SHARED_SEED_DEFAULT_TIMEOUT})"
                    )
            store.set(SHARED_SEED, "")
            _sd_recv_cnt = store.add(SHARED_SEED_COUNTER, -self._world_size)
            assert _sd_recv_cnt == 0
        else:
            _sd_str = ""
            start = time.time()
            while len(_sd_str) == 0:
                time.sleep(SHARED_SEED_CHECK_INTERVAL)
                _sd_str = store.get(SHARED_SEED)
                if timedelta(seconds=(time.time() - start)) > \
                        timedelta(seconds=SHARED_SEED_DEFAULT_TIMEOUT):
                    raise RuntimeError(
                        "Timed out receiving the shared seed from the "
                        f"distribtued store on Rank {self._rank}. "
                        f"(world_size={self._world_size}, "
                        f"timeout={SHARED_SEED_DEFAULT_TIMEOUT})"
                    )
            _sd_recv_cnt = store.add(SHARED_SEED_COUNTER, 1)
            while _sd_recv_cnt > 0:
                time.sleep(SHARED_SEED_CHECK_INTERVAL)
                _sd_recv_cnt = store.add(SHARED_SEED_COUNTER, 0)
            _sd = int(_sd_str)

        return _sd
