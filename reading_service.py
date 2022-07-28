from torch.utils.data.graph import DataPipe
from torchdata.dataloader2 import ReadingServiceInterface

from datapipe import FullSyncIterDataPipe


class DistributedReadingService(ReadingServiceInterface):
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        return FullSyncIterDataPipe(datapipe)
