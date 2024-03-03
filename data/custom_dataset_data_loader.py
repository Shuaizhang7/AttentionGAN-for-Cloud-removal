import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'alignedrandom':
        from data.aligned_random_dataset import AlignedRandomDataset
        dataset = AlignedRandomDataset()
    elif opt.dataset_mode == 'Coco':
        from data.coco_dataset import UnalignedCocoDataset
        dataset = UnalignedCocoDataset()
    elif opt.dataset_mode == 'CocoSeg':
        from data.cocoseg_dataset import CocoSegDataset
        dataset = CocoSegDataset()
    elif opt.dataset_mode == 'unaligned_sar':
        from data.unaligned_dataset_sar import UnalignedDataset_sar
        dataset = UnalignedDataset_sar()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers= 0) #int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
