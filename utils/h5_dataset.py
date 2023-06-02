import h5py
import numpy as np
from torch.utils.data import Dataset

default_opener = lambda p_: h5py.File(p_, 'r')

class HDF5Dataset(Dataset):
    def __init__(self, file_path,
                 transform = None):

        self.file_path = file_path
        self.lengths = []

        with h5py.File(self.file_path, 'r') as hf:
            self.image_name = [i for i in list(hf.keys()) if 'Image' in i][0]
            self.label_name = [i for i in list(hf.keys()) if 'Labels' in i][0]
            self.image = np.array(hf[self.image_name])
            self.label = np.array(hf[self.label_name])
        self.transform = transform

    def __len__(self):
        assert len(self.image) == len(self.label)
        return len(self.label)

    def __getitem__(self, index):
        x = self.image[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)

        return x, y