import h5py
from collections import namedtuple

from typing import Dict


DataSetEntry = namedtuple('DataSetEntry', ('chunk_length', 'shape', 'dtype'))


class HDF5Store(object):
    """
    Class to append value to a hdf5 file.
    """

    def __init__(self, datapath, datasets: Dict[str, DataSetEntry], compression="gzip", mode: str = "w-"):
        self.datapath = datapath
        self.dataset = datasets
        self.i = 0
        self.mode = mode

        with h5py.File(self.datapath, mode=self.mode) as h5f:
            for name, x in self.dataset.items():
                h5f.create_dataset(
                    name,
                    shape=(0,) + x.shape,
                    maxshape=(None,) + x.shape,
                    dtype=x.dtype,
                    compression=compression,
                    chunks=(x.chunk_length,) + x.shape)

    def append(self, x: Dict):
        with h5py.File(self.datapath, mode='a') as h5f:
            for key, values in x.items():
                dset = h5f[key]
                dset.resize((self.i + 1,) + self.dataset[key].shape)
                dset[self.i] = [values]

            self.i += 1
            h5f.flush()
