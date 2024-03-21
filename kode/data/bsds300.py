import numpy as np
import h5py

import kode.data as datasets

class BSDS300:
    """
    A dataset of patches from BSDS300.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):

            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self, normalized=True):

        # load dataset
        f = h5py.File(datasets.root + 'BSDS300/BSDS300.hdf5', 'r')

        self.trn = self.Data(f['train'])
        self.val = self.Data(f['validation'])
        self.tst = self.Data(f['test'])

        if normalized is True:
            data = np.vstack((self.trn.x, self.val.x))
            mu = data.mean(axis=0)
            s = data.std(axis=0)
            self.trn.x = (self.trn.x - mu) / s
            self.val.x = (self.val.x - mu) / s
            self.tst.x = (self.tst.x - mu) / s

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2

        f.close()


# data = np.vstack((data_train, data_validate))
# mu = data.mean(axis=0)
# s = data.std(axis=0)
# data_train = (data_train - mu) / s
# data_validate = (data_validate - mu) / s
# data_test = (data_test - mu) / s




