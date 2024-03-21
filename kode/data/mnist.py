# mnist.py

import torchvision as tv
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import kode.data as datasets

class MNIST:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        file = datasets.root + 'mnist/'
        trn, trn_labels, val, val_labels, tst, tst_labels = \
            load_data_and_clean(file)

        self.trn = self.Data(trn)
        self.trn_labels = self.Data(trn_labels)
        self.val = self.Data(val)
        self.val_labels = self.Data(val_labels)
        self.tst = self.Data(tst)
        self.tst_labels = self.Data(tst_labels)

        self.n_dims = self.trn.x.shape[1]


def load_data(file):

    # load data using torch
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize((0.1307,),
                                                         (0.3081,))])
    train = tv.datasets.MNIST(file, train=True, transform=transform,
                           download=True)
    test = tv.datasets.MNIST(file, train=False, transform=transform,
                          download=True)

    # data loaders
    train_loader = DataLoader(train, batch_size=len(train))
    train_array = next(iter(train_loader))[0].numpy().reshape(-1, 784)
    train_labels = next(iter(train_loader))[1].numpy()

    test_loader = DataLoader(test, batch_size=len(test))
    test_array = next(iter(test_loader))[0].numpy().reshape(-1, 784)
    test_labels = next(iter(test_loader))[1].numpy()

    return train_array, train_labels, test_array, test_labels


def load_data_and_clean(file):

    train, train_labels, test, test_labels = load_data(file)

    # validation
    train, val, train_labels, val_labels = train_test_split(train,
                                                           train_labels,
                                                           test_size=0.2,
                                                           random_state=42)

    # scaler = StandardScaler()
    #
    # # normalize
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)
    # val = scaler.transform(val)

    return train, train_labels, val, val_labels, test, test_labels







