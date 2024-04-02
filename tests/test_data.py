import unittest
from kode.data import load_dataset, utils


class test_load_dataset(unittest.TestCase):
    def test_2d_dataset(self):
        data = load_dataset.two_dimensional_data('pinwheel', rng=None,
                                                 batch_size=200)
        true_shape = (200, 2)
        self.assertEqual(data.shape, true_shape, '2D dataset is not correctly '
                                               'loaded.')  # add assertion here

    def test_high_dimensional_dataset(self):
        datasets = ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']
        true_dims = [6, 8, 21, 43, 63]
        for data_name, true_dim in zip(datasets, true_dims):
            data = load_dataset.high_dimensional_data(data_name).trn.x
            self.assertEqual(data.shape[1], true_dim, f'{data_name} dataset '
                                                     f'is  not correctly loaded.')


    def test_utils_dataloader(self):
        data = load_dataset.high_dimensional_data('power')
        train = data.trn.x[:10020]
        batch_size = 1000
        expected_iters = int(len(train) / batch_size) # 10
        counter = 0
        for X_batch in utils.DataLoader([train], batch_size=batch_size, \
                seed=20):
            counter += 1
        self.assertEqual(counter, expected_iters, f'utils.DataLoader is '
                                                       f'not working. '
                                                       f'Expected iterations and actual iterations do not match.')


if __name__ == '__main__':
    unittest.main()
