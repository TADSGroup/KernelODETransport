from kode.data import load_dataset


two_dim_dataset = load_dataset.two_dimensional_data('pinwheel', None, 200)
power_dataset = load_dataset.high_dimensional_data('power').trn.x
gas_dataset = load_dataset.high_dimensional_data('gas').trn.x
hepmass_dataset = load_dataset.high_dimensional_data('hepmass').trn.x
miniboone_dataset = load_dataset.high_dimensional_data('miniboone').trn.x
bsds_dataset = load_dataset.high_dimensional_data('bsds300').trn.x
mnist_dataset = load_dataset.high_dimensional_data('mnist').trn.x