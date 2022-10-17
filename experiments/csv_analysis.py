# import pandas
# import torch
# import scipy.signal as signal
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge
# from create_local_data import make_test_data
# from configs.default_config import get_config_from_file
# from setup import set_device, set_seeds
# import h5py
# from nlb_tools.make_tensors import h5_to_dict
# import sys
# import shutil
# import numpy as np
# import os

import pandas
import torch
# import scipy.signal as signal
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge
# from create_local_data import make_test_data
# from configs.default_config import get_config_from_file
# from setup import set_device, set_seeds
# import h5py
# from nlb_tools.make_tensors import h5_to_dict
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os

df = pandas.read_csv('/home/dmifsud/Projects/NDT-U/05.10.2022_16.41.csv')
smth_std = 60
lag = 40

fig, ax = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)

plt.xlabel("Smoothing Std Dev (ms)")
plt.ylabel("R²")
plt.title("Smoothing vs Velocity Decoding R²")

ax.scatter(df['std'], df['r2'], marker='^', c='tab:blue', label='Smooth')
# ax.scatter(df['Original Decoding'], df['Co-bps'], marker='^', c='tab:blue', label='Original')
# ax.set_ylim([0.12, 0.16])
# ax.set_xlim([0.4, 0.7])
# ax.scatter(df['Smooth Decoding'], df['Co-bps'], marker='^', c='tab:red', label='Smoothed')
# plt.legend()
plt.tight_layout()
plt.savefig('smth_std_vs_r2.png', facecolor='white', transparent=False)

# df = pandas.read_csv('/home/dmifsud/Projects/NDT-U/cobps_vs_rates.csv')
# smth_std = 80
# lag = 120

# make_test_data(window=30, overlap=24, lag=lag, smooth_std=smth_std)
# with h5py.File('/home/dmifsud/Projects/NDT-U/data/mc_rtt_cont_24_test.h5', 'r') as h5file:
#     h5dict = h5_to_dict(h5file)

# path_base = '/home/dmifsud/Projects/NDT-U/runs/train/'

# new_vals = []
# smooth_vals = []

# for i in df.Name:
#     path = path_base + i + '/best_lt_co_bps.pt'
#     name = path[:path.rindex('/')].split('/')[-1]
#     config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')
#     if not os.path.isdir(f"plots/{name}"): os.makedirs(f"plots/{name}")
#     shutil.copyfile(path[:path.rindex('/')+1]+'config.yaml', f"plots/{name}/config.yaml")
#     set_device(config)
#     device = torch.device('cuda:0')

#     set_seeds(config)

#     model = torch.load(path).to(device)
#     model.name = name
#     model.eval()

#     smth_spikes = torch.Tensor(h5dict['test_hi_smth_spikes'])
#     heldout_smth_spikes = torch.Tensor(h5dict['test_ho_smth_spikes'])
#     test_smth_spikes = torch.cat([smth_spikes, heldout_smth_spikes], -1)

#     smth_spikes = torch.Tensor(h5dict['train_hi_smth_spikes'])
#     heldout_smth_spikes = torch.Tensor(h5dict['train_ho_smth_spikes'])
#     train_smth_spikes = torch.cat([smth_spikes, heldout_smth_spikes], -1)

#     gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv.fit(train_smth_spikes.numpy(), h5dict['train_vel_segments'])
#     print(f'\n {gscv.score(test_smth_spikes, h5dict["test_vel_segments"]):.3f} R\u00b2 \n')
#     new_vals.append(gscv1.score(test_rates, h5dict['test_vel_segments']))

#     with torch.no_grad():
#         train_rates = []
#         test_ho_spikes = []
#         for spikes, heldout_spikes in zip(
#             torch.Tensor(h5dict['train_spikes_heldin']).to(device), torch.Tensor(h5dict['train_spikes_heldout']).to(device)
#         ):
#             ho_spikes = torch.zeros_like(heldout_spikes).to(device)
#             spikes_new = torch.cat([spikes, ho_spikes], -1).to(device)
#             output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
#             train_rates.append(output.cpu())
#             test_ho_spikes.append(heldout_spikes.unsqueeze(dim=0)[:, -1, :].cpu())

#     train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates
    
#     with torch.no_grad():
#         test_rates = []
#         test_ho_spikes = []
#         for spikes, heldout_spikes in zip(
#             torch.Tensor(h5dict['test_spikes_heldin']).to(device), torch.Tensor(h5dict['test_spikes_heldout']).to(device)
#         ):
#             ho_spikes = torch.zeros_like(heldout_spikes).to(device)
#             spikes_new = torch.cat([spikes, ho_spikes], -1).to(device)
#             output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
#             test_rates.append(output.cpu())
#             test_ho_spikes.append(heldout_spikes.unsqueeze(dim=0)[:, -1, :].cpu())

#     test_rates = torch.cat(test_rates, dim=0).exp() # turn into tensor and use exponential on rates

#     kern_sd = int(round(smth_std / 10))
#     window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
#     window /= np.sum(window)
#     filt = lambda x: np.convolve(x, window, 'same')
#     smth_train_rates = np.apply_along_axis(filt, 0, train_rates)
#     smth_test_rates = np.apply_along_axis(filt, 0, test_rates)
        
#     gscv1 = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv1.fit(train_rates, h5dict['train_vel_segments'])
#     print(i, 'New Decoding', gscv1.best_score_)
#     new_vals.append(gscv1.score(test_rates, h5dict['test_vel_segments']))

#     gscv2 = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv2.fit(smth_train_rates, h5dict['train_vel_segments'])
#     print(i, 'Smoothing', gscv2.best_score_)
#     smooth_vals.append(gscv2.score(smth_test_rates, h5dict['test_vel_segments']))
    
# df['New Decoding'] = new_vals
# df['Smooth Decoding'] = smooth_vals

# print(df)

# df.to_csv('cobps_vs_decoding_new.csv')