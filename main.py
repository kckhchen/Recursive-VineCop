import numpy as np
import pandas as pd
import time
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
import argparse
import pyvinecopulib as pv
from scipy.integrate import cumulative_trapezoid
from utils import train_one_perm, get_cdf_pdf, crps_integral
from trainers import SlidingWindowDataset, inv_cdf_transform, train_rho, train_vinecop

parser = argparse.ArgumentParser(description='Recursive-VineCop Training and Testing')
parser.add_argument('--data_folder', type=str, help='name of folder where data are stored', default="./data")
parser.add_argument('--data_name', type=str, help='name of the data, only csv files are allowed', default='AR3')
parser.add_argument('--component', type=int, help='which data component to use, only eligible for Lorenz96', default=1)
parser.add_argument('--fig_folder', type=str, help='name of folder to store result figures', default="./results")
parser.add_argument('--train_size', type=int, help='training data size (proportion)', default=0.3)
parser.add_argument('--val_size', type=int, help='training data size (proportion)', default=0.2)
parser.add_argument('--init_dist', type=str, help='prior distrbution, can be Normal or Cauchy', default='Cauchy')
parser.add_argument('--init_loc', type=float, help='initial mean for the prior', default=0.)
parser.add_argument('--init_scale', type=float, help='initial standard deviation for the prior', default=1.)
parser.add_argument('--init_rho', type=float, help='initial rho value for training', default=0.1)
parser.add_argument('--set_rho', type=float, help='set rho value and skip the optimisation process')
parser.add_argument('--max_window', type=int, help='max window size allowed for vine copula', default=10)
parser.add_argument('--tolerance', type=float, help='tolerance for rho optimisation early stopping', default=1e-4)
parser.add_argument('--patience', type=int, help='patience for rho optimisation early stopping', default=5)
parser.add_argument('--max_iter', type=int, help='max iterations for rho optimisation', default=100)
parser.add_argument('--lr', type=float, help='learning rate for rho optimisation early stopping', default=0.05)
parser.add_argument('--eta_min', type=float, help='mininum learning rate for the scheduler to decay to', default=1e-3)
parser.add_argument('--n_lags', type=int, help='prediction lead time i.e. how many steps ahead to be predicted', default=1)
parser.add_argument('--trunc_lvl', type=int, help='truncation level for vine copula', default=5)
parser.add_argument('--vine_structure', type=str, help='type of vine to use, can be C, D, or R', default='D')
parser.add_argument('--ci', type=float, help='credible interval for point forecasts', default=0.9)
parser.add_argument('--plot_length', type=int, help='number of observations to be shown in forecast plot', default=100)
args = parser.parse_args()

start_time = time.time()

data_folder = args.data_folder
fig_folder =  args.fig_folder
data_name = args.data_name
component = args.component
train_size = args.train_size
val_size = args.val_size
init_dist = args.init_dist
init_loc = args.init_loc
init_scale = args.init_scale
init_rho = args.init_rho
set_rho = args.set_rho
max_window = args.max_window
n_lags = args.n_lags
trunc_lvl = args.trunc_lvl
vine_structure = args.vine_structure
ci = args.ci
plot_length = args.plot_length

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

data_path = data_folder + "/" + data_name + ".csv"

time_series = pd.read_csv(data_path, index_col=0).to_numpy()
if data_name == "L96": time_series = time_series[:, component-1]
time_series = torch.as_tensor(time_series, dtype=torch.float).flatten()

## Summary plots
_, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
axes[0].plot(time_series, label='Generated Series', color='blue')
axes[0].set_title('Time Series Plot')
axes[0].grid(True)
acf_vals = acf(time_series, nlags=100, fft=True)
sns.barplot(x=np.arange(1, 100 + 1), y=acf_vals[1:], ax=axes[1], color='orangered')
axes[1].set_title('Autocorrelation Function (ACF)')
axes[1].set_xticks(np.arange(-1, 100, 5))
conf_level = 1.96 / np.sqrt(len(time_series))
axes[1].axhline(y=conf_level, linestyle='--', color='blue', linewidth=1.2)
axes[1].axhline(y=-conf_level, linestyle='--', color='blue', linewidth=1.2)
plt.tight_layout()
plt.savefig(fig_folder + "/" + data_name + "_summary.png")
plt.close()

## Standardisation and split
time_series = (time_series - torch.mean(time_series)) / torch.std(time_series)
train_val_size = int(len(time_series) * (train_size + val_size))
train_size = int(len(time_series) * train_size)
train_samples = time_series[:train_size]
val_samples = time_series[train_size:train_val_size]
train_val_samples = time_series[:train_val_size]
test_samples = time_series[train_val_size:]

## Marginal estimation (or specifiy rho)
if set_rho is None:
    rho = train_rho(train_samples, val_samples, init_dist, init_loc, init_scale, init_rho,
                    args.max_iter, args.lr, args.patience, args.tolerance, args.eta_min, fig_folder, data_name)
else:
    rho = set_rho

grid, trained = train_one_perm(train_val_samples, init_dist, init_loc, init_scale, rho)
cdf, pdf = get_cdf_pdf(trained, grid, init_dist, init_loc, init_scale, rho)

## Marginal estimation figure
plt.figure(figsize=(7, 5))
plt.hist(train_val_samples, bins='auto', density=True, color='lightgrey', label="Observations")
plt.plot(grid, pdf, label="Estimated density")
plt.title(f"Estimated marginal distribution with rho={rho:.3f}")
plt.legend()
plt.savefig(fig_folder + "/" + data_name + "_marginal.png")
plt.close()

## vine fitting
vine, window_size, best_crps = train_vinecop(train_samples, val_samples, grid, cdf, pdf, n_lags, vine_structure, trunc_lvl, max_window)

## Bivariate copula fitting
train_dataset = SlidingWindowDataset(train_val_samples, window_size, steps_ahead=n_lags)
test_dataset = SlidingWindowDataset(test_samples, window_size, steps_ahead=n_lags)
test_size = len(test_dataset)
train_histories_targets = torch.stack([torch.cat((history, target.unsqueeze(-1))) for history, target in train_dataset]).squeeze(1)
train_histories_targets_unif = inv_cdf_transform(train_histories_targets, grid, cdf)
bv_cop = pv.Bicop.from_data(train_histories_targets_unif[:, -2:])

## Forecast plot
crps = torch.zeros(3, test_size) # (vine, bv, naive)
forecasts_vine = torch.zeros(3, test_size) # (median, upper_ci, lower_ci)
forecasts_bv = torch.zeros(3, test_size)

test_histories = torch.stack([history for history, _ in test_dataset]).T
true_values = torch.stack([target for _, target in test_dataset])
test_histories_unif = inv_cdf_transform(test_histories, grid, cdf)

for i in range(test_size):
    true_value = true_values[i]
    repeated_array = np.tile(test_histories_unif[:, i], (len(cdf), 1))

    est_pdf = vine.pdf(np.column_stack([repeated_array, cdf])) * pdf.numpy()
    est_pdf = est_pdf / np.trapezoid(est_pdf, grid)
    est_cdf = torch.tensor(cumulative_trapezoid(est_pdf, grid, initial=0), dtype=torch.float)

    est_pdf_bv = bv_cop.pdf(np.column_stack([repeated_array[:, -1], cdf])) * pdf.numpy()
    est_pdf_bv = est_pdf_bv / np.trapezoid(est_pdf_bv, grid)
    est_cdf_bv = torch.tensor(cumulative_trapezoid(est_pdf_bv, grid, initial=0), dtype=torch.float)

    crps[:, i] = crps_integral(true_value, grid, torch.stack([est_cdf, est_cdf_bv, cdf]))
    
    forecasts_vine[:, i] = torch.stack([grid[torch.argmin(abs(est_cdf - 0.5))],
                                        grid[torch.argmin(abs(est_cdf - (1 - ci) / 2))],
                                        grid[torch.argmin(abs(est_cdf - (1 - (1 - ci) / 2)))]])
    
    forecasts_bv[:, i] = torch.stack([grid[torch.argmin(abs(est_cdf_bv - 0.5))],
                                      grid[torch.argmin(abs(est_cdf_bv - (1 - ci) / 2))],
                                      grid[torch.argmin(abs(est_cdf_bv - (1 - (1 - ci) / 2)))]])

plt.figure(figsize=(15, 4))
ax = plt.gca()
plt.plot(forecasts_bv[0, :plot_length], label="BiCop", color='orange')
plt.plot(forecasts_vine[0, :plot_length], label="VineCop")
ax.fill_between(range(plot_length), *forecasts_bv[1:, :plot_length], alpha=0.3, color='orange')
ax.fill_between(range(plot_length), *forecasts_vine[1:, :plot_length], alpha=0.4)
plt.plot(true_values[:plot_length], '--', label="True", color='black')
plt.legend()
plt.savefig(fig_folder + "/" + data_name + "_forecasts.png")
plt.close()

## Boxplots
plt.figure(figsize=(7, 3))
plt.boxplot(crps.T, orientation="horizontal", tick_labels=['Vine', 'Bicop', 'Naïve'])
ax = plt.gca()
ax.invert_yaxis()
plt.savefig(fig_folder + "/" + data_name + "_boxplots.png")
plt.close()

## Metrics
forecasts_naive = grid[torch.argmin(abs(cdf - 0.5))].expand(test_size)
forecasts_persistence = torch.cat([train_dataset[-1][1].reshape(1), true_values[:-1]])
rmse = torch.abs(torch.stack([forecasts_vine[0],
                              forecasts_bv[0],
                              forecasts_naive,
                              forecasts_persistence]) - true_values).mean(dim=1).sqrt().numpy()

print("\n------ Performance Comparison ------")
print(pd.DataFrame([crps.mean(dim=1).numpy().round(5),
                    crps.std(dim=1).numpy().round(5),
                    rmse.round(5)],
                    columns=["VineCop", "BiCop", "Naive", "Persistence"], index=["Mean CRPS", "Std CRPS", "RMSE"]).T)

elapsed_time = time.time() - start_time
print(f"\n--- Time elapsed: {elapsed_time:.4f} seconds ---\n")