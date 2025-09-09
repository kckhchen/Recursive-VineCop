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
from torch.utils.data import Subset
from utils import *
from trainers import *

parser = argparse.ArgumentParser(description='Recursive-VineCop training')
parser.add_argument('--data_folder', type=str, help='name of folder where data are stored', default="./data")
parser.add_argument('--data_name', type=str, help='name of the data', default='AR3')
parser.add_argument('--component', type=int, help='which data component to use, only eligible for Lorenz96', default=1)
parser.add_argument('--fig_folder', type=str, help='name of folder to store result figures', default="./results")
parser.add_argument('--train_size', type=int, help='training data size', default=1500)
parser.add_argument('--init_dist', type=str, help='prior distrbution, can be Normal or Cauchy', default='Cauchy')
parser.add_argument('--init_loc', type=float, help='initial mean for the prior', default=0.)
parser.add_argument('--init_scale', type=float, help='initial standard deviation for the prior', default=1.)
parser.add_argument('--init_rho', type=float, help='initial rho value for training', default=0.1)
parser.add_argument('--train_prop', type=float, help='train-validation split for rho and vine training', default=0.7)
parser.add_argument('--max_window', type=int, help='max window size allowed for vine copula', default=10)
parser.add_argument('--tolerance', type=float, help='tolerance for rho optimisation early stopping', default=1e-4)
parser.add_argument('--patience', type=int, help='patience for rho optimisation early stopping', default=5)
parser.add_argument('--max_iter', type=int, help='max iterations for rho optimisation', default=500)
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
folder =  args.fig_folder
data_name = args.data_name
component = args.component
train_size = args.train_size
init_dist = args.init_dist
init_loc = args.init_loc
init_scale = args.init_scale
init_rho = args.init_rho
train_prop = args.train_prop
max_window = args.max_window
n_lags = args.n_lags
trunc_lvl = args.trunc_lvl
vine_structure = args.vine_structure
ci = args.ci
plot_length = args.plot_length

if not os.path.exists(folder):
    os.makedirs(folder)

data_path = data_folder + "/" + data_name + ".csv"

full_dataset = pd.read_csv(data_path, index_col=0).to_numpy()
if data_name == "L96": full_dataset = full_dataset[:, component-1]
full_dataset = torch.as_tensor(full_dataset, dtype=torch.float).flatten()

## Summary plots
_, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
axes[0].plot(full_dataset, label='Generated Series', color='blue')
axes[0].set_title('Time Series Plot')
axes[0].grid(True)
acf_vals = acf(full_dataset, nlags=100, fft=True)
sns.barplot(x=np.arange(1, 100 + 1), y=acf_vals[1:], ax=axes[1], color='orangered')
axes[1].set_title('Autocorrelation Function (ACF)')
axes[1].set_xticks(np.arange(-1, 100, 5))
conf_level = 1.96 / np.sqrt(len(full_dataset))
axes[1].axhline(y=conf_level, linestyle='--', color='blue', linewidth=1.2)
axes[1].axhline(y=-conf_level, linestyle='--', color='blue', linewidth=1.2)
plt.tight_layout()
plt.savefig(folder + "/" + data_name + "_summary.png")
plt.close()

## Standardisation
data_mean = torch.mean(full_dataset)
data_std = torch.std(full_dataset)
full_dataset = (full_dataset - data_mean) / data_std
train_samples = full_dataset[:train_size]
test_samples = full_dataset[train_size:]

## Marginal estimation
rho = train_rho(train_samples, init_dist, init_loc, init_scale, init_rho,
                args.max_iter, args.lr, args.patience, args.tolerance, args.train_prop, args.eta_min, folder, data_name)
grid, trained = train_one_perm(train_samples, init_dist, init_loc, init_scale, rho)
cdf, pdf = get_cdf_pdf(trained, grid, init_dist, init_loc, init_scale, rho)

## Marginal estimation figure
plt.figure(figsize=(7, 5))
plt.hist(train_samples, bins='auto', density=True, color='lightgrey', label="Observations")
plt.plot(grid, pdf, label="Estimated density")
plt.title(f"Estimated marginal distribution with rho={rho:.3f}")
plt.legend()
plt.savefig(folder + "/" + data_name + "_marginal.png")
plt.close()

## vine fitting
vine, window_size, best_crps = train_vinecop(full_dataset, grid, cdf, pdf, n_lags, vine_structure, trunc_lvl, train_prop, max_window)

## Bivariate copula fitting
dataset = SlidingWindowDataset(full_dataset, window_size, steps_ahead=n_lags)
dataset_size = len(dataset)
train_set = Subset(dataset, list(range(train_size)))
test_set = Subset(dataset, list(range(train_size, dataset_size)))
histories = torch.stack([torch.cat((history, target.unsqueeze(-1))) for history, target in train_set]).squeeze(1)
histories_u = inv_cdf_transform(histories, grid, cdf)
bicop = pv.Bicop.from_data(histories_u[:, -2:])

## Forecast plot
crps = []
bicop_crps = []
naive_crps = []
point_forecasts = []
ci_lower_list = []
ci_upper_list = []
bv_point_forecasts = []
ci_lower_list_bv = []
ci_upper_list_bv = []

test_histories = torch.stack([history for history, _ in test_set]).T
true_values = torch.stack([target for _, target in test_set])
test_histories_u = inv_cdf_transform(test_histories, grid, cdf)

for i in range(len(test_set)):
    true_value = true_values[i]
    repeated_array = np.tile(test_histories_u[:, i], (len(cdf), 1))

    est_pdf = vine.pdf(np.column_stack([repeated_array, cdf])) * pdf.numpy()
    est_pdf = est_pdf / np.trapezoid(est_pdf, grid)
    est_cdf = cumulative_trapezoid(est_pdf, grid, initial=0)

    est_pdf_bicop = bicop.pdf(np.column_stack([repeated_array[:, -1], cdf])) * pdf.numpy()
    est_pdf_bicop = est_pdf_bicop / np.trapezoid(est_pdf_bicop, grid)
    est_cdf_bicop = cumulative_trapezoid(est_pdf_bicop, grid, initial=0)

    crps.append(crps_integral(true_value, grid, torch.as_tensor(est_cdf, dtype=torch.float)))
    bicop_crps.append(crps_integral(true_value, grid, torch.as_tensor(est_cdf_bicop, dtype=torch.float)))
    naive_crps.append(crps_integral(true_value, grid, torch.as_tensor(cdf, dtype=torch.float)))

    point_forecasts.append(grid[np.argmin(abs(est_cdf - 0.5))])
    ci_lower_list.append(grid[np.argmin(abs(est_cdf - (1 - ci) / 2))])
    ci_upper_list.append(grid[np.argmin(abs(est_cdf - (1 - (1 - ci) / 2)))])
    bv_point_forecasts.append(grid[np.argmin(abs(est_cdf_bicop - 0.5))])
    ci_lower_list_bv.append(grid[np.argmin(abs(est_cdf_bicop - (1 - ci) / 2))])
    ci_upper_list_bv.append(grid[np.argmin(abs(est_cdf_bicop - (1 - (1 - ci) / 2)))])

plt.figure(figsize=(15, 4))
pred = torch.stack(point_forecasts)
bv_pred = torch.stack(bv_point_forecasts)
ci_upper = torch.stack(ci_upper_list)
ci_lower = torch.stack(ci_lower_list)
ci_upper_bv = torch.stack(ci_upper_list_bv)
ci_lower_bv = torch.stack(ci_lower_list_bv)
ax = plt.gca()
plt.plot(bv_pred[:plot_length], label="BiCop", color='orange')
plt.plot(pred[:plot_length], label="VineCop")
ax.fill_between(range(plot_length), ci_lower_bv[:plot_length], ci_upper_bv[:plot_length], alpha=0.3, color='orange')
ax.fill_between(range(plot_length), ci_lower[:plot_length], ci_upper[:plot_length], alpha=0.4)
plt.plot(true_values[:plot_length], '--', label="True", color='black')
plt.legend()
plt.savefig(folder + "/" + data_name + "_forecasts.png")
plt.close()

## Boxplots
plt.figure(figsize=(7, 3))
plt.boxplot(torch.stack([torch.stack(crps).flatten(),
                         torch.stack(bicop_crps).flatten(),
                         torch.stack(naive_crps).flatten()],
                         dim=1),
            orientation="horizontal", tick_labels=['Vine', 'Bicop', 'Naïve'])
ax = plt.gca()
ax.invert_yaxis()
plt.savefig(folder + "/" + data_name + "_boxplots.png")
plt.close()

## Metrics
mean_crps_vine = torch.stack(crps).mean().numpy()
mean_crps_bicop = torch.stack(bicop_crps).mean().numpy()
mean_crps_stationary = torch.stack(naive_crps).mean().numpy()
std_crps_vine = torch.stack(crps).std().numpy()
std_crps_bicop = torch.stack(bicop_crps).std().numpy()
std_crps_stationary = torch.stack(naive_crps).std().numpy()

result_df = pd.DataFrame([[mean_crps_vine.round(5), mean_crps_bicop.round(5), mean_crps_stationary.round(5)],
                    [std_crps_vine.round(5), std_crps_bicop.round(5), std_crps_stationary.round(5)]],
                    columns=["VineCop", "BiCop", "Naive"], index=["Mean", "Std"])

print("\n--- CRPS Comparison ---\n", result_df)

pred = torch.stack(point_forecasts)
rmse = torch.abs(pred - true_values).mean().sqrt()

bv_pred = torch.stack(bv_point_forecasts)
bv_rmse = torch.abs(bv_pred - true_values).mean().sqrt()

stationary_pf = grid[np.argmin(abs(cdf - 0.5))]
sta_rmse = torch.abs(stationary_pf - true_values).mean().sqrt()

per_rmse = torch.abs(true_values[1:] - true_values[:-1]).mean().sqrt()

print("\n--- RMSE Comparison ---",
      "\nVineCop:    ", rmse.numpy().round(5),
      "\nBiCop:      ", bv_rmse.numpy().round(5), 
      "\nNaive:      ", sta_rmse.numpy().round(5), 
      "\nPersistence:", per_rmse.numpy().round(5))

elapsed_time = time.time() - start_time
print(f"\n--- Time elapsed: {elapsed_time:.4f} seconds ---\n")