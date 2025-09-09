import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='Recursive-VineCop training')
parser.add_argument('--folder', type=str, help='Data folder name', default="./data")
parser.add_argument('--name', type=str, help='Data name', default="unnamed_ARMA")
parser.add_argument('--ar_params', type=str, help='AR parameters, string with values separated with commas', default="0.")
parser.add_argument('--ma_params', type=str, help='MA parameters, string with values separated with commas', default="0.")
parser.add_argument('--n_samples', type=int, help='Number of samples', default=5000)
parser.add_argument('--drift', type=float, help='drift parameter', default=0.)
parser.add_argument('--sigma', type=float, help='Noise variance', default=1.)
parser.add_argument('--burn_in', type=int, help='Burn in size', default=500)
parser.add_argument('--seed', type=int, help='Random seed', default=None)
args = parser.parse_args()

folder = args.folder
data_name = args.name
ar_params = [float(item) for item in args.ar_params.split(',')]
ma_params = [float(item) for item in args.ma_params.split(',')]
n_samples = args.n_samples
drift = args.drift
sigma = args.sigma
burn_in = args.burn_in
seed = args.seed

if seed is not None: np.random.seed(seed)

if not os.path.exists(folder):
    os.makedirs(folder)

p = len(ar_params)
q = len(ma_params)
total_len = n_samples + burn_in
white_noise = np.random.normal(loc=0, scale=sigma, size=total_len)
series = np.zeros(total_len)

for t in range(max(p, q), total_len):
    ar_term = 0
    if p > 0:
        ar_vals = series[t-p:t][::-1] - drift
        ar_term = np.dot(ar_params, ar_vals)

    ma_term = 0
    if q > 0:
        ma_vals = white_noise[t-q:t][::-1]
        ma_term = np.dot(ma_params, ma_vals)

    series[t] = drift + ar_term + ma_term + white_noise[t]

series = series[burn_in:]
pd.DataFrame(series).to_csv(folder + "/" + data_name + ".csv")

