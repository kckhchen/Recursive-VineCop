import xarray as xr
import numpy as np
import pandas as pd
from numba import jit
from utils import *

# generating functions for lorenz63 and lorenz96 
# are taken from https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting

@jit(nopython=True, cache=True)
def l96_truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j - 1]) - c * Y[j] + h * c / b * X[
            int(j / J)]
    return dXdt, dYdt


@jit(nopython=True, cache=True)
def run_lorenz96_truth(x_initial, y_initial, time_step, num_steps, burn_in, skip, h=1, b=10.0, c=10.0, F=20.0):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        F (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    # Calculate total Y forcing over archive period using trapezoidal rule
    y_trap = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    y_trap[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        #if (n * time_step) % 1 == 0:
        #    print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, F, b, c)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2,
                                                h, F, b, c)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2,
                                                h, F, b, c)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step,
                                                h, F, b, c)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = (y + y_trap) / skip
            i += 1
        elif n % skip == 1:
            y_trap[:] = y
        else:
            y_trap[:] += y
    return x_out, y_out, times, steps

# lorenz63

def lorenz(x, y, z, s=10, r=28, b=2.667): # directly taken from Picchiardi's code
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

discard_interval = 15
n_steps = 5000
spinup_steps = 10
integration_steps = (n_steps + spinup_steps) * discard_interval  # then keep one every 30 for the dataset

dt = 0.01

# Need one more for the initial values
xs = np.empty(integration_steps + 1)
ys = np.empty(integration_steps + 1)
zs = np.empty(integration_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point. That is simple Euler integration
for i in range(integration_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

timeseries = ys[::discard_interval][spinup_steps:].reshape(-1, 1)

# lorenz96

dt_observation = 0.2
dt_integration = 0.001
discard_interval = int(dt_observation / dt_integration)
K = 8  # number of observed variables
J = 32  # number of unobserved variable for each observed one
X_init = np.zeros(K)
Y_init = np.zeros(J * K)
X_init[0] = 1
Y_init[0] = 1
h = 1
b = 10.0
c = 10.0
F = 20.0
n_steps = 5000
discard_interval = 20
burnin_steps = 2000

burnin_steps = int(2 / dt_integration)  # discard two time units of burn-in
total_integration_steps = n_steps * discard_interval + burnin_steps

timeseries, Y_out, times, steps = run_lorenz96_truth(X_init, Y_init, dt_integration, total_integration_steps,
                                                        burn_in=burnin_steps, skip=discard_interval, h=h, F=F, b=b,
                                                        c=c)
timeseries


# ar3
n_samples = 5000
ar3_samples = generate_arma(ar_params=[0.3, 0.25, 0.1, 0.2], n_samples=n_samples, burn_in=500)


# weatherbench

dataset = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr')

z500_data = dataset.geopotential.sel(level=500)
z500_data = z500_data.transpose('time', 'latitude', 'longitude')

all_data = []
for bin_idx in range(50):
    bin_start = (50 - bin_idx) * -100
    bin_end = (50 - bin_idx - 1) * -100
    print("bin", bin_idx)
    part_data = z500_data.sel(latitude=52.5, longitude=358.5).isel(time=range(bin_start, bin_end)).load()
    all_data.append(part_data)

final_time_series = xr.concat(all_data, dim='time')

pd.DataFrame(final_time_series).to_csv("uwarwick_z500.csv")

## ARMA

def generate_arma(ar_params=[], ma_params=[], n_samples=1000, drift=0, sigma=1, burn_in=200):
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
    return series[burn_in:]


## ARFIMA data generator ##
# taken directly from https://github.com/akononovicius/arfima

def __ma_model(
    params: list[float],
    n_points: int,
    *,
    noise_std: float = 1,
    noise_alpha: float = 2,
) -> list[float]:
    """Generate discrete series using MA process.

    Args:
        params: list[float]
            Coefficients used by the MA process:
                x[t] = epsi[t] + params[1]*epsi[t-1] + params[2]*epsi[t-2] + ...
            Order of the MA process is inferred from the length of this array.
        n_points: int
            Number of points to generate.
        noise_std: float, optional
            Scale of the generated noise (default: 1).
        noise_alpha: float, optional
            Parameter of the alpha-stable distribution (default: 2). Default
            value corresponds to Gaussian distribution.

    Returns:
        Discrete series (array of length n_points) generated by
        MA(len(params)) process
    """
    ma_order = len(params)
    if noise_alpha == 2:
        noise = norm.rvs(scale=noise_std, size=(n_points + ma_order))
    else:
        noise = levy_stable.rvs(
            noise_alpha, 0, scale=noise_std, size=(n_points + ma_order)
        )

    if ma_order == 0:
        return noise
    ma_coeffs = np.append([1], params)
    ma_series = np.zeros(n_points)
    for idx in range(ma_order, n_points + ma_order):
        take_idx = np.arange(idx, idx - ma_order - 1, -1).astype(int)
        ma_series[idx - ma_order] = np.dot(ma_coeffs, noise[take_idx])
    return ma_series[ma_order:]


def __arma_model(params: list[float], noise: list[float]) -> list[float]:
    """Generate discrete series using ARMA process.

    Args:
        params: list[float]
            Coefficients used by the AR process:
                x[t] = params[1]*x[t-1] + params[2]*x[t-2] + ... + epsi[t]
            Order of the AR process is inferred from the length of this array.
        noise: list[float]
            Values of the noise for each step. Length of the output array is
            automatically inferred from the length of this array. Note that
            noise needs not to be standard Gaussian noise (MA(0) process). It
            may be also generated by a higher order MA process.

    Returns:
        Discrete series (array of the same length as noise array) generated by
        the ARMA(len(params), ?) process.
    """
    ar_order = len(params)
    if ar_order == 0:
        return noise
    n_points = len(noise)
    arma_series = np.zeros(n_points + ar_order)
    for idx in np.arange(ar_order, len(arma_series)):
        take_idx = np.arange(idx - 1, idx - ar_order - 1, -1).astype(int)
        arma_series[idx] = np.dot(params, arma_series[take_idx]) + noise[idx - ar_order]
    return arma_series[ar_order:]


def __frac_diff(x: list[float], d: float) -> list[float]:
    """Fast fractional difference algorithm (by Jensen & Nielsen (2014)).

    Args:
        x: list[float]
            Array of values to be differentiated.
        d: float
            Order of the differentiation. Recommend to use -0.5 < d < 0.5, but
            should work for almost any reasonable d.

    Returns:
        Fractionally differentiated series.
    """

    def next_pow2(n):
        # we assume that the input will always be n > 1,
        # so this brief calculation should be fine
        return (n - 1).bit_length()

    n_points = len(x)
    fft_len = 2 ** next_pow2(2 * n_points - 1)
    prod_ids = np.arange(1, n_points)
    frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
    dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
    return np.real(dx[0:n_points])


def arfima(
    ar_params: list[float],
    d: float,
    ma_params: list[float],
    n_points: int,
    *,
    noise_std: float = 1,
    noise_alpha: float = 2,
    warmup: int = 0,
) -> list[float]:
    """Generate series from ARFIMA process.

    Args:
        ar_params: list[float]
            Coefficients to be used by the AR process.
        d: float
            Differentiation order used by the ARFIMA process.
        ma_params: list[float]
            Coefficients to be used by the MA process.
        n_points: int
            Number of points to generate.
        noise_std: float, optional
            Scale of the generated noise (default: 1).
        noise_alpha: float, optional
            Parameter of the alpha-stable distribution (default: 2). Default
            value corresponds to Gaussian distribution.
        warmup: int, optional
            Number of points to generate as a warmup for the model
            (default: 0).

    Returns:
        Discrete series (array of length n_points) generated by the
        ARFIMA(len(ar_params), d, len(ma_params)) process.
    """
    ma_series = __ma_model(
        ma_params, n_points + warmup, noise_std=noise_std, noise_alpha=noise_alpha
    )
    frac_ma = __frac_diff(ma_series, -d)
    series = __arma_model(ar_params, frac_ma)
    return series[-n_points:]