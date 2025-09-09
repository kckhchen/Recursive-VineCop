import numpy as np
import pandas as pd
from numba import jit
import argparse
import os

parser = argparse.ArgumentParser(description='Recursive-VineCop training')
parser.add_argument('--folder', type=str,  help='Data folder name', default="./data")
parser.add_argument('--name', type=str, help='Data name, L63 or L96', default="L63")
parser.add_argument('--n_samples', type=int, help='Number of samples', default=5000)
parser.add_argument('--burn_in', type=int, help='Burn in steps', default=100)
parser.add_argument('--discard_interval', type=int, help='Discard interval', default=15)
args = parser.parse_args()

folder = args.folder
data_name = args.name
n_samples= args.n_samples
burn_in = args.burn_in
discard_interval = args.discard_interval

if not os.path.exists(folder):
    os.makedirs(folder)

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

def lorenz(x, y, z, s=10, r=28, b=2.667): # directly taken from Picchiardi's code
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

if data_name == "L63":
    integration_steps = (n_samples + burn_in) * discard_interval 
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

    timeseries = ys[::discard_interval][burn_in:].reshape(-1, 1)
    pd.DataFrame(timeseries).to_csv(folder + "/" + data_name + ".csv")

if data_name == "L96":
    dt_integration = 0.001
    dt_observation = dt_integration * discard_interval
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

    burn_in = int(2 / dt_integration)  # discard two time units of burn-in
    total_integration_steps = n_samples * discard_interval + burn_in

    timeseries, Y_out, times, steps = run_lorenz96_truth(X_init, Y_init, dt_integration, total_integration_steps,
                                                            burn_in=burn_in, skip=discard_interval, h=h, F=F, b=b,
                                                            c=c)
    pd.DataFrame(timeseries).to_csv(folder + "/" + data_name + ".csv")

