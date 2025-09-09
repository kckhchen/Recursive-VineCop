import torch
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.cauchy import Cauchy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from scipy.special import gamma
from scipy.fft import fft, ifft
from scipy.stats import levy_stable, norm

FLT = 1e-6 # for clipping extreme values
TAIL = 0.1 # extra margins for plotting
GRID_SIZE = 500 # controls fineness of grids
FINENESS = 10

def create_perms(data, n_perms):
    perms = []
    for _ in range(n_perms):
        perm_idx = torch.randperm(len(data))
        sequence = data[perm_idx]
        perms.append(sequence)
    return torch.stack(perms)

def H_copula(rho, u, v):
    numer = Normal(0, 1).icdf(u) - rho * Normal(0, 1).icdf(v)
    denom = torch.sqrt(1 - rho ** 2)
    output = Normal(0, 1).cdf(numer / denom)
    output = torch.clip(output, min=FLT, max=1-FLT)
    return output

# work in the log space for numerical stability
def ldbinorm(rho, u, v, loc1=0, scale1=1, loc2=0, scale2=1):
    target_device = u.device
    target_dtype = u.dtype
    mean_vector = torch.tensor([loc1, loc2], dtype=target_dtype, device=target_device)
    cov_matrix = torch.tensor([[scale1, rho], 
                                [rho, scale2]], dtype=target_dtype, device=target_device)
    mvnorm = MultivariateNormal(mean_vector, cov_matrix)
    uv_pair = torch.stack((u, v * torch.ones_like(u)), dim=1)
    # this is for plotting the pdf. v will be a tensor of size 1 (from v=trained_recursion[k] below), so need to be stretched to len(u)
    output = mvnorm.log_prob(uv_pair)
    return output

def c_copula(rho, u, v):
    normal = Normal(0, 1)
    v_d = normal.log_prob(normal.icdf(v))
    u_d = normal.log_prob(normal.icdf(u))
    denom = u_d + v_d
    numer = ldbinorm(rho, normal.icdf(u), normal.icdf(v))
    output = (numer - denom).exp()
    return output


# for Hahn (2018), the functions below use the helper functions above

def train_one_perm(data, init_dist, init_loc, init_scale, rho):
    target_device = data.device
    target_dtype = data.dtype
    rho = torch.as_tensor(rho, dtype=target_dtype, device=target_device)
    grid_lims = torch.min(data) - TAIL, torch.max(data) + TAIL
    grid = torch.linspace(*grid_lims, GRID_SIZE, device=target_device)

    if init_dist == "Normal":
        cdf = Normal(init_loc, init_scale).cdf(data) # the p_0 distribution. Change init_loc and init_scale for better convergence
    elif init_dist == "Cauchy":
        cdf = Cauchy(init_loc, init_scale).cdf(data)
    else:
        cdf = torch.as_tensor(np.interp(data, grid, init_dist), dtype=torch.float)
        
    cdf = torch.clip(cdf, min=FLT, max=1-FLT) # clips extreme values to avoid inf or nan
    predictive_dist = torch.zeros(len(data)) # for storing P_0(Y1), P_1(Y2), etc.
    predictive_dist[0] = cdf[0] # P_0(Y1)

    for k in range(1, len(data)): # compute every predictive dist iteratively
        alpha = (2 - 1/k) * (1/(k+1))
        Cop = H_copula(rho=rho, u=cdf[1:], v=cdf[0])
        cdf = (1 - alpha) * cdf[1:] + alpha * Cop
        cdf = torch.clip(cdf, min=FLT, max=1-FLT)
        predictive_dist[k] = cdf[0] # P_i-1(Y) for i from 2 to n

    return grid, predictive_dist

def get_cdf_pdf(trained_recursion, grid, init_dist, init_loc, init_scale, rho, list=False): # trained_recursion = output from train_one_perm
    target_device = trained_recursion.device
    target_dtype = trained_recursion.dtype
    cdf_grid_list = []
    pdf_grid_list = []

    rho = torch.as_tensor(rho, dtype=target_dtype, device=target_device)
    if init_dist == "Normal": # create grid P_0(X1), P_0(X2) where X's are grid values or test data
        cdf_grid = Normal(init_loc, init_scale).cdf(grid)
        pdf_grid = Normal(init_loc, init_scale).log_prob(grid).exp()
    elif init_dist == "Cauchy":
        cdf_grid = Cauchy(init_loc, init_scale).cdf(grid)
        pdf_grid = Cauchy(init_loc, init_scale).log_prob(grid).exp()
    else:
        cdf_grid = init_dist
        pdf_grid = init_loc
 
    cdf_grid = torch.clip(cdf_grid, min=FLT, max=1-FLT)
    pdf_grid = torch.clip(pdf_grid, min=FLT, max=1-FLT)

    for k in range(0, len(trained_recursion)): # iterate over training size to re-perform the recursion 
        alpha = (2 - 1/(k+1)) * (1/(k+2)) # k starts from 0 so +1
        Cop = H_copula(rho=rho, u=cdf_grid, v=trained_recursion[k])
        cop = c_copula(rho=rho, u=cdf_grid, v=trained_recursion[k])
        cdf_grid = (1 - alpha) * cdf_grid + alpha * Cop
        cdf_grid = torch.clip(cdf_grid, min=FLT, max=1-FLT)
        
        pdf_grid = (1 - alpha) * pdf_grid + alpha * cop * pdf_grid
        pdf_grid = torch.clip(pdf_grid, min=FLT, max=1-FLT)
        if list:
            cdf_grid_list.append(cdf_grid)
            pdf_grid_list.append(pdf_grid)
            if k + 1 == len(trained_recursion):
                return cdf_grid_list, pdf_grid_list
        
    return cdf_grid, pdf_grid

def train_multi_perms(data, init_dist, init_loc, init_scale, rho, n_perms=10):
    target_device = data.device
    target_dtype = data.dtype

    rho = torch.as_tensor(rho, dtype=target_dtype, device=target_device)
    perm_list = create_perms(data, n_perms)
    cdfs = []
    pdfs = []
    for one_perm in perm_list:
        grid, trained = train_one_perm(one_perm, init_dist, init_loc=init_loc, init_scale=init_scale, rho=rho)
        cdf, pdf = get_cdf_pdf(trained, grid, init_dist, rho=rho, init_loc=init_loc, init_scale=init_scale)
        cdfs.append(cdf)
        pdfs.append(pdf)
    return grid, cdfs, pdfs

# to evaluate energy score (crps)
def crps_integral(obs, grid, cdf):
    obs = torch.atleast_1d(obs)
    obs = obs.unsqueeze(1) # Shape: (n_sample, 1)
    sq_diff = (cdf - (grid >= obs).float()) ** 2
    crps_scores = torch.trapz(sq_diff, grid, dim=1)
    return crps_scores

# call the crps_integral above. Take average cdf and do crps computation.
def crps_over_perms(obs, grid, cdf_list, method='integral'):
    assert method == 'integral', "method currently not supported"
    avg_cdf = torch.stack(cdf_list).mean(dim=0)
    crps_scores = crps_integral(obs, grid, avg_cdf)
    return crps_scores

############# time series #############

def plot_series_and_acf(series, axes, lags=100):

    # Plot the time series
    axes[0].plot(series, label='Generated Series', color='blue')
    axes[0].set_title('Time Series Plot')
    axes[0].grid(True)

    # Plot the ACF
    acf_vals = acf(series, nlags=lags, fft=True)
    sns.barplot(x=np.arange(1, lags + 1), y=acf_vals[1:], ax=axes[1], color='orangered')
    axes[1].set_title('Autocorrelation Function (ACF)')
    axes[1].set_xticks(np.arange(-1, 100, 5))
    
    # Add confidence interval lines
    conf_level = 1.96 / np.sqrt(len(series))
    axes[1].axhline(y=conf_level, linestyle='--', color='blue', linewidth=1.2)
    axes[1].axhline(y=-conf_level, linestyle='--', color='blue', linewidth=1.2)

    plt.tight_layout()
    return axes


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

## used for adjusting noise variance

def arma_var(ar_param, ma_param, sigma_epsilon=1.0):
    return sigma_epsilon ** 2 * (1 + 2 * ar_param * ma_param + ma_param ** 2) / (1 - ar_param ** 2)

def arfima0d0_std(d, sigma=1.0):
    return np.sqrt(sigma**2 * gamma(1 - 2*d) / (gamma(1 - d)**2))


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