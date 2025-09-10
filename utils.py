import torch
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.cauchy import Cauchy
import numpy as np

FLT = 1e-6 # for clipping extreme values
TAIL = 0.1 # extra margins for plotting
GRID_SIZE = 500 # controls fineness of grids
FINENESS = 10

def H_copula(rho, u, v):
    numer = Normal(0, 1).icdf(u) - rho * Normal(0, 1).icdf(v)
    denom = torch.sqrt(1 - rho ** 2)
    output = Normal(0, 1).cdf(numer / denom)
    output = torch.clip(output, min=FLT, max=1-FLT)
    return output

def ldbinorm(rho, u, v, locs=[0, 0], scales=[1, 1]):
    # work in the log space for numerical stability
    target_device = u.device
    target_dtype = u.dtype
    mean_vector = torch.tensor(locs, dtype=target_dtype, device=target_device)
    cov_matrix = torch.tensor([[scales[0], rho], 
                               [rho, scales[1]]], dtype=target_dtype, device=target_device)
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

def train_one_perm(data: torch.Tensor,
                   init_dist: str,
                   init_loc: float,
                   init_scale: float,
                   rho: float
                   ):
    """Intermediate Step of the R-BP algorithm.
    Args:
        data (Tensor): 1d tensor for permutation.
        init_dist (str): prior distribution. Can either be "Normal" or "Cauchy".
        init_loc (float): mean of the prior distribution.
        init_scale (float): standard deviation of the prior distribution.
        rho (float): correlation coefficient for the Gaussian copula.
    Returns:
        grid (Tensor): grid for plotting and getting cdf and pdf later.
        trained_recursion (Tensor): P_0(Y1), P_1(Y2), etc., for getting cdf and pdf.
    """
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
    trained_recursion = torch.zeros(len(data)) # for storing P_0(Y1), P_1(Y2), etc.
    trained_recursion[0] = cdf[0] # P_0(Y1)

    for k in range(1, len(data)): # compute every predictive dist iteratively
        alpha = (2 - 1/k) * (1/(k+1))
        Cop = H_copula(rho=rho, u=cdf[1:], v=cdf[0])
        cdf = (1 - alpha) * cdf[1:] + alpha * Cop
        cdf = torch.clip(cdf, min=FLT, max=1-FLT)
        trained_recursion[k] = cdf[0] # P_i-1(Y) for i from 2 to n

    return grid, trained_recursion

def get_cdf_pdf(trained_recursion: torch.Tensor,
                grid: torch.Tensor,
                init_dist: str,
                init_loc: float,
                init_scale: float,
                rho: float,
                list: bool = False
                ):
    """Evaluating the Predictive CDF and PDF.
    Args:
        trained_recursion (Tensor): output from train_one_perm().
        grid (Tensor): grid from train_one_perm().
        init_dist (str): prior distribution, should be the same as in train_one_perm().
        init_loc (float): mean of the prior distribution, should be the same as in train_one_perm().
        init_scale (float): standard deviation of the prior distribution, should be the same as in train_one_perm().
        rho (float): correlation coefficient for the Gaussian copula, should be the same as in train_one_perm().
        list (bool): if true then cdf and pdf in every step will be recorded and returned. default False.
    Returns:
        cdf_grid (Tensor): estimated cdf on grid.
        pdf_grid (Tensor): estimated pdf on grid.
        ** if list = True then this will return a list of cdfs and pdfs, one for every new point. **
    """
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

# to evaluate crps
def crps_integral(obs: torch.Tensor, grid:torch.Tensor , cdf: torch.Tensor) -> torch.Tensor:
    """Evaluating the CRPS.
    Args:
        obs (Tensor): true observations, can be 0d or 1d tensors.
        grid (Tensor): grid to evaluate the CRPS on.
        cdf (Tensor): estimated cdf.
    Returns:
        crps_scores (Tensor): the CRPS for the estimated cdf aganist each observation.
    """
    obs = torch.atleast_1d(obs)
    obs = obs.unsqueeze(1) # Shape: (n_sample, 1)
    sq_diff = (cdf - (grid >= obs).float()) ** 2
    crps_scores = torch.trapz(sq_diff, grid, dim=1)
    return crps_scores