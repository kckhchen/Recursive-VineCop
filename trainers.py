import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pyvinecopulib as pv
from scipy.integrate import cumulative_trapezoid
import numpy as np
from tqdm import tqdm
from utils import *

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size=10, steps_ahead=1):
        self.data = data
        self.steps_ahead = steps_ahead
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size - self.steps_ahead + 1
    def __getitem__(self, idx):
        history = self.data[idx : idx + self.window_size]
        target = self.data[idx + self.window_size + self.steps_ahead - 1]
        return history, target
    
def inv_cdf_transform(data, grid, cdf):
    return np.interp(data, grid, cdf)

class EarlyStopping:
    def __init__(self, patience=5, tolerance=1e-5, verbose=False):
        self.patience = patience
        self.tolerance = tolerance
        self.patience_counter = 0
        self.best_value = float('inf')
        self.verbose = verbose

    def __call__(self, current_value):
        if current_value < self.best_value - self.tolerance:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f"--- Early Stopping Countdown: {self.patience_counter}/{self.patience} ---")
        
        if self.patience_counter >= self.patience:
            if self.verbose: print("Early stopping criterion met.")
            return True
        return False

def train_rho(train_samples: np.ndarray | torch.Tensor,
              val_samples: np.ndarray | torch.Tensor,
              init_dist: str,
              init_loc: float,
              init_scale: float,
              init_rho: float,
              max_iter: int,
              lr: float,
              patience: int,
              tolerance: float,
              eta_min: float,
              figpath: str,
              data_name: str
              ):
    """Optimise rho for the R-BP algorithm.
    Args:
        tran_samples (ndarray or Tensor): training samples.
        val_samples (ndarray or Tensor): validation samples.
        init_dist (str): prior distribution. Can either be "Normal" or "Cauchy".
        init_loc (float): mean of the prior distribution.
        init_scale (float): standard deviation of the prior distribution.
        init_rho (float): starting rho value.
        max_iter (int): max number of iteration.
        lr (float): learning rate.
        patience (int): patience of the early stopper.
        tolerance (float): tolerance of the early stopper.
        eta_min (float): min rate of the scheduler.
        figpath (str): folder for storing the loss plot.
        data_name (str): name of the data, for naming the loss plot.
    Returns:
        final_rho (Tensor): final optimised rho.
    """
    rho = torch.tensor(np.log(init_rho / (1-init_rho)), dtype=torch.float, requires_grad=True)
    optimizer = torch.optim.Adam([rho], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

    early_stopper = EarlyStopping(patience=patience, tolerance=tolerance, verbose=False)
    rho_history = []
    train_loss_history = []
    val_loss_history = []

    train_samples = torch.as_tensor(train_samples, dtype=torch.float)
    val_samples = torch.as_tensor(val_samples, dtype=torch.float)
    train_size = len(train_samples)

    pbar = tqdm(range(max_iter), desc="Optimizing Rho")
    for i in pbar:
        crps_train_list = []
        optimizer.zero_grad()
        current_rho = torch.sigmoid(rho)

        grid, trained = train_one_perm(train_samples, init_dist, init_loc, init_scale, current_rho)
        cdfs, _ = get_cdf_pdf(trained, grid, init_dist, init_loc, init_scale, current_rho, list=True)
        for i in range(train_size):
            crps = crps_integral(train_samples[i], grid, cdfs[i])
            crps_train_list.append(crps)

        train_loss = torch.stack(crps_train_list).mean()

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            current_rho_val = torch.sigmoid(rho)
            val_loss = crps_integral(val_samples, grid, cdfs[-1]).mean()

        rho_history.append(current_rho_val.item())
        train_loss_history.append(train_loss.item())
        val_loss_history.append(val_loss.item())
        pbar.set_postfix({"Rho": current_rho_val.item()})

        if early_stopper(val_loss.item()):
            break

    final_rho = torch.sigmoid(rho).detach()
    print(f"Optimization finished. Final optimized rho: {final_rho.item():.5f}")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    line1, = ax1.plot(train_loss_history, 'g-', label="Training loss")
    line2, = ax2.plot(val_loss_history, 'b-', label="Validation loss")
    ax1.set_xlabel("Iterations");ax1.set_ylabel("Training loss");ax2.set_ylabel("Validation Loss")
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.savefig(figpath + "/" + data_name + "_loss.png")
    plt.close()

    return final_rho


def train_vinecop(train_samples: torch.Tensor,
                  val_samples: torch.Tensor,
                  grid: torch.Tensor,
                  cdf: torch.Tensor,
                  pdf: torch.Tensor,
                  n_lags: int,
                  vine_structure: str,
                  trunc_lvl: int,
                  max_window: int
                  ):
    """Optimise Observation Window Size and Fit Final Vine.
    Args:
        data (Tensor): training and validation samples.
        grid (Tensor): grid of the cdf and pdf.
        cdf (Tensor): estimated marginal cdf.
        pdf (Tensor): estimated marginal pdf.
        n_lags (int): how many steps ahead to predict.
        vine_structure (str): which vine structure to use. Can be 'C', 'D', or 'R'
        trunc_lvl (int): truncation level for the vine copula.
        max_window (int): max observation window size allowed.
    Returns:
        best_vine (Vinecop object): the optimal vine copula.
        opt_window_size (int): the optimal observation window size.
        best_crps (float): the CRPS yielded by the best_vine with opt_window_size.
    """
    assert vine_structure in ['D', 'C', 'R'], "Vine structure not supported."
    print("\nOptimizing observation window size...")
    best_crps = np.inf
    window_size = 1

    while True:
        train_dataset = SlidingWindowDataset(train_samples, window_size, steps_ahead=n_lags)
        val_dataset = SlidingWindowDataset(val_samples, window_size, steps_ahead=n_lags)
        histories = torch.stack([torch.cat((history, target.unsqueeze(-1))) for history, target in train_dataset]).squeeze(1)
        histories_unif = inv_cdf_transform(histories, grid, cdf)

        val_histories = torch.stack([history for history, _ in val_dataset]).T
        true_values = torch.stack([target for _, target in val_dataset])
        val_histories_unif = inv_cdf_transform(val_histories, grid, cdf)

        controls = pv.FitControlsVinecop(
            family_set=[pv.gaussian, pv.student, pv.tll, pv.indep],
            trunc_lvl=trunc_lvl,
            select_trunc_lvl=False
        )
        
        if vine_structure == 'D':
            structure = pv.DVineStructure(order=range(1, window_size+2))
        elif vine_structure == 'C':
            structure = pv.CVineStructure(order=range(1, window_size+2))
        elif vine_structure == 'R':
            structure = pv.RVineStructure(order=range(1, window_size+2))
                  
        vine = pv.Vinecop.from_structure(structure)
        vine.select(histories_unif, controls=controls)

        crps = []

        for i in range(0, len(val_dataset)):
            true_value = true_values[-i]
            repeated_array = np.tile(val_histories_unif[:, -i], (len(cdf), 1))

            den = vine.pdf(np.column_stack([repeated_array, cdf]))
            den = den / np.trapezoid(den, cdf)
            final_den = pdf.numpy() * den
            final_den = final_den / np.trapezoid(final_den, grid)
            est_cdf = cumulative_trapezoid(final_den, grid, initial=0)
            crps.append(crps_integral(true_value, grid, torch.as_tensor(est_cdf, dtype=torch.float)))
        
        avg_crps = torch.stack(crps).mean().numpy()

        if avg_crps <= best_crps and max_window <= 10:
            print("indow_size:", window_size, "current CRPS", avg_crps)
            best_crps = avg_crps
            best_vine = vine
            opt_window_size = window_size
            window_size += 1
        else:
            print("Optimization finished. Best window size:", window_size - 1, ". Best CRPS:", best_crps.round(5))
            break

    return best_vine, opt_window_size, best_crps.item()