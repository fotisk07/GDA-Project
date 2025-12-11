import numpy as np
import torch
import time
import psutil, os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def gpu_memory_peak_mb(device):
    """Peak GPU memory usage in MB."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0

def gpu_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def rbf_kernel_gpu(Xb, C, sigma=1.0):
    X2 = (Xb**2).sum(dim=1, keepdim=True)
    C2 = (C**2).sum(dim=1).unsqueeze(0)
    K = X2 + C2 - 2 * Xb @ C.T
    return torch.exp(-K / (2*sigma**2))

def knm_full_mv_gpu(X, C, v, device, sigma=1.0):
    """
    Computes K(X, C)v in a single GPU operation.
    """
    if device.type == "cpu":
        raise EnvironmentError("GPU Full Kernel requires CUDA device.")

    # --- Transfer CPU → GPU ---
    t0_transfer = time.time()
    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    C_gpu = torch.tensor(C, dtype=torch.float32, device=device)
    v_gpu = torch.tensor(v, dtype=torch.float32, device=device)
    gpu_sync()
    t_transfer = time.time() - t0_transfer

    t0_compute = time.time()
    with torch.no_grad():
        # 1. the full K(n, m) matrix
        K_full = rbf_kernel_gpu(X_gpu, C_gpu, sigma=sigma)
        # 2. Matrix-vector product
        res_gpu = K_full @ v_gpu
    gpu_sync()
    t_compute = time.time() - t0_compute

    # --- GPU → CPU ---
    res = res_gpu.cpu().numpy()

    del X_gpu, C_gpu, v_gpu, K_full, res_gpu
    torch.cuda.empty_cache()
    return res, {
        "transfer_time": t_transfer,
        "compute_time": t_compute,
        "total_time": t_transfer + t_compute
    }

def knm_block_mv_gpu(X, C, v, device, q=2000, sigma=1.0):
    """
    GPU ops for kernel(X,C) v in chunks of size q.
    """
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float32)
    C_gpu = torch.tensor(C, dtype=torch.float32, device=device)
    v_gpu = torch.tensor(v, dtype=torch.float32, device=device)
    total_transfer = 0.
    total_compute  = 0.
    total_batches  = 0
    with torch.no_grad():
        for i in range(0, n, q):
            Xb = X[i:i+q]

            # --- TRANSFER CPU → GPU ---
            t_transfer = time.time()
            Xb_gpu = torch.tensor(Xb, dtype=torch.float32, device=device)
            gpu_sync()
            t_transfer = time.time() - t_transfer

            # --- GPU COMPUTE ---
            t_compute = time.time()
            Kb = rbf_kernel_gpu(Xb_gpu, C_gpu, sigma=sigma) # Compute the kernel matrix batch 
            res_gpu = Kb @ v_gpu # compute matrix vector product 
            gpu_sync()
            t_compute = time.time() - t_compute

            # --- GPU → CPU ---
            res = res_gpu.cpu().numpy()
            out[i:i+q] = res # Aggregate the output 

            del Xb_gpu, Kb, res_gpu
            torch.cuda.empty_cache()

            total_transfer += t_transfer
            total_compute  += t_compute
            total_batches  += 1

    del C_gpu, v_gpu
    torch.cuda.empty_cache()
    return out, {
        "avg_transfer": total_transfer / total_batches,
        "avg_compute":  total_compute  / total_batches,
        "total_time":   total_transfer + total_compute
    }


def knm_block_mv_gpu_qr(X, C, v, device, q=2000, r=1500, sigma=1.0):
    """
    Product K(X, C) v with double batching :
    - batching with q over n (rows de X)
    - batching with r over m (centers C)

    X : (n, d)  np.float32
    C : (m, d)  np.float32
    v : (m,)    np.float32
    """
    n, d = X.shape
    m    = C.shape[0]
    out  = np.zeros(n, dtype=np.float32)

    total_transfer = 0.0
    total_compute  = 0.0
    n_transfers    = 0
    n_computes     = 0

    with torch.no_grad():
        for i in range(0, n, q):
            Xb = X[i:i+q]                       
            q_i = Xb.shape[0]

            # ---- TRANSFER Xb → GPU (once per block q) ----
            t_tr = time.time()
            Xb_gpu = torch.tensor(Xb, dtype=torch.float32, device=device)
            gpu_sync()
            t_tr = time.time() - t_tr

            total_transfer += t_tr
            n_transfers    += 1

            # accumulation over the block of q_i rows
            out_block = torch.zeros(q_i, dtype=torch.float32, device=device)

            # ---- loop over r batches ----
            for j in range(0, m, r):
                Cbj = C[j:j+r]                  # (r_j, d)
                vj  = v[j:j+r]                  # (r_j,)

                # Transfer Cbj, vj -> GPU
                t_tr2 = time.time()
                C_gpu = torch.tensor(Cbj, dtype=torch.float32, device=device)
                v_gpu = torch.tensor(vj,  dtype=torch.float32, device=device)
                gpu_sync()
                t_tr2 = time.time() - t_tr2

                total_transfer += t_tr2
                n_transfers    += 1

                # ---- COMPUTE over block (q_i × r_j) ----
                t_comp = time.time()
                Kb     = rbf_kernel_gpu(Xb_gpu, C_gpu, sigma=sigma)   # (q_i, r_j)
                res_gpu = Kb @ v_gpu                                  # (q_i,)
                out_block += res_gpu
                gpu_sync()
                t_comp = time.time() - t_comp

                total_compute += t_comp
                n_computes    += 1


                del C_gpu, v_gpu, Kb, res_gpu
                torch.cuda.empty_cache()

            # GPU -> CPU
            out[i:i+q_i] = out_block.cpu().numpy()

            del Xb_gpu, out_block
            torch.cuda.empty_cache()

    stats = {
        "avg_transfer": total_transfer / max(n_transfers, 1),
        "avg_compute":  total_compute  / max(n_computes, 1),
        "total_time":   total_transfer + total_compute,
    }
    return out, stats


def upscale_heatmap(Z, q_list, r_list, factor=4, method='cubic'):
    """
    Upscale heatmap matrix Z by interpolating on a denser (q,r) grid.

    Parameters:
    - Z : (len(q_list) × len(r_list)) matrix
    - q_list : original q grid (1D array)
    - r_list : original r grid (1D array)
    - factor : upscale factor (4 → 16× more points)
    - method : 'linear', 'cubic', or 'nearest'
    """
    q = np.array(q_list)
    r = np.array(r_list)

    Q, R = np.meshgrid(q, r, indexing='ij')  # shape = (nq, nr)
    points = np.vstack([Q.ravel(), R.ravel()]).T
    values = Z.ravel()
    q_new = np.logspace(np.log10(q.min()), np.log10(q.max()), len(q)*factor)
    r_new = np.logspace(np.log10(r.min()), np.log10(r.max()), len(r)*factor)

    Q_new, R_new = np.meshgrid(q_new, r_new, indexing='ij')

    # Interpolate
    Z_new = griddata(points, values, (Q_new, R_new), method=method)

    return Z_new, q_new, r_new

