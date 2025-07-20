import numpy as np
from scipy.stats import pearsonr

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NSE(pred, true):
    return 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

def KGE(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    r, _ = pearsonr(pred_flat, true_flat)
    α = np.std(pred_flat) / np.std(true_flat)
    β = np.mean(pred_flat) / np.mean(true_flat)
    return 1 - np.sqrt((r - 1) ** 2 + (α - 1) ** 2 + (β - 1) ** 2)
    
def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    nse = NSE(pred, true)
    kge = KGE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, nse, kge, r2
