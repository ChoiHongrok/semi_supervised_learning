import numpy as np
import pandas as pd


def MSE(preds, targets, mode):
    if mode == 'all':
        return np.mean((preds - targets)**2)
    else:
        return np.mean((preds - targets)**2, axis=0)
        
def RMSE(preds, targets, mode):
    return np.sqrt(MSE(preds, targets, mode))

def ACC(preds, targets, mode):
    preds = np.maximum(preds.round(), 0)
    if mode == 'all':
        return np.mean(preds==targets)
    else:
        return np.mean(preds==targets, axis=0)
        

def REL_ERROR(preds, targets, mode):
    if mode == 'all':
        return np.mean(np.abs((targets - preds) / targets))
    else:
        return np.mean(np.abs((targets - preds) / targets), axis=0)
        
    
def record_loss(losses, X_org, X_imp, cat_idx, num_idx, col_names,  
                miss_algo, data_name, model_name, seed=None):
    # for mode in ['all', 'column']:
    #     rmse = RMSE(X_imp[:, num_idx], X_org[:, num_idx], mode)
    #     acc = ACC(X_imp[:, cat_idx], X_org[:, cat_idx], mode)
    #     rel = REL_ERROR(X_imp[:, cat_idx], X_org[:, cat_idx], mode)
        
    #     common_dict = losses.setdefault(mode, {})
    #     common_dict.setdefault('RMSE', {}).setdefault(miss_algo, {}).setdefault(data_name, {})[model_name] = rmse
    #     common_dict.setdefault('ACC', {}).setdefault(miss_algo, {}).setdefault(data_name, {})[model_name] = acc
    #     common_dict.setdefault('REL', {}).setdefault(miss_algo, {}).setdefault(data_name, {})[model_name] = rel
    
    for mode in ['all', 'column']:
        rmse = RMSE(X_imp[:, num_idx], X_org[:, num_idx], mode)
        acc = ACC(X_imp[:, cat_idx], X_org[:, cat_idx], mode)
        rel = REL_ERROR(X_imp[:, num_idx], X_org[:, num_idx], mode)
        
        if mode == 'column':
            for cat_i, i in enumerate(cat_idx):
                losses.append([col_names[i], miss_algo, data_name, model_name, seed, 'ACC', acc[cat_i]])
            for num_i, i in enumerate(num_idx):
                losses.append([col_names[i], miss_algo, data_name, model_name, seed, 'RMSE', rmse[num_i]])
                losses.append([col_names[i], miss_algo, data_name, model_name, seed, 'REL', rel[num_i]])
        else:
            losses.append([mode, miss_algo, data_name, model_name, seed, 'ACC', acc])
            losses.append([mode, miss_algo, data_name, model_name, seed, 'RMSE', rmse])
            losses.append([mode, miss_algo, data_name, model_name, seed, 'REL', rel])
            