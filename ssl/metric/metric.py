import numpy as np
import pandas as pd


def MAE(preds, targets):
    return np.mean(np.abs(preds - targets))

def MSE(preds, targets):
    return np.mean((preds - targets)**2)

def RMSE(preds, targets):
    return np.sqrt(MSE(preds, targets))

def ACC(preds, targets):
    assert len(targets.shape) == 1 and len(preds.shape)< 3
    if len(preds.shape) == 2 and preds.shape[1]!=1:
        preds = preds.argmax(axis=1)
    preds = np.maximum(preds.round(), 0)
    return np.mean(preds==targets)

from sklearn.metrics import roc_auc_score
def ROC_AUC(preds_prob, targets):
    assert len(targets.shape) == 1 and len(preds_prob.shape) < 3 
    if len(preds_prob.shape) == 2 :
        preds_prob = preds_prob[:, 1]
    return roc_auc_score(targets, preds_prob)

def relative_error(true, pred):
    eps = 1e-8
    return (true-pred)/(true+eps)

def similar(preds, targets, mode='mean'):
    if mode == 'mean':
        func = np.mean
    elif mode == 'std':
        func = np.std
    if len(preds.shape) == 2 and preds.shape[1]!=1: 
        preds = preds.argmax(axis=1)
    stat_t = func(targets)
    stat_p = func(preds)
    return relative_error(stat_t, stat_p)

def make_cnt_vec(key, cnt, size):
    arr = np.empty(int(size)+1)
    for k, c in zip(key, cnt):
        arr[int(k)] = c
    return arr

def MLE(preds, targets):
    if len(preds.shape) == 2:
        if preds.shape[1]==1:
            preds.squeeze()
        else:
            preds = preds.argmax(axis=1)
    preds = np.maximum(preds.round(), 0)
    # make p's size dim vector
    target_u, target_cnt = np.unique(targets, return_counts=True)
    target_p = make_cnt_vec(target_u, target_cnt, max(target_u))
    pred_u, pred_cnt = np.unique(preds, return_counts=True)
    pred_p = make_cnt_vec(pred_u, pred_cnt, max(pred_u))
    # unify vector size
    len_t = len(target_p)
    len_p = len(pred_p)
    len_diff = abs(len_t-len_p)
    if len_t > len_p:
        pred_p = np.pad(pred_p, (0, len_diff))
    else:
        target_p = np.pad(target_p, (0, len_diff))
    target_mle = target_p / sum(target_p)
    pred_mle = pred_p / sum(pred_p)
    return np.mean(np.abs(relative_error(target_mle, pred_mle)))
    
from sklearn.metrics import r2_score
def r2(preds, targets):
    if len(preds.shape) == 2:
        if preds.shape[1]==1:
            preds.squeeze()
        else:
            preds = preds.argmax(axis=1)
    return r2_score(targets, preds, force_finite=True)


# import torch
# from torchmetrics.functional.nominal import theils_u
# def theil(preds, targets):
#     if len(preds.shape) == 2:
#         if preds.shape[1]==1:
#             preds.squeeze()
#         else:
#             preds = preds.argmax(axis=1)
#     preds = max(0, preds.round())
#     print(torch.tensor(preds).shape)
#     print(torch.tensor(targets).shape)
#     return theils_u(torch.tensor(preds), torch.tensor(targets)).item()


def corr(preds, targets):
    if len(preds.shape) > 1:
        preds = preds.squeeze()
    return np.corrcoef(preds, targets)[0,1]


from sklearn.metrics import f1_score as F1
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
def fit_predict(model, X_train, y_train, X_test):
    model = model()
    if len(X_train.shape) == 1:
        X_train = X_train[:, np.newaxis]
        X_test = X_test[:, np.newaxis]
    model.fit(X_train, y_train)
    return model.predict(X_test)
    
def relational_sim(X_true, X_pred, y, model='DTC'):
    if len(X_pred.shape) > 1:
        if X_pred.shape[1] == 1:
            X_pred = X_pred.squeeze()
        else:
            X_pred = X_pred.argmax(axis=1)
            
    if model == 'DTC':
        model = DTC
    ind = np.arange(len(X_true))
    ind_train, ind_test = train_test_split(ind, test_size=0.3)
    y_true = fit_predict(model, X_true[ind_train], y[ind_train], X_true[ind_test])
    y_pred = fit_predict(model, X_pred[ind_train], y[ind_train], X_pred[ind_test])
    f_true = F1(y_true, y[ind_test])
    f_pred = F1(y_pred, y[ind_test])
    # print(f_true, f_pred)
    return relative_error(f_true, f_pred)


def evaluate(preds, targets, y, mode):
    # shape = (10, 1)
    # preds = np.random.rand(*shape)
    # y = np.random.binomial(1, 0.3, shape[0])
    # y = preds.round()

    score_dict = {}
    shape = preds.shape
    #numerical
    score_dict['mae'] = MAE(preds, targets)
    score_dict['mse'] = MSE(preds, targets)
    score_dict['r2'] = r2(preds, targets)
    if mode == "num":
        score_dict['mean_sim'] = similar(preds, targets, mode='mean')
        score_dict['std_sim'] = similar(preds, targets, mode='std')
        score_dict['corr'] = corr(preds, targets)
    #categorical
    elif mode == "cat":
        score_dict['acc'] = ACC(preds, targets)
        # if shape[1] == 2:
        #     score_dict['auc'] = ROC_AUC(preds, targets)
        score_dict['mle_rel'] = MLE(preds, targets)
        # score_dict['theil_rel'] = theil(preds, targets)
    
    # # F
    # score_dict['F1_sim'] = relational_sim(targets, preds, y, model='DTC')
    return score_dict


def evaluate_X_each(X_origin, X_imp, y, mask, mode, index=['x1', 'x2', 'x3']):
    df_pre = []
    for i in range(X_origin.shape[1]):
        if mask[:, i].sum()==0:
            df_pre.append({})
            continue
        preds = X_imp[:, i][mask[:, i]]
        targets = X_origin[:, i][mask[:, i]]
        scores = evaluate(preds, targets, y, mode=mode)
        df_pre.append(scores)
    df = pd.DataFrame(df_pre, 
                index= index
                )
    return df


def evaluate_X(X_origin, X_imps, y, mask, mode, names):
    df_pre = []
    for X_imp in X_imps:
        preds = X_imp[mask]
        targets = X_origin[mask]
        scores = evaluate(preds, targets, y, mode=mode)
        df_pre.append(scores)
    df = pd.DataFrame(df_pre, 
                      index=names
                      ).T
    return df


def make_data(n, seed=10):
    def sigmoid(X, beta):
        z = X@beta
        return np.exp(z) / (1 + np.exp(z))
    
    np.random.seed(seed)
    X1 = np.random.gamma(1, 1, n)
    X2 = 1 + 0.5 * X1 + np.random.normal(0, 1, n)
    X3 = 2 + 0.5 + X1 + 0.5*X2 + np.random.normal(0, 1, n)

    X_obs = np.concatenate([X1, X2, X3]).reshape(3, n).T

    r1 = np.random.binomial(1, 0.5, n)
    r2 = np.random.binomial(1, 0.3, n)
    r3 = np.random.binomial(1, 0.1, n)
    mask = np.concatenate([r1, r2, r3]).reshape(n, 3).astype(bool)
    X_mis = X_obs.copy()
    X_mis[mask] = np.nan 
    
    # make y
    beta = np.array([0.02, -0., 0.03])
    probs = sigmoid(X_obs, beta)
    y = probs.round().astype(int)
    
    return X_obs, X_mis, mask, y


def mask_inv(X, mask):
    X = X.copy()
    X[~mask] = np.nan
    return X

def make_dist_df(Xs, names, mask, index=['x1', 'x2', 'x3']):
    df_pre = []
    df_names = []
    Xs_masked_inv = {}
    for X, name in zip(Xs, names):
        masked_inv = mask_inv(X, mask)
        Xs_masked_inv[name] = masked_inv
        mean = np.nanmean(masked_inv, axis=0)
        std = np.nanstd(masked_inv, axis=0)
        df_pre.extend([mean, std])
        df_names.extend([f'{name}(mean)', f'{name}(std)'])
        
    df = pd.DataFrame(df_pre, 
                    index = df_names,
                    columns= index
                    ).T
    return df, Xs_masked_inv


# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import KNNImputer
# class Imputer:
#     def __init__(self, seed, M, cat_ids, num_ids):
#         self.cat_ids = cat_ids
#         self.num_ids = num_ids
#         self.models = []
#         for i in range(M):
#             model = IterativeImputer(random_state=seed+i, sample_posterior=True)
#             self.models.append(model)
        
#         self.cat_model = KNNImputer(n_neighbors=10)
    
#     def fit(self, X):
#         for model in self.models:
#             model.fit(X)
#         self.cat_model.fit(X)
        
#     def transform(self, X):
#         # make num features
#         self.rets = []
#         for model in self.models:
#             ret = model.transform(X)
#             self.rets.append(ret)
#         num_pred = np.mean(self.rets, axis=0)
        
#         # make cat features
#         cat_pred = self.cat_model.transform(X)
        
#         # merge cat, num features
#         pred = []
#         i, j = 0, 0
#         # print(len(self.cat_ids), len(self.num_ids))
#         while i < len(self.cat_ids) or j < len(self.num_ids):
#             print(i, j)
#             cat_i = self.cat_ids[i] if i < len(self.cat_ids) else -1
#             num_j = self.num_ids[j] if j < len(self.num_ids) else -1
#             print('>>',cat_i, num_j)
#             if cat_i > num_j and num_j != -1:
#                 pred.append(num_pred[:, [num_j]])
#                 j += 1
#             else:
#                 pred.append(cat_pred[:, [cat_i]])
#                 i += 1
#         # print(i,j)
#         return np.concatenate(pred, axis=1)
    
    
# class Imputer:
#     def __init__(self, seed, M, cat_ids, num_ids):
#         self.cat_ids = cat_ids
#         self.num_ids = num_ids
#         self.models = []
#         if num_ids:
#             for i in range(M):
#                 model = IterativeImputer(random_state=seed+i, sample_posterior=True)
#                 self.models.append(model)
#         if cat_ids:
#             self.cat_model = KNNImputer(n_neighbors=10)
    
#     def fit(self, X):
#         if self.num_ids:
#             for model in self.models:
#                 model.fit(X)
#         if self.cat_ids:
#             self.cat_model.fit(X)
        
#     def transform(self, X):
#         # make num features
#         if self.num_ids:
#             self.rets = []
#             for model in self.models:
#                 ret = model.transform(X)
#                 self.rets.append(ret)
#             num_pred = np.mean(self.rets, axis=0)
        
#         # make cat features
#         if self.cat_ids:
#             cat_pred = self.cat_model.transform(X)
        
#         # merge cat, num features
#         pred = []
#         i, j = 0, 0
#         while i < len(self.cat_ids) or j < len(self.num_ids):
#             # print(i, j)
#             cat_i = self.cat_ids[i] if i != len(self.cat_ids) else -1
#             num_j = self.num_ids[j] if j != len(self.num_ids) else -1
#             if cat_i > num_j and num_j != -1:
#                 pred.append(num_pred[:, [num_j]])
#                 j += 1
#             else:
#                 pred.append(cat_pred[:, [cat_i]])
#                 i += 1
#         # print(i,j)
#         return np.concatenate(pred, axis=1)
    
    
#         return num_pred