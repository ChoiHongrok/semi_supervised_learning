import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import OrderedDict
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import statsmodels.formula.api as smf
import statsmodels.api as sm
from utils.util import *
from utils.dataset import *

import numpy as np

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
            
            
def train(model, n_epoch, train_set, val_set,
          device, optimizer, scheduler, criterion, history,
          es_patience=600, every_epoch=1, tau=1):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)  
    es = EarlyStopping(patience=es_patience, mode='max', verbose=False)
    
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        ## Train
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(X) / tau
            pred = torch.argmax(out, dim=-1)
            
            acc = sum(pred == y)
            loss = criterion(out, y)
            train_loss += loss.item()
            train_acc += acc.item()
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_set)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                out = model(X) / tau
                pred = torch.argmax(out, dim=-1)
                
                acc = sum(pred == y)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_acc += acc.item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_set)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if epoch % every_epoch == 0 or epoch+1 == n_epoch:
            ## Validation
            print(f'Epoch:{epoch+1:04d}/{n_epoch:04d}, train loss: {train_loss:4.4}, train acc: {train_acc:4.4}, val loss: {val_loss:4.4}, val acc: {val_acc:4.4}')
        if train_acc > 0.999:
            print(f'Epoch:{epoch+1:04d}/{n_epoch:04d}, train loss: {train_loss:4.4}, train acc: {train_acc:4.4}, val loss: {val_loss:4.4}, val acc: {val_acc:4.4}')
            break
        
        es(val_acc)
        if es.early_stop:
            break
        

def evaluate(model, test_set, device, criterion, tau, take_out=False, take_residual=False):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    preds = []
    outs = []
    resid_lst = []
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X) / tau
            pred = torch.argmax(out, dim=-1)
            preds.extend([p.item() for p in pred])
            
            acc = sum(pred == y)
            loss = criterion(out, y)
            test_loss += loss.item()
            test_acc += acc.item()
            if take_out:
                outs.extend(out.cpu())
                
            if take_residual:
                y_onehot = F.one_hot(y, num_classes=model.num_classes)
                out = torch.nn.Softmax(dim=-1)(out)
                resid = (y_onehot-out).cpu()
                resid_lst.extend(resid)
            
                
    test_loss /= len(test_loader)
    test_acc /= len(test_set)
    
    if take_out:
        return torch.stack(outs)
    
    if take_residual:
        return torch.stack(resid_lst) 
    
    result = [preds, test_set.y]
    f1 = f1_score(test_set.y, preds, average='macro')
    return test_loss, test_acc, f1, result


class sm_LogisticRegression:
    def __init__(self):
        pass
    
    def fit(self, X, y, w):
        self.model = sm.GLM(endog=y, 
                            exog=sm.add_constant(X, has_constant='add'), 
                            family=sm.families.Binomial(),
                            var_weights=w).fit()
        self.intercept_ = self.model.params[:1]
        self.coef_ = [self.model.params[1:]]
        
    def predict_proba(self, X):
        pred = self.model.predict(sm.add_constant(X, has_constant='add'))[:, np.newaxis]
        return np.concatenate([1-pred, pred], axis=1)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class ResponseModel:
    def __init__(self, dataset_obs, model, device, criterion, num_classes, tau, use_pool=True):
        self.dataset_obs = dataset_obs
        self.model = model
        self.device = device
        self.criterion = criterion
        self.converge = False
        self.use_pool = use_pool
        self.tau = tau
        labels, freqs = np.unique(dataset_obs.y, return_counts=True)
        self.label2freq = {label : freq/sum(freqs) for label, freq in zip(labels, freqs)}
        self.feature_extractor = torch.nn.Sequential(
            OrderedDict(
                list(self.model.model.named_children())[:-1]
                )
            )
        # self.feature_extractor = self.model
        self.X_obs = self._preprocess_X(dataset_obs)
        self.y_obs = F.one_hot(torch.tensor(self.dataset_obs.y), num_classes=num_classes).numpy() # (n, num_classes)
        
        # self.X_obs = np.tile(self.X_obs, (10, 1))
        # self.y_obs = np.tile(self.y_obs, (10, 1))
        self.num_classes = num_classes
        
    def _maxpool_x(self, dataset):
        return torch.stack([nn.MaxPool2d(kernel_size=16)(dataset[i][0]).flatten() for i in range(len(dataset))])  
    
    def _extract_feature(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
        self.feature_extractor.eval()
        feats = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.cuda()
                feat = self.feature_extractor(X)
                feats.append(feat.cpu().numpy())
        return np.concatenate(feats)
        
    def _softmax(self, x, axis=1):
        return(np.exp(x)/(np.exp(x).sum(axis=axis, keepdims=True)))
    
    def _preprocess_X(self, dataset):
        return np.zeros((len(dataset), 1)) ### X없애기
        if self.use_pool:
            X_mis = self._maxpool_x(dataset).numpy() # [m, 12]
        else:
            X_mis = self._extract_feature(dataset) # [m, 512]
        return X_mis
    
        
    def iterate(self, n_iter, dataset_mis, epsilon=1e-5, verbose=True, sampling=False):
        n = len(self.dataset_obs) 
        m = len(dataset_mis)
        random_idx = range(m)
        # # sampling
        if sampling:
            random_idx = np.random.choice(range(m), n, replace=False)
        else:
            n = m
        K = self.num_classes
        response_model_params_prev = None
        X_mis = self._preprocess_X(dataset_mis)[random_idx]
        X_total = np.concatenate([self.X_obs, *[X_mis for _ in range(K)]], axis=0)
        y_mis_pred_init = evaluate(self.model, dataset_mis, self.device, self.criterion, tau=self.tau, take_out=True).numpy() # (m, num_classes)
        self.out = y_mis_pred_init.copy()
        y_mis_pred_init = self._softmax(y_mis_pred_init)[random_idx]
        # y_mis_pred_init = np.log1p(y_mis_pred_init)
        self.X_mis = X_mis
        self.y_mis_pred_init = y_mis_pred_init #확인용
        y_dummy = np.eye(self.num_classes)
        
        w_pred = y_mis_pred_init
        y_total = np.concatenate([self.y_obs, 
                                  *[np.tile(y_dummy[k], (n, 1)) for k in range(K)]], axis=0)
        response = np.concatenate([np.ones_like(self.y_obs[:, 0]),
                                   *[np.zeros_like(y_mis_pred_init[:, 0]) for _ in range(K)]], axis=0)
        
        # dummy
        nonres_odds = np.empty_like(w_pred)
        y_mis_dummy = np.array([np.zeros_like(w_pred) for _ in range(K)]) # (K, m, K)
        for k in range(K):
            y_mis_dummy[k][:, k] = 1
        Xy_mis_dummy = [np.concatenate([X_mis, y_mis_dummy[k]], axis=1) for k in range(K)]
        self.Xy_mis_dummy = Xy_mis_dummy #확인용
        
        mis_acc = sum(dataset_mis.y[random_idx] == np.argmax(w_pred, axis=-1))/n
        self.response_model_params = []
        if verbose:
            print(f'iter({0})>> diff: {None}, lr_score: {None}, missing_acc: {mis_acc:2.4f}')

        Xy_total = np.concatenate([X_total, y_total], axis=1)

        for i in range(n_iter):
            self.w_pred = w_pred #확인용
            w_total = np.concatenate([np.ones_like(self.y_obs[:, 1]), 
                            *[w_pred[:, k] for k in range(K)]], axis=0)
            self.Xy_total = Xy_total #확인용
            self.response = response #확인용
            self.w_total = w_total #확인용
            # train response model
            self.response_model = LogisticRegression(max_iter=10000, 
                                                     fit_intercept=False,
                                                    #  penalty=None,
                                                    #  class_weight='balanced'
                                                     )
            # self.response_model = sm_LogisticRegression()
               
            self.response_model.fit(Xy_total, response, 
                                    w_total
                                    )
            # self.response_model.intercept_ = [0]
            # self.response_model.coef_[0] = np.array([0, -2, -4])
            response_model_params_curr = np.concatenate([self.response_model.intercept_, self.response_model.coef_[0]]) 
            self.response_model_params_curr = response_model_params_curr
            self.response_model_params.append(response_model_params_curr)
            self.res_prob_ks = []
            for k in range(self.num_classes):
                
                res_prob_k = self.response_model.predict_proba(Xy_mis_dummy[k])[:, 1]
                self.res_prob_k = res_prob_k #확인용
                self.res_prob_ks.append(res_prob_k.mean())
                if verbose:
                    print(len(response), response.sum(), self.res_prob_k.round().sum(), self.res_prob_k.mean())
                res_prob_k[res_prob_k < 1e-8] = 1e-8
                nonres_odds_k = (1-res_prob_k) / res_prob_k
                nonres_odds[:, k] = nonres_odds_k
            print('odds: ', nonres_odds[0])
            self.nonres_odds = nonres_odds
            w_pred = y_mis_pred_init * nonres_odds # (m, K)
            
            w_pred /= w_pred.sum(axis=1, keepdims=1)

            # metric
            lr_score = self.response_model.score(Xy_total, response)
            mis_acc = sum(dataset_mis.y[random_idx] == np.argmax(w_pred, axis=-1))/n
            if type(response_model_params_prev) != type(None):
                diff = ((response_model_params_prev - response_model_params_curr) ** 2).sum()
                if verbose:
                    print(f'iter({i+1})>> diff: {diff:4.5f}, lr_score: {lr_score:2.5f}, missing_acc: {mis_acc:2.4f}')
                if diff < epsilon:
                    if verbose:
                        print('break: convergence')
                    self.converge = True
                    break
                
            else:
                if verbose:
                    print(f'iter({i+1})>> diff: {None}, lr_score: {lr_score:2.5f}, missing_acc: {mis_acc:2.4f}')
            
            response_model_params_prev = response_model_params_curr
        
    
    def predict_proba(self, dataset_mis):
        X_mis = self._preprocess_X(dataset_mis)
        y_mis_pred_init = evaluate(self.model, dataset_mis, self.device, self.criterion, tau=self.tau, take_out=True).numpy()
        y_mis_pred_init = self._softmax(y_mis_pred_init)
        self.y_mis_pred_init = y_mis_pred_init
        nonres_odds = np.empty_like(y_mis_pred_init)
        for k in range(self.num_classes):
            y_mis_k = np.zeros_like(y_mis_pred_init)
            y_mis_k[:, k] = 1
            Xy_mis_k = np.concatenate([X_mis, y_mis_k], axis=1)
            res_prob_k = self.response_model.predict_proba(Xy_mis_k)[:, 1]
            nonres_odds_k = (1-res_prob_k) / res_prob_k
            nonres_odds[:, k] = nonres_odds_k
        self.nonres_odds = nonres_odds
        y_mis_pred = y_mis_pred_init * nonres_odds
        y_mis_pred /= y_mis_pred.sum(axis=1, keepdims=1)
            
        # Xy_mis_0 = np.concatenate([X_mis, np.zeros_like(y_mis_pred_init)[:, :1]], axis=1)
        # Xy_mis_1 = np.concatenate([X_mis, np.ones_like(y_mis_pred_init)[:, 1:]], axis=1)
        
        # res_prob_0 = self.response_model.predict_proba(Xy_mis_0)[:, 1:]
        # res_prob_1 = self.response_model.predict_proba(Xy_mis_1)[:, 1:]
        # nonres_odds_0 = (1-res_prob_0) / res_prob_0
        # nonres_odds_1 = (1-res_prob_1) / res_prob_1
        # w0 = y_mis_pred_init[:, :1] * nonres_odds_0
        # w1 = y_mis_pred_init[:, 1:] * nonres_odds_1
        # w0 = w0 / (w0+w1)
        # w1 = 1 - w0
        # y_mis_pred = np.concatenate([w0, w1], axis=1) 
        
        return y_mis_pred
    
    def predict(self, dataset_mis):
        prob = self.predict_proba(dataset_mis)
        return np.argmax(prob, axis=1)
        
    def evaluate(self, dataset_mis, y_true, plot=False):
        y_mis_pred = self.predict(dataset_mis)
        if plot:
            self.plot_model_result(y_mis_pred, dataset_mis)
        acc = np.mean(y_mis_pred == y_true)
        f1 = f1_score(y_true, y_mis_pred, average='macro')
        return acc, f1 
    
    def make_class2acc(self, dataset_mis):
        y_mis_pred = self.predict(dataset_mis)
        result_df = pd.DataFrame([y_mis_pred, dataset_mis.y], index=['pred', 'true']).T
        result_df['correct'] = result_df['pred'] == result_df['true']
        class2acc = result_df.groupby('true')['correct'].agg(('sum', 'count')).apply(lambda x: x['sum']/x['count'], axis=1)
        return class2acc
    
    def plot_model_result(self, y_mis_pred, dataset_mis):
        result_df = pd.DataFrame([y_mis_pred, dataset_mis.y], index=['pred', 'true']).T
        # vcs = []
        # for col in result_df.columns:
        #     vc = result_df[col].value_counts()
        #     vcs.append(vc)

        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        result_df['correct'] = result_df['pred'] == result_df['true']
        class2acc = result_df.groupby('true')['correct'].agg(('sum', 'count')).apply(lambda x: x['sum']/x['count'], axis=1)

        class2acc.plot.bar(color='navy', ax=axes[0])
        axes[0].set_title('Accuracy by class(mis)', size=15, weight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticks(range(self.num_classes), range(self.num_classes), rotation=0)
        axes[0].set_xlabel('Class', size=12)
        
        cm = confusion_matrix(result_df['true'], result_df['pred'])
        sns.heatmap(cm,
                vmin=0, vmax=cm.max().max(), center=0,
                cmap='coolwarm',
                annot=True, fmt='d',
                linewidth=0.1, square=True, cbar=False,
                ax=axes[1]
                )
        axes[1].set_xlabel('Label (Prediction)', size=12)
        axes[1].set_ylabel('Label (True)', size=12)
        axes[1].set_title('Outcome model prediction for Missing', size=15)
        
        plt.show()


def compare_raw_adjust(dataset_obs, dataset_mis, n_class, 
                       model, device, criterion, tau,
                       return_preds = False, return_odds=False):
    
    label, freq_obs = np.unique(dataset_obs.y, return_counts=True)
    label, freq_mis = np.unique(dataset_mis.y, return_counts=True)

    res_prob = freq_obs/(freq_obs+freq_mis)
    nonres_odds = (1-res_prob) / res_prob
    if return_odds:
        return nonres_odds
    print('res_prob: ', res_prob)
    print('nonres_odds: ', nonres_odds)
    pred = evaluate(model, dataset_mis, device=device, 
                    criterion=criterion, tau=tau, take_out=True).numpy()
    pred = softmax(pred)
    # pred = pred/pred.sum(axis=1, keepdims=True)
    pred_odds = pred * nonres_odds # broadcast
    pred_adjust = pred_odds.argmax(axis=1)
    
    acc = np.mean(pred.argmax(axis=1) == dataset_mis.y) 
    f1 = f1_score(dataset_mis.y, pred.argmax(axis=1), average='macro')
    adjust_acc = np.mean(pred_adjust == dataset_mis.y)
    adjust_f1 = f1_score(dataset_mis.y, pred_adjust, average='macro')
    if return_preds:
        return acc, adjust_acc, f1, adjust_f1, pred, pred_adjust
    
    # m = len(dataset_mis)
    # labels, freqs = np.unique(dataset_obs.y, return_counts=True)
    # label2freq = {label : freq/sum(freqs) for label, freq in zip(labels, freqs)}
    # test_pred = evaluate(model, dataset_mis, device=device, 
    #                      criterion=criterion, take_out=True).numpy()
    # probs = np.tile([label2freq[i] for i in range(n_class)], (len(dataset_mis), 1))
    # nonres_odds = (1-probs) / probs
    # test_pred_odds = test_pred * nonres_odds
    # test_pred_adjust = test_pred_odds.argmax(axis=1)
    
    # test_acc = np.mean(test_pred.argmax(axis=1) == dataset_mis.y)
    # test_adjust_acc = np.mean(test_pred_adjust == dataset_mis.y)
    # if return_preds:
    #     return test_acc, test_adjust_acc, test_pred, test_pred_adjust
        
    return acc, adjust_acc, f1, adjust_f1, res_prob


class SSL:
    def __init__(self, outcome_model, response_model, ssl_model, 
                 dataset_obs_train, subset_obs_val_norm, dataset_mis_none, dataset_mis, tau):
        self.outcome_model = outcome_model
        self.response_model = response_model
        self.ssl_model = ssl_model
        self.X_obs = np.array([X for X, y in dataset_obs_train])
        self.y_obs = np.array([y for X, y in dataset_obs_train])
        self.X_mis = np.array([X for X, y in dataset_mis_none])
        self.y_mis_pred = response_model.predict_proba(dataset_mis)
        
        mask = self.y_mis_pred.max(axis=1) > tau
        self.y_mis_pred = self.y_mis_pred.argmax(axis=1)
        self.X_total = np.concatenate([self.X_obs, self.X_mis[mask]])
        self.y_total = np.concatenate([self.y_obs, self.y_mis_pred[mask]])
        
        self.trainset = Dataset_from_norm_Xy(self.X_total, self.y_total, use_transform=True)
        self.valset = subset_obs_val_norm


    def fit(self, n_epoch,
          device, optimizer, scheduler, criterion, history, every_epoch=1):
        
        train(self.ssl_model, n_epoch=n_epoch, 
              train_set=self.trainset,
              val_set=self.valset,
              device=device,
              optimizer = optimizer,
              scheduler = scheduler,
              criterion = criterion,
              history = history,
              es_patience = 50,
              every_epoch=50
              )