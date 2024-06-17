import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_loss(history, ax):
    
    ax.plot(history['train_loss'], label='train')
    ax.plot(history['val_loss'], label='val')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_title('loss', size=15)
    
def plot_acc(history, ax):
    ax.plot(history['train_acc'], label='train')
    ax.plot(history['val_acc'], label='val')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_title('acc', size=15)
    
    

def fix_seed(seed, deterministic=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def make_label_df(data_dict, make_sum=False):
    labels_df = pd.DataFrame({'label' : list(set(data_dict['train_labels']))})
    labels_df['name'] = data_dict['label_names']
    for key_labels in ['train_labels', 'test_labels', 'train_obs_labels', 'train_mis_labels']:
        labels_df[key_labels] = pd.Series(data_dict[key_labels]).value_counts().sort_index().values

    for key_labels in ['train_obs_labels', 'train_mis_labels']:
        labels_df[f'{key_labels}_ratio'] = labels_df[[key_labels]].apply(lambda x: x/x.sum()).values

    labels_df.sort_values('train_obs_labels', ascending=False)
    if make_sum:
        labels_df.loc['sum', :] = labels_df.sum()
        labels_df.loc['sum', ['label', 'name']] = '-'
    return labels_df

def add_label_df(label_df, data_dict, dataset_obs_train, dataset_obs_val):
    label_df['dataset_obs_train'] = pd.Series(data_dict['train_obs_labels'][dataset_obs_train.indices]).value_counts().sort_index().values
    label_df['dataset_obs_val'] = pd.Series(data_dict['train_obs_labels'][dataset_obs_val.indices]).value_counts().sort_index().values
    for key_labels in ['dataset_obs_train', 'dataset_obs_val']:
        label_df[f'{key_labels}_ratio'] = label_df[[key_labels]].apply(lambda x: x/x.sum())
        
def add_sum(df):
    df = df.copy()
    df.loc['sum', :] = df.sum()
    df.loc['sum', 'name'] = '-'
    df.round(2)
    return df

def plot_model_result(y_mis_pred, dataset_mis):
    result_df = pd.DataFrame([np.argmax(y_mis_pred, axis=1), dataset_mis.y], index=['pred', 'true']).T
    vcs = []
    for col in result_df.columns:
        vc = result_df[col].value_counts()
        vcs.append(vc)

    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    result_df['correct'] = result_df['pred'] == result_df['true']
    class2acc = result_df.groupby('true')['correct'].agg(('sum', 'count')).apply(lambda x: x['sum']/x['count'], axis=1)

    class2acc.plot.bar(color='navy', ax=axes[0])
    axes[0].set_title('Accuracy by class(mis)', size=15, weight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticks(range(2), range(2), rotation=0)
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
    
def softmax(x, axis=1):
        return(np.exp(x)/(np.exp(x).sum(axis=axis, keepdims=True)))