import torch
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

from utils.util import *
from utils.model import *
from utils.train import *
from utils.dataset import *

def main(N_max=20, gamma=20, model_seed=5, NUM_CLASSES=10, every_epoch=50, use_pool=True):
    print('========================================Start========================================')
    print(f'>>>>>>>>> model_seed: {model_seed}\n>>>>>>>>> N_max: {N_max}\n>>>>>>>>> gamma: {gamma}\n>>>>>>>>> num_classes: {NUM_CLASSES}') 
    ret = {}
    ret.setdefault('params', {}).setdefault('N_max', N_max)
    ret.setdefault('params', {}).setdefault('gamma', gamma)
    ret.setdefault('params', {}).setdefault('model_seed', model_seed)
    ret.setdefault('params', {}).setdefault('NUM_CLASSES', NUM_CLASSES)
    ret.setdefault('params', {}).setdefault('use_pool', use_pool)
    
    
    
    data_dict = load_cifar10()
    if NUM_CLASSES == 2:
        data_dict = cut_class(data_dict, remain = ['bird', 'cat'])
    # print(data_dict['label_names'])
    split_obs_mis(data_dict, 
                N_max = N_max,
                gamma = gamma,
                seed = 100)

    # 클래스별 분포 프린트 넣기
    class_dist = pd.Series(data_dict['train_obs_labels']).value_counts()
    print(class_dist)
    ret['class_dist'] = class_dist
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('========================================Outcome model========================================')
    fix_seed(model_seed, deterministic=True)

    # dataset
    dataset_obs = Dataset(data_dict, mode='train_obs', use_transform=False)
    dataset_mis = Dataset(data_dict, mode='train_mis', use_transform=True)
    dataset_obs_train, dataset_obs_val = torch.utils.data.random_split(dataset_obs, (0.7, 0.3))
    # dataset_mis_train, dataset_mis_val = torch.utils.data.random_split(dataset_mis, (0.7, 0.3))

    subset_obs_train_aug = Subset(dataset_obs_train, use_transform=True)
    subset_obs_train_norm = Subset(dataset_obs_train, use_transform=False)
    subset_obs_val_norm = Subset(dataset_obs_val, use_transform=False)

    # add_label_df(label_df, data_dict, dataset_obs_train, dataset_obs_val)

    # model param
    model = ResNet18_cifar10(device=device, pretrained=False, num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, #0.001 #0.001
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # result dict
    history = {'train_acc': [],
            'train_loss' : [],
            'val_acc' : [],
            'val_loss' : []}


    n_epoch = 600
    train(model, n_epoch = n_epoch, 
        train_set = subset_obs_train_aug,
        val_set = subset_obs_val_norm,
        device=device,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        history = history,
        every_epoch=every_epoch)
    
    
    ## evaluate train_obs
    dataset_obs = Dataset(data_dict, mode='train_obs', use_transform=True)
    obs_loss, obs_acc, obs_result = evaluate(model, dataset_obs, device=device, criterion=criterion)
    print(f'obs loss: {obs_loss}, obs acc: {obs_acc}')

    ## evaluate train_mis
    mis_loss, mis_acc, mis_result = evaluate(model, dataset_mis, device=device, criterion=criterion)
    print(f'mis loss: {mis_loss}, mis acc: {mis_acc}')

    # evaluate trainset
    dataset_train = Dataset(data_dict, mode='train', use_transform=True)
    train_loss, train_acc, train_result = evaluate(model, dataset_train, device=device, criterion=criterion)
    print(f'train loss: {train_loss}, train acc: {train_acc}')

    ## evaluate testset
    dataset_test = Dataset(data_dict, mode='test', use_transform=True)
    test_loss, test_acc, test_result = evaluate(model, dataset_test, device=device, criterion=criterion)
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    
    ret.setdefault('result', {}).setdefault('outcome', {})
    ret.setdefault('model', {}).setdefault('outcome', model)
    
    ret['result']['outcome']['obs_train_loss'] = history['train_loss'][-1]
    ret['result']['outcome']['obs_train_acc'] = history['train_acc'][-1]
    ret['result']['outcome']['obs_val_loss'] = history['val_loss'][-1]
    ret['result']['outcome']['obs_val_acc'] = history['val_acc'][-1]
    ret['result']['outcome']['obs_loss']= obs_loss
    ret['result']['outcome']['obs_acc'] = obs_acc
    ret['result']['outcome']['mis_loss']= mis_loss
    ret['result']['outcome']['mis_acc'] = mis_acc
    ret['result']['outcome']['train_loss'] = train_loss
    ret['result']['outcome']['train_acc']= train_acc
    ret['result']['outcome']['test_loss']= test_loss
    ret['result']['outcome']['test_acc'] = test_acc
    
    def make_class2acc(result):
        result_df = pd.DataFrame(result, index=['pred', 'true']).T
        result_df['correct'] = result_df['pred'] == result_df['true']
        class2acc = result_df.groupby('true')['correct'].agg(('sum', 'count')).apply(lambda x: x['sum']/x['count'], axis=1)
        return class2acc
    
    ret['result']['outcome']['mis_class_acc'] = make_class2acc(mis_result)
    ret['result']['outcome']['test_class_acc'] = make_class2acc(test_result)

    print()
    print('========================================Response model========================================')

    response_model = ResponseModel(subset_obs_train_norm, model, device, criterion, 
                                num_classes=NUM_CLASSES, use_pool=use_pool)
    response_model.iterate(n_iter = 50, dataset_mis=dataset_mis)
    obs_acc = response_model.evaluate(dataset_obs, dataset_obs.y, 
                                    #   plot=True
                                    )
    mis_acc = response_model.evaluate(dataset_mis, dataset_mis.y, 
                                    # plot=True
                                    )
    test_acc = response_model.evaluate(dataset_test, dataset_test.y, 
                                    #    plot=True
                                    )
    print(f'obs_acc: {obs_acc}')
    print(f'mis_acc: {mis_acc}')
    print(f'test_acc: {test_acc}')
    
    ret.setdefault('result', {}).setdefault('response', {})
    ret.setdefault('model', {}).setdefault('response', response_model)
    
    ret['result']['response']['obs_acc'] = obs_acc
    ret['result']['response']['mis_acc'] = mis_acc
    ret['result']['response']['test_acc'] = test_acc
    ret['result']['response']['converge'] = response_model.converge
    
    
    ret['result']['response']['mis_class_acc'] = response_model.make_class2acc(dataset_mis)
    ret['result']['response']['test_class_acc'] = response_model.make_class2acc(dataset_test)
    
    return ret


if __name__ == "__main__":
    seeds = [100, 2000, 400, 3313, 1111]
    N_max_gamma = [(40, 20), (100, 50), (200, 100), 
                (2000, 100), (2000, 10), (1000, 30), (1000, 10)]
    NUM_CLASSES = 10
    for N_max, gamma in tqdm(N_max_gamma):
        for seed in tqdm(seeds):
            ret = main(N_max=N_max, gamma=gamma, model_seed=seed, NUM_CLASSES=NUM_CLASSES, every_epoch=300)
            torch.save(ret, f'../results/ret_class{NUM_CLASSES}_{N_max}_{gamma}_{seed}.pt')