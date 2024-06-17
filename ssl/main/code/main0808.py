import torch
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from glob import glob

# from main import *
from utils.util import *
from utils.model import *
from utils.train import *
from utils.dataset import *

def main(model_seed, label_weight, alpha, tau, only_y =True, NUM_CLASSES = 2):
    print('========================================Start========================================')
    print(f'>>>>>>>>> model_seed: {model_seed}\n>>>>>>>>> label_weight: {label_weight}\n>>>>>>>>> num_classes: {NUM_CLASSES}') 
    print(f'tau: {tau}')
    result = {}
    result.setdefault('params', {}).setdefault('model_seed', model_seed)
    result.setdefault('params', {}).setdefault('label_weight', label_weight)
    result.setdefault('params', {}).setdefault('tau', tau)
    result.setdefault('params', {}).setdefault('NUM_CLASSES', NUM_CLASSES)
    result.setdefault('params', {}).setdefault('only_y', only_y)
    
    
    data_dict = load_cifar10()
    print(data_dict['label_names'])
    if NUM_CLASSES == 2:
        data_dict = cut_class(data_dict, remain = ['bird', 'deer'])

    split_obs_mis_with_lr(data_dict, label_weight=label_weight, seed=100)
    class_dist = pd.Series(data_dict['train_obs_labels']).value_counts().sort_index()
    print(class_dist)
    result['class_dist'] = class_dist
    print('========================================Outcome model========================================')
    # outcome model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    fix_seed(model_seed, deterministic=True)

    # dataset
    dataset_obs = Dataset(data_dict, mode='train_obs', use_transform=False)
    dataset_mis = Dataset(data_dict, mode='train_mis', use_transform=True)
    dataset_mis_none = Dataset(data_dict, mode='train_mis', use_transform=False)
    dataset_obs_train, dataset_obs_val = torch.utils.data.random_split(dataset_obs, (0.7, 0.3))
    # dataset_mis_train, dataset_mis_val = torch.utils.data.random_split(dataset_mis, (0.7, 0.3))

    subset_obs_train_aug = Subset(dataset_obs_train, use_transform=True)
    subset_obs_train_norm = Subset(dataset_obs_train, use_transform=False)
    subset_obs_val_norm = Subset(dataset_obs_val, use_transform=False)

    # model param
    # alpha=np.random.choice([0.05, 0.14, 0.3])
    result.setdefault('params', {}).setdefault('alpha', alpha)
    print(f'alpha: {alpha}')
    # model = ResNet18_cifar10(device=device, pretrained=False, num_classes=NUM_CLASSES)
    model = AlexNet_cifar10(num_classes=NUM_CLASSES)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, #0.001 #0.001
    #                             momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # result dict
    history = {'train_acc': [],
            'train_loss' : [],
            'val_acc' : [],
            'val_loss' : []}


    n_epoch = 800
    train(model, n_epoch = n_epoch, 
        train_set = subset_obs_train_aug,
        val_set = subset_obs_val_norm,
        device=device,
        tau = tau,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        history = history,
        es_patience=n_epoch,
        every_epoch=50)
    
    ## evaluate train_obs
    # dataset_obs = Dataset(data_dict, mode='train_obs', use_transform=True)
    obs_loss, obs_acc, obs_f1, obs_result = evaluate(model, subset_obs_train_norm, device=device, criterion=criterion, tau=tau)
    print(f'obs loss: {obs_loss}, obs_f1: {obs_f1}, obs acc: {obs_acc}')

    obs_val_loss, obs_val_acc, obs_val_f1, obs_val_result = evaluate(model, subset_obs_val_norm, device=device, criterion=criterion, tau=tau)
    print(f'obs_val loss: {obs_val_loss}, obs_val_f1: {obs_val_f1}, obs_val acc: {obs_val_acc}')

    ## evaluate train_mis
    mis_loss, mis_acc, mis_f1, mis_result = evaluate(model, dataset_mis, device=device, criterion=criterion, tau=tau)
    print(f'mis loss: {mis_loss}, mis_f1: {mis_f1}, mis acc: {mis_acc}')

    # evaluate trainset
    dataset_train = Dataset(data_dict, mode='train', use_transform=True)
    train_loss, train_acc, train_f1, train_result = evaluate(model, dataset_train, device=device, criterion=criterion, tau=tau)
    print(f'train loss: {train_loss}, train_f1: {train_f1}, train acc: {train_acc}')

    ## evaluate testset
    dataset_test = Dataset(data_dict, mode='test', use_transform=True)
    test_loss, test_acc, test_f1, test_result = evaluate(model, dataset_test, device=device, criterion=criterion, tau=tau)
    print(f'test loss: {test_loss}, test_f1: {test_f1}, test acc: {test_acc}')
    
    result.setdefault('result', {}).setdefault('outcome', {})
    result.setdefault('model', {}).setdefault('outcome', model)
    result['result']['outcome']['obs_train_loss'] = history['train_loss'][-1]
    result['result']['outcome']['obs_train_acc'] = history['train_acc'][-1]
    result['result']['outcome']['obs_val_loss'] = history['val_loss'][-1]
    result['result']['outcome']['obs_val_acc'] = history['val_acc'][-1]
    result['result']['outcome']['obs_loss']= obs_loss
    result['result']['outcome']['obs_acc'] = obs_acc
    result['result']['outcome']['obs_f1'] = obs_f1
    result['result']['outcome']['mis_loss']= mis_loss
    result['result']['outcome']['mis_acc'] = mis_acc
    result['result']['outcome']['mis_f1'] = mis_f1
    result['result']['outcome']['train_loss'] = train_loss
    result['result']['outcome']['train_acc']= train_acc
    result['result']['outcome']['train_f1']= train_f1
    result['result']['outcome']['test_loss']= test_loss
    result['result']['outcome']['test_acc'] = test_acc
    result['result']['outcome']['test_f1'] = test_f1
    
    def make_class2acc(result):
        result_df = pd.DataFrame(result, index=['pred', 'true']).T
        result_df['correct'] = result_df['pred'] == result_df['true']
        class2acc = result_df.groupby('true')['correct'].agg(('sum', 'count')).apply(lambda x: x['sum']/x['count'], axis=1)
        return class2acc
    
    result['result']['outcome']['mis_class_acc'] = make_class2acc(mis_result)
    result['result']['outcome']['test_class_acc'] = make_class2acc(test_result)

    print()
    print('========================================Response model========================================')
    result.setdefault('result', {}).setdefault('response', {})
    # raw adjust방법( [ 1 - p(delta) / p(delta) ]과 비교
    mis_acc, mis_adjust_acc, mis_f1, mis_adjust_f1, mis_res_prob = compare_raw_adjust(dataset_obs, dataset_mis, n_class=2, 
                                                model=model, device=device, criterion=criterion, tau=tau)

    test_acc, test_adjust_acc, test_f1, test_adjust_f1, test_res_prob = compare_raw_adjust(dataset_obs, dataset_test, n_class=2, 
                                                model=model, device=device, criterion=criterion, tau=tau)

    print(f'mis_acc: {mis_acc}, mis_adjust_acc: {mis_adjust_acc}, mis_f1: {mis_f1}, mis_adjust_f1: {mis_adjust_f1}')
    print(f'test_acc: {test_acc}, test_adjust_acc: {test_adjust_acc}, test_f1: {test_f1}, test_adjust_f1: {test_adjust_f1}')

    result['result']['response']['mis_adjust_acc'] = mis_adjust_acc
    result['result']['response']['mis_adjust_f1'] = mis_adjust_f1
    result['result']['response']['test_adjust_acc'] = test_adjust_acc
    result['result']['response']['test_adjust_f1'] = test_adjust_f1
    result['result']['response']['mis_res_prob'] = mis_res_prob
    result['result']['response']['test_res_prob'] = test_res_prob
    
    # Response model
    response_model = ResponseModel(subset_obs_train_norm, model, device, criterion, 
                                num_classes=NUM_CLASSES, use_pool=False, tau=tau)
    response_model.iterate(n_iter = 50, dataset_mis=dataset_mis, sampling=False)
    obs_acc, obs_f1 = response_model.evaluate(subset_obs_train_norm, subset_obs_train_norm.y, 
                                    #   plot=True
                                      )
    mis_acc, mis_f1 = response_model.evaluate(dataset_mis, dataset_mis.y, 
                                    # plot=True
                                    )
    test_acc, test_f1 = response_model.evaluate(dataset_test, dataset_test.y, 
                                    #  plot=True
                                    )
    # print(f'obs_acc: {obs_acc}, obs_f1: {obs_f1}')
    print(f'mis_acc: {mis_acc}, mis_f1: {mis_f1}')
    print(f'test_acc: {test_acc}, test_f1: {test_f1}')


    result.setdefault('model', {}).setdefault('response', response_model)
    
    result['result']['response']['mis_adjust_acc'] = mis_adjust_acc
    result['result']['response']['test_adjust_acc'] = test_adjust_acc
    
    result['result']['response']['obs_acc'] = obs_acc
    result['result']['response']['obs_f1'] = obs_f1
    result['result']['response']['mis_acc'] = mis_acc
    result['result']['response']['mis_f1'] = mis_f1
    result['result']['response']['test_acc'] = test_acc
    result['result']['response']['test_f1'] = test_f1
    result['result']['response']['converge'] = response_model.converge
    
    
    result['result']['response']['mis_class_acc'] = response_model.make_class2acc(dataset_mis)
    result['result']['response']['test_class_acc'] = response_model.make_class2acc(dataset_test)
    
    result['result']['response']['coef'] = response_model.response_model_params[-1][-2:]
    result['result']['response']['response_model_res_prob'] = response_model.res_prob_ks

#     print()
#     print('========================================Option 4========================================')
    
#     # model param
#     ssl_model = ResNet18_cifar10(device=device, pretrained=False, num_classes=NUM_CLASSES)


#     ssl = SSL(model, response_model, ssl_model, 
#             dataset_obs_train, subset_obs_val_norm, dataset_mis_none, dataset_mis, tau=0.9)


#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     optimizer = torch.optim.SGD(ssl_model.parameters(), lr=0.01, #0.001 #0.001
#                                 momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#     # result dict
#     history = {'train_acc': [],
#             'train_loss' : [],
#             'val_acc' : [],
#             'val_loss' : []}

#     n_epoch = 600
#     ssl.fit(n_epoch = n_epoch,
#             device=device,
#             optimizer = optimizer,
#             scheduler = scheduler,
#             criterion = criterion,
#             history = history,
#             every_epoch=50
#             )
# ####
#     obs_loss, obs_acc, obs_f1, obs_result = evaluate(ssl_model, subset_obs_train_norm, device=device, criterion=criterion)
#     print(f'obs loss: {obs_loss}, obs_f1: {obs_f1}, obs acc: {obs_acc}')

#     obs_val_loss, obs_val_acc, obs_val_f1, obs_val_result = evaluate(ssl_model, subset_obs_val_norm, device=device, criterion=criterion)
#     print(f'obs_val loss: {obs_val_loss}, obs_val_f1: {obs_val_f1}, obs_val acc: {obs_val_acc}')

#     ## evaluate train_mis
#     mis_loss, mis_acc, mis_f1, mis_result = evaluate(ssl_model, dataset_mis, device=device, criterion=criterion)
#     print(f'mis loss: {mis_loss}, mis_f1: {mis_f1}, mis acc: {mis_acc}')

#     # evaluate trainset
#     dataset_train = Dataset(data_dict, mode='train', use_transform=True)
#     train_loss, train_acc, train_f1, train_result = evaluate(ssl_model, dataset_train, device=device, criterion=criterion)
#     print(f'train loss: {train_loss}, train_f1: {train_f1}, train acc: {train_acc}')

#     ## evaluate testset
#     dataset_test = Dataset(data_dict, mode='test', use_transform=True)
#     test_loss, test_acc, test_f1, test_result = evaluate(ssl_model, dataset_test, device=device, criterion=criterion)
#     print(f'test loss: {test_loss}, test_f1: {test_f1}, test acc: {test_acc}')
    
#     result.setdefault('result', {}).setdefault('option4', {})
#     result.setdefault('model', {}).setdefault('option4', ssl_model)
#     result['result']['option4']['obs_train_loss'] = history['train_loss'][-1]
#     result['result']['option4']['obs_train_acc'] = history['train_acc'][-1]
#     result['result']['option4']['obs_val_loss'] = history['val_loss'][-1]
#     result['result']['option4']['obs_val_acc'] = history['val_acc'][-1]
#     result['result']['option4']['obs_loss']= obs_loss
#     result['result']['option4']['obs_acc'] = obs_acc
#     result['result']['option4']['obs_f1'] = obs_f1
#     result['result']['option4']['mis_loss']= mis_loss
#     result['result']['option4']['mis_acc'] = mis_acc
#     result['result']['option4']['mis_f1'] = mis_f1
#     result['result']['option4']['train_loss'] = train_loss
#     result['result']['option4']['train_acc']= train_acc
#     result['result']['option4']['train_f1']= train_f1
#     result['result']['option4']['test_loss']= test_loss
#     result['result']['option4']['test_acc'] = test_acc
#     result['result']['option4']['test_f1'] = test_f1
# #### 
    
    return result

if __name__ == "__main__":
    label_weights = [[-2, -4]
        # [-2, -4], [-4, -2], [-1.8, -3.2], [-2.4, -3]
                     ] # class 2
    # label_weights = [np.linspace(-4, -1, 10), np.linspace(-2, 0, 10), np.linspace(-1, 1, 10), np.linspace(-2, 2, 10), np.linspace(-2.2, 0.3, 10)]
    model_seeds = [2000, 200, 10, 22, 5555
                #    10, 22, 5555, 7, 122
                   ]
    NUM_CLASSES = 2
    for label_weight in label_weights:
        for model_seed in model_seeds:
            for tau in [1, 2]:
                for alpha in [0.00, 0.14]:
                # for alpha in [0.05, 0.14, 0.3]:
                    result = main(model_seed=model_seed, label_weight=label_weight, alpha=alpha, tau=tau,
                                only_y =True, NUM_CLASSES = NUM_CLASSES)
                    torch.save(result, f'../results_alex/ret_class{NUM_CLASSES}_{"_".join(map(str, label_weight)).replace("-","_")}_{model_seed}_{alpha}_tau{tau}.pt')
        