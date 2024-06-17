import numpy as np
import matplotlib.pyplot as plt 
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def load_cifar10():
    data_dict = {}

    for batch_idx in range(1, 6):
        train_path = f'../data/cifar-10-batches-py/data_batch_{batch_idx}'
        train_dict = unpickle(train_path)
        train_dict[b'labels']
        data_dict.setdefault('train_labels', []).extend(train_dict[b'labels'])
        data_dict.setdefault('train', []).append(train_dict[b'data'])
        
    data_dict['train'] = np.concatenate(data_dict['train'], axis=0)
    data_dict['train_labels'] = np.array(data_dict['train_labels'])
    
    test_path = '../data/cifar-10-batches-py/test_batch'
    test_dict = unpickle(test_path)
    data_dict['test_labels'] = np.array(test_dict[b'labels'])
    data_dict['test'] = test_dict[b'data']

    

    meta_path = '../data/cifar-10-batches-py/batches.meta'
    meta_dict = unpickle(meta_path)
    data_dict['label_names'] = [name.decode() for name in meta_dict[b'label_names']]
    return data_dict



def cut_class(data_dict, remain):
    data_dict = data_dict.copy()
    name2idx = {name : idx for idx, name in enumerate(data_dict['label_names']) if name in remain}
    data_dict['label_names'] = remain
    
    for mode in ['train', 'test']:
        check = np.zeros_like(data_dict[f'{mode}_labels'])
        dummy_labels = np.zeros_like(data_dict[f'{mode}_labels'])
        
        for idx, name in enumerate(remain):
            _check = (data_dict[f'{mode}_labels'] == name2idx[name])
            dummy_labels[_check] = idx
            check += _check
            
        check = check.astype(bool)
        data_dict[f'{mode}'] = data_dict[f'{mode}'][check]
        data_dict[f'{mode}_labels'] = dummy_labels[check]
    return data_dict


def get_N_obs(N_max, gamma, i, n_cls):
    n_obs = N_max * gamma**(-i / (n_cls-1))
    return int(n_obs)


def split_obs_mis(data_dict, N_max, gamma, seed=100):
    np.random.seed(seed)
    
    n_cls = len(data_dict['label_names'])
    n_train = len(data_dict['train'])
    
    class_shuffled = [i for i in set(data_dict['train_labels'])]
    np.random.shuffle(class_shuffled)
    
    train_idx = np.array([i for i in range(n_train)])
    labeled = np.array([False for _ in range(n_train)])
    
    class2n_obs = {}
    y_train = data_dict['train_labels']
    for idx, class_num in enumerate(class_shuffled):
        class_bool = y_train == class_num
        class_idx = train_idx[class_bool]
        n_obs = get_N_obs(N_max, gamma, idx, n_cls)
        class2n_obs[class_num] = n_obs
        obs_idx = np.random.choice(class_idx, n_obs, replace=False)
        labeled[obs_idx] = True
    
    data_dict['class2n_obs'] = class2n_obs
    data_dict['train_obs'] = data_dict['train'][labeled]
    data_dict['train_obs_labels'] = data_dict['train_labels'][labeled]
    data_dict['train_mis'] = data_dict['train'][~labeled]
    data_dict['train_mis_labels'] = data_dict['train_labels'][~labeled]
    data_dict['labeled'] = labeled

def split_obs_mis_with_lr(data_dict, label_weight=[0.2, 0.6], seed=100):
    np.random.seed(seed)
    
    imgs = data_dict['train']
    labels = data_dict['train_labels']

    img_weights = np.random.normal(size=len(imgs[0]))
    label_weights = np.array(label_weight)
    # z = imgs @ img_weights / len(img_weights) + label_weights[labels]
    z = label_weights[labels]
    probs = 1/(1+np.exp(-z)) #* label_weights[labels]
    response = np.random.binomial(1, p=probs)
    labeled = response.astype(bool)
    
    class2n_obs = {}
    for i in set(labels):
        class2n_obs[i] = np.unique(response[labels==i], return_counts=True)[1][1]
    
    data_dict['class2n_obs'] = class2n_obs
    data_dict['train_obs'] = data_dict['train'][labeled]
    data_dict['train_obs_labels'] = data_dict['train_labels'][labeled]
    data_dict['train_mis'] = data_dict['train'][~labeled]
    data_dict['train_mis_labels'] = data_dict['train_labels'][~labeled]
    

def show_img(img, label=None):
    img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    plt.imshow(img)
    if label:
        plt.title(label)
    plt.show()
    

class Dataset(Dataset):
    def __init__(self, data_dict, mode, use_transform=True):
        self.mode = mode
        self.num_classes = len(data_dict['label_names'])
        self.X = data_dict[f'{mode}'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.y = data_dict[f'{mode}_labels']
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        if use_transform:
            # mean = data_dict['train_obs'].reshape(-1, 3, 32, 32).mean(axis=(0, 2, 3))/255
            # std = data_dict['train_obs'].reshape(-1, 3, 32, 32).std(axis=(0, 2, 3))/255
            self.transform = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ])

        
    def __getitem__(self, idx):
        X_i = self.X[idx]
        y_i = self.y[idx]
        return self.transform(X_i), y_i
        
    
    def __len__(self):
        return len(self.X)
    
    
    def show_img(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        plt.imshow(img)
        plt.title(label)
        plt.show()


class Subset(Dataset):
    def __init__(self, subset, use_transform):
        self.subset = subset
        self.use_transform= use_transform
        self.y = [s[1] for s in subset]
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if use_transform:            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
                    
    def __getitem__(self, idx): 
        X_i, y_i = self.subset[idx]
        
        return self.transform(X_i), y_i
    
    def __len__(self):
        return len(self.subset)


class Dataset_from_norm_Xy(Dataset):
    def __init__(self, X, y, use_transform=True):
        self.X = X
        self.y = y
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        if use_transform:            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
    def __getitem__(self, idx): 
        X_i, y_i = self.X[idx], self.y[idx]
        
        return self.transform(X_i), y_i
    
    def __len__(self):
        return len(self.X)
    
# class TestDataset(Dataset):
#     def __init__(self, data_dict, use_transform=True):
#         self.X = data_dict[f'test'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#         self.y = data_dict[f'test_labels']
#         if use_transform:
#             # mean = data_dict['train_obs'].reshape(-1, 3, 32, 32).mean(axis=(0, 2, 3))/255
#             # std = data_dict['train_obs'].reshape(-1, 3, 32, 32).std(axis=(0, 2, 3))/255
#             mean = (0.4914, 0.4822, 0.4465)
#             std = (0.2023, 0.1994, 0.2010)
             
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
            
#     def __getitem__(self, idx):
#         X_i = self.X[idx]
#         y_i = self.y[idx]
        
#         return self.transform(X_i), y_i
    
#     def __len__(self):
#         return len(self.X)


# class TrainDataset(Dataset):
#     def __init__(self, data_dict, use_transform=True):
#         self.X = data_dict[f'train'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#         self.y = data_dict[f'train_labels']
#         if use_transform:
#             # mean = data_dict['train_obs'].reshape(-1, 3, 32, 32).mean(axis=(0, 2, 3))/255
#             # std = data_dict['train_obs'].reshape(-1, 3, 32, 32).std(axis=(0, 2, 3))/255
#             mean = (0.4914, 0.4822, 0.4465)
#             std = (0.2023, 0.1994, 0.2010)
             
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
            
#     def __getitem__(self, idx):
#         X_i = self.X[idx]
#         y_i = self.y[idx]
        
#         return self.transform(X_i), y_i
    
#     def __len__(self):
#         return len(self.X)