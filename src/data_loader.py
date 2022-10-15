import torch
import torch.utils.data as torchdata
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd
from pathlib import Path
from PIL import Image

class CelebData(torchdata.Dataset):
    ''' Custom Dataset for Celeb Data '''
    
    ATTR_FP = Path('data/list_attr_celeba.csv')
    DATA_DIR = Path('data/img_align_celeba')
    
    def __init__(self, root = '.', sample = False):
        super().__init__()
        self.root = Path(root)
        
        if sample:
            self.ATTR_FP = Path('data_sample/list_attr_celeba.csv')
            self.DATA_DIR = Path('data_sample/img_align_celeba')
        
        self.attr_filter = pd.read_csv(self.root / self.ATTR_FP)
        self.filter = ['Male']
    
    def get_all_filter(self):
        return self.attr_filter.columns[1:] # all except for image_id
    
    def set_filter(self, filter_):
        '''filter: List of columns to output as tensor of -1 and 1'''
        self.filter = filter_
    
    def __len__(self):
        # number of rows in df
        return self.attr_filter.shape[0]
    
    def __getitem__(self, idx):
        # get img as tensor
        img_name = self.attr_filter.iloc[idx]['image_id']
        img = Image.open(self.root / self.DATA_DIR / img_name)
        img_tensor = pil_to_tensor(img)
        
        # get target as filter
        target = torch.tensor(self.attr_filter.iloc[idx][self.filter].values.tolist())
        return img_tensor, target
        
    def filter_dataset(self, filter_):
        ''' return a subset of data with the filtered index 
        filter: Dict of column: value to filter out'''
        tmp = self.attr_filter
        for key, value in filter_.items():
            tmp = tmp[tmp[key] == value]
        
        return torchdata.Subset(self, indices=tmp.index)
    
class FairFaceData(torchdata.Dataset):
    
    VAL_FP = Path('data/fairface_label_val.csv')
    TRAIN_FP = Path('data/fairface_label_train.csv')
    DATA_DIR = Path('data/fairface-img-margin025-trainval')
    
    def __init__(self, root = '.', sample = False) -> None:
        super().__init__()
        self.root = Path(root)
        
        if sample:
            self.VAL_FP = Path('data_sample/fairface_label_val.csv')
            self.TRAIN_FP = Path('data_sample/fairface_label_train.csv')
            self.DATA_DIR = Path('data_sample/fairface-img-margin025-trainval')
                
        self.attr_filter = pd.concat([pd.read_csv(root / self.TRAIN_FP), pd.read_csv(root / self.VAL_FP)]) # combine train and val with our own data style
        self.attr_filter['index'] = list(range(len(self.attr_filter)))
        self.attr_filter.set_index('index', inplace = True)
        # create map of attr and their encodings
        self.attr_map = {col: {name: idx for idx, name in enumerate(self.attr_filter[col].unique())} for col in self.attr_filter.columns if col != 'file'} 
        
        # fill array with encodings
        self.attr_filter.replace(self.attr_map, inplace = True)
        
        
        self.filter = ['gender']
    
    def get_attr_map(self):
        return self.attr_map

    def get_all_filter(self):
        return self.attr_filter.columns[1:] # all except for image_id
    
    def set_filter(self, filter_):
        '''filter: List of columns to output as tensor of -1 and 1'''
        self.filter = filter_
        
    def __len__(self):
        # number of rows in df
        return self.attr_filter.shape[0]
    
    def __getitem__(self, idx):
        # get img as tensor
        img_name = self.attr_filter.iloc[idx]['file']
        img = Image.open(self.root / self.DATA_DIR / img_name)
        img_tensor = pil_to_tensor(img)
        
        # get target as filter
        target = torch.tensor(self.attr_filter.iloc[idx][self.filter].values.tolist())
        return img_tensor, target

    def filter_dataset(self, filter_):
        ''' return a subset of data with the filtered index 
        filter: Dict of column: value to filter out'''
        tmp = self.attr_filter
        for key, value in filter_.items():
            tmp = tmp[tmp[key] == self.attr_map[key][value]]
        return torchdata.Subset(self, indices=tmp.index)