import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange


def transform_all(img, is_train):
    if is_train:
        return transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224), 
                    transforms.RandomRotation(5), 
                    transforms.ToTensor(),
                    ])(img)
    else:
        return transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224), 
                    transforms.ToTensor(),
                    ])(img)
    
def clahe_equalized(is_train=True):
    def inner(img):
        img = np.array(img, dtype='uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final_img = clahe.apply(img)
        final_img = Image.fromarray(final_img)
    
        return transform_all(final_img, is_train)
    
    return inner

def histogram_equalize(img, is_train=True):
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    he_img = cv2.equalizeHist(img)
    he_img = Image.fromarray(he_img)
    return transform_all(he_img, is_train)

def gammaCorrection(img):
    # sourced from: https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
    gamma = 2.2 # to be updated?
    img = np.array(img, dtype='uint8')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    gamma_corrected = cv2.LUT(img, table)
    gamma_corrected = Image.fromarray(gamma_corrected)
    return transform_all(gamma_corrected)

def BCET_inner(img):
    # sourced from: https://primer-computational-mathematics.github.io/book/d_geosciences/remote_sensing/Image_Point_Operations.html
    l= np.min(img)
    h= np.max(img)
    e= np.mean(img)
    L = 0
    H = 255 # use full dynamic range
    E = 110 # can be varied
    s = np.mean(img^2)
    b = (h**2 *(E-L)-s*(H-L)+l**2 *(H-E)) / (2*(h*(E-L)-e*(H-L)+l*(H-E)))
    a = (H-L)/((h-l)*(h+l-2*b))
    c = L-a*(l-b)**2
    y = a*(img-b)**2 +c
    y.astype(int) # ensure all integers to plot image
    return y #transform['BCET']['test'](y)

def BCET(img):
    img = np.array(img)
    r = BCET_inner(img[:,:,0])
    g = BCET_inner(img[:,:,1]) 
    b = BCET_inner(img[:,:,2]) 
    rgb = np.dstack([r,g,b])
    rgb = Image.fromarray((rgb * 255).astype(np.uint8))
    return transform_all(rgb)


metadata_path = '~/Documents/Adrian/ChestX-ray14/Data_Entry_2017_v2020.csv'
class_mappings = {
    'Cardiomegaly': 'Cardiomegaly',
    'Opacity': 'Atelectasis|Efussion|Infiltration|Pneumonia|Consolidation|Edema|Fibrosis|Pleural_Thickening',
    'Emphysema': 'Emphysema'
}

transform = {
    'orig': {
        'train':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.RandomRotation(5), 
            transforms.ToTensor(),
            ]),
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        },
    'norm': {
        'train':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.RandomRotation(5), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                          (0.229, 0.224, 0.225))
            ]),
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                          (0.229, 0.224, 0.225))
            ]),
        'test':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                              (0.229, 0.224, 0.225))
            ]),
        },
    'CLAHE': {
        'train': clahe_equalized(is_train=True),
        'val':
            clahe_equalized(is_train=False),
        'test':
            clahe_equalized(is_train=False),
    },
    'HE': {
        'train': histogram_equalize,
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                ]),
        
    },
    'complement': {
        'train':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.RandomRotation(15), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: torch.tensor([1.]).repeat(img.shape).sub(img))
            ]),
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        },
    'gamma_correct': {
        'train': gammaCorrection,
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                ]),
        
    },
    'BCET': {
        'train': BCET,
        'val':
            transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                ]),
        
    },
   
   
}


class ChestXray14(Dataset):
    def __init__(self, phase, csv_file=metadata_path, class_name='Cardiomegaly', transform_type='orig'):
        metadata = pd.read_csv(csv_file)
        metadata = metadata
        positive = metadata.loc[metadata['Finding Labels'].str.contains(class_mappings[class_name], regex=True)]
        negative = metadata.loc[metadata['Finding Labels'].str.contains('No Finding')]
        
        num_train = int(.6 * len(positive))
        num_val = int(.3 * len(positive))
        num_test = int(.1 * len(positive))
        
        train_metadata = pd.concat([positive[:num_train], negative[:num_train]])
        val_metadata = pd.concat([positive[num_train:num_train + num_val], negative[num_train : num_train + num_val]])
        test_metadata = pd.concat([positive[num_train + num_val:], negative[num_train + num_val : num_train + num_val + num_test]])

        # balanced metadata with # negative examples = # positive examples
        self.metadata = {
            'train': train_metadata,
            'val': val_metadata,
            'test': test_metadata
        }
        self.phase = phase
        self.transform = transform_type
        
    def __len__(self): 
        return len(self.metadata[self.phase])
    
    def __getitem__(self, idx):
        row = self.metadata[self.phase].iloc[idx]    
        image = cv2.imread('/home/developer/Documents/Adrian/ChestX-ray14/images/' + row['Image Index'])
        image = Image.fromarray(image)
        image = transform[self.transform][self.phase](image)
        if 'No Finding' in row['Finding Labels']:
            return image, 0
        return image, 1