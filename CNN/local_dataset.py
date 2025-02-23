# Library Imports ---------------------------------------------------------------------------------------------------------------------

from PIL import Image
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms

# Dataset Class -----------------------------------------------------------------------------------------------------------------------

local_neg_dir = 'TrainingImages/negatives'
local_pos_dir = 'TrainingImages/positives'

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor()  
])

class Task1Dataset(torch.utils.data.Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        # Directory and filename setup
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.pos_filenames = os.listdir(pos_dir)
        self.neg_filenames = os.listdir(neg_dir)

        # Transform setup
        self.transform = transform

    # Returns total samples in dataset (pos+neg len)
    def __len__(self):
        return len(self.pos_filenames)+len(self.neg_filenames)

    # Split items into negative and positive 
    def __getitem__(self, idx):
        ## Positive
        if idx<len(self.pos_filenames): 
            img_name = os.path.join(self.pos_dir, self.pos_filenames[idx])
            label = 1
        ## Negative
        else: 
            img_name = os.path.join(self.neg_dir, self.neg_filenames[idx-len(self.pos_filenames)])
            label = 0
        image = Image.open(img_name)

        # Apply transforms 
        if self.transform:
            image = self.transform(image)

        return image, label
    
## -- Initialize dataset
full_dataset = Task1Dataset(
    neg_dir=local_neg_dir,
    pos_dir=local_pos_dir,
    transform=transform
)

## -- Split Dataset 
train_size = int(0.75*len(full_dataset))
val_size = len(full_dataset)-train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

## -- Setup DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)