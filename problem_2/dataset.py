import glob
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from utils import *

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = [file.replace('_sat.jpg', '') for file in glob.glob(os.path.join(root, '*_sat.jpg'))]
        self.filenames.sort()
        # self.images = read_image(root)
        # self.masks = read_masks(root)
        self.transform = transform

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = read_single_image(self.filenames[index] + '_sat.jpg')
        mask = read_single_mask(self.filenames[index] + '_mask.png')

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def __len__(self):
        return self.len

def load_dataloader(type, config):
    if type == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = CustomDataset(root=config['path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    elif type == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = CustomDataset(root=config['path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    return dataloader