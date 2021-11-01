import glob
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(50):
            filenames = glob.glob(os.path.join(root, f"{i}_*.png"))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair            

        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.len

def load_dataloader(type, config):
    if type == "train":
        transform = transforms.Compose([
            transforms.Resize(config['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = CustomDataset(root=config['path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    elif type == "val":
        transform = transforms.Compose([
            transforms.Resize(config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = CustomDataset(root=config['path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    return dataloader