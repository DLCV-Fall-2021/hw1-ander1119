import torch.optim as optim
import torch
import numpy as np
import math
import imageio
import os
from PIL import Image


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def load_optimizer(model, config):
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    if config['name'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif config['name'] == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum)
    return optimizer

def read_image(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_image = len(file_list)
    images = np.empty((n_image, 512, 512, 3), dtype=np.float32)

    for i, file in enumerate(file_list):
        image = imageio.imread(os.path.join(filepath, file))
        images[i] = image

    return images

def read_single_image(filename):
    # return np.array(imageio.imread(filename))
    return np.array(Image.open(filename))

def read_single_mask(filename):
    # mask = imageio.imread(filename)
    mask = np.array(Image.open(filename))
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

    new_mask = np.zeros_like(mask)
    new_mask[mask == 3] = 0  # (Cyan: 011) Urban land 
    new_mask[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    new_mask[mask == 5] = 2  # (Purple: 101) Rangeland 
    new_mask[mask == 2] = 3  # (Green: 010) Forest land 
    new_mask[mask == 1] = 4  # (Blue: 001) Water 
    new_mask[mask == 7] = 5  # (White: 111) Barren land 
    new_mask[mask == 0] = 6  # (Black: 000) Unknown 

    return np.array(new_mask, dtype=int)

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512), dtype=np.longlong)

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    # print("pred:", pred)
    # print("labels:", labels)
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
