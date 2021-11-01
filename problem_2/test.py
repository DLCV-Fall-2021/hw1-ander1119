import torch
import argparse
import scipy.ndimage
import imageio
import os
import numpy as np 
from matplotlib import colors as mcolors

from dataset import *
from model import *
from utils import *

voc_cls = {'urban':0, 
           'rangeland': 2,
           'forest':3,  
           'unknown':6,  
           'barreb land':5,  
           'Agriculture land':1,  
           'water':4} 
cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def mask_edge_detection(mask, edge_width):
    h = mask.shape[0]
    w = mask.shape[1]

    edge_mask = np.zeros((h,w))
    
    for i in range(h):
        for j in range(1,w):
            j_prev = j - 1 
            # horizantal #
            if not mask[i][j] == mask[i][j_prev]: # horizontal
                if mask[i][j]==1: # 0 -> 1
                    edge_mask[i][j] = 1
                    for add in range(1,edge_width):
                        if j + add < w and mask[i][j+add] == 1:
                            edge_mask[i][j+add] = 1

                        
                else : # 1 -> 0
                    edge_mask[i][j_prev] = 1
                    for minus in range(1,edge_width):
                        if j_prev - minus >= 0 and mask[i][j_prev - minus] == 1: 
                            edge_mask[i][j_prev - minus] = 1
            # vertical #
            if not i == 0 :
                i_prev = i - 1
                if not mask[i][j] == mask[i_prev][j]: 
                    if mask[i][j]==1: # 0 -> 1
                        edge_mask[i][j] = 1 
                        for add in range(1,edge_width):
                            if i + add < h and mask[i+add][j] == 1:
                                edge_mask[i+add][j] = 1 
                    else : # 1 -> 0
                        edge_mask[i_prev][j] = 1
                        for minus in range(1,edge_width):
                            if i_prev - minus >= 0 and mask[i_prev-minus][j] == 1:
                                edge_mask[i_prev-minus][j] == 1
    return edge_mask

def viz_data(im, seg, color, inner_alpha = 0.3, edge_alpha = 1, edge_width = 5):
     
    edge = mask_edge_detection(seg, edge_width)

    color_mask = np.zeros((edge.shape[0]*edge.shape[1], 3))
    l_loc = np.where(seg.flatten() == 1)[0]
    color_mask[l_loc, : ] = color
    color_mask = np.reshape(color_mask, im.shape)
    mask = np.concatenate((seg[:,:,np.newaxis],seg[:,:,np.newaxis],seg[:,:,np.newaxis]), axis = -1)
    
    color_edge = np.zeros((edge.shape[0]*edge.shape[1], 3))
    l_col = np.where(edge.flatten() == 1)[0]
    color_edge[l_col,:] = color
    color_edge = np.reshape(color_edge, im.shape)
    edge = np.concatenate((edge[:,:,np.newaxis],edge[:,:,np.newaxis],edge[:,:,np.newaxis]), axis = -1)


    im_new = im*(1-mask) + im*mask*(1-inner_alpha) + color_mask * inner_alpha
    im_new =  im_new*(1-edge) + im_new*edge*(1-edge_alpha) + color_edge*edge_alpha

    return im_new 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--test-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'fcn8' in opt.checkpoint:
        model = FCN8().to(device)
    elif 'fcn32' in opt.checkpoint:
        model = FCN32().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    load_checkpoint(opt.checkpoint, model, optimizer)

    # print("model:\n", model)

    model.eval()
    with torch.no_grad():
        filepaths = glob.glob(os.path.join(opt.test_dir, '*.jpg'))
        filepaths.sort()
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for filepath in filepaths:
            test_img = Image.open(filepath)
            test_tensor = transform(test_img)
            input_batch = test_tensor.unsqueeze(0)

            output = model(input_batch.to(device))[0]
            segmentation_map = output.argmax(0).byte().cpu().numpy()

            mask = np.zeros((512, 512, 3))
            for cls in range(7):
                mask[segmentation_map == cls] = cls_color[cls]
            
            filename = filepath.split('/')[-1].replace('jpg', 'png')
            imageio.imsave(os.path.join(opt.output_dir, filename), np.uint8(mask))