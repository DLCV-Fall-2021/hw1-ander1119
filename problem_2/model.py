import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class FCN32(nn.Module):
    def __init__(self, num_classes=7):
        super(FCN32, self).__init__()
        self.feature = models.vgg16(pretrained=True).features
        self.conv1 = nn.Sequential(nn.Conv2d(512, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.conv3 = nn.Conv2d(4096, num_classes, 1)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=32, bias=False)

    def forward(self, x):
        # print("forward input", x.shape)
        x = self.feature(x)
        # print("forward vgg feature", x.shape)
        x = self.conv1(x)
        # print("forward conv1", x.shape)
        x = self.conv2(x)
        # print("forward conv2", x.shape)
        x = self.conv3(x)
        # print("forward conv3", x.shape)
        x = self.upsample(x)
        # print("forward upsample", x.shape)
        return x

class FCN8(nn.Module):
    def __init__(self, num_classes=7):
        super(FCN8, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
        )

        self.conv1 = nn.Sequential(nn.Conv2d(512, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        
        self.score_32 = nn.Conv2d(4096, num_classes, 1)
        self.score_16 = nn.Conv2d(512, num_classes, 1)
        self.score_8 = nn.Conv2d(256, num_classes, 1)

        self.upsample_2x_32 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x_32.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x_16 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x_16.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_8x_8 = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x_8.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.load_pretrained_vgg16()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        pool3 = x
        x = self.block4(x)
        pool4 = x
        x = self.block5(x)
        x = self.conv1(x)
        x = self.conv2(x)

        score32 = self.score_32(x)
        score32 = self.upsample_2x_32(score32)

        score16 = self.score_16(pool4) + score32
        score16 = self.upsample_2x_16(score16)

        score8 = self.score_8(pool3) + score16
        score8 = self.upsample_8x_8(score8)

        # print("upsample 32:", upsample32.shape)
        # print("upsample 16:", upsample16.shape)
        # print("upsample 8:", upsample8.shape)

        return score8

    def load_pretrained_vgg16(self):
        vgg16_feat = models.vgg16(pretrained=True).features
        for key_fcn8, key_vgg16 in zip(self.state_dict(), vgg16_feat.state_dict()):
            self.state_dict()[key_fcn8].copy_(vgg16_feat.state_dict()[key_vgg16])
            # print(self.state_dict()[key_fcn8], vgg16_feat.state_dict()[key_vgg16])
            # print(key_fcn8, key_vgg16)
        # print("load from vgg16 complete")

def load_model(device, model_name):
    if model_name == "fcn32":
        model = FCN32()
    elif model_name == "fcn8":
        model = FCN8()
    return model.to(device)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)