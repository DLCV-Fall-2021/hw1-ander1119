import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # convolutional layer (sees 32x32x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # convolutional layer (sees 16x16x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # convolutional layer (sees 16x16x64 tensor)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        # linear layer (128*8*8 -> 500)
        self.fc1 = nn.Linear(128 * 8 * 8, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 50)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        
        
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # flatten image input
        x = x.view(-1, 128 * 8 * 8)
        # add dropout layer
        x = self.dropout(x)
        # linear layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # linear output layer
        x = self.fc2(x)
        return x


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, 50)

    def forward(self, x):
        return self.model(x)

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 50)
        self.model.aux_logits = False

    def forward(self, x):
        return self.model(x)

def load_model(device, model_name):
    if model_name == "vgg16":
        model = Vgg16()
    elif model_name == "inception_v3":
        model = InceptionV3()
    else:
        model = CustomNet()
    return model.to(device)