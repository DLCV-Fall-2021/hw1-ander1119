import torch
import torch.nn as nn
import argparse
import yaml
from shutil import copyfile

from dataset import *
from model import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_args()

    config = yaml.safe_load(open(opt.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join("checkpoints", config['name'])
    os.makedirs(save_dir, exist_ok=True)
    copyfile(opt.config, os.path.join(save_dir, 'config.yaml'))


    train_dataloader = load_dataloader('train', config['train'])
    validate_dataloader = load_dataloader('val', config['validate'])
    model = load_model(device, config['model'])
    optimizer = load_optimizer(model, config['optimizer'])
    criterion = nn.CrossEntropyLoss()
    print("model:\n", model)
    print("optimizer:\n", optimizer)
    
    best_acc = 0.
    iteration = 0
    for ep in range(config['train']['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % config['train']['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))
            iteration += 1

        model.eval() 
        test_loss = 0
        correct = 0
        with torch.no_grad(): # This will free the GPU memory used for back-prop
            for data, target in validate_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(validate_dataloader.dataset)
        valid_acc = 100. * correct / len(validate_dataloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(validate_dataloader.dataset),
            100. * correct / len(validate_dataloader.dataset)))

        if valid_acc > best_acc:
            best_acc = valid_acc
            save_checkpoint(os.path.join(save_dir, 'best.pth'), model, optimizer)
            print('saving model with acc {:.3f}'.format(best_acc))