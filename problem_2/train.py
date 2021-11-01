from numpy import dtype
import torch
from torch.functional import Tensor
import torch.nn as nn
import argparse
import yaml
from shutil import copyfile
import torch.nn.functional as F


from dataset import *
from model import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
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
    if opt.resume:
        load_checkpoint(os.path.join(save_dir, "best.pth"), model, optimizer)
    criterion = nn.CrossEntropyLoss()
    print("model:\n", model)
    print("optimizer:\n", optimizer)

    best_iou = 0.
    iteration = 0
    for ep in range(config['train']['epochs']):
        model.train()

        total_loss =0.
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label= data.to(device), label.to(device)
            output = model(data)
            # output = F.log_softmax(output, dim=1)
            # pred = output.argmax(1)
            # print(output.shape)
            # print(label.shape)
            # print(output.shape)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Train Epoch: {} Loss: {:.6f}'.format(ep, total_loss))

        model.eval() 
        with torch.no_grad(): # This will free the GPU memory used for back-prop
            preds = []
            targets = []
            for data, label in validate_dataloader:
                data = data.to(device, dtype=torch.float)
                output = model(data)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(1)
                # preds.append(pred.detach().cpu().numpy())
                # targets.append(target.cpu().numpy())
                preds.append(pred)
                targets.append(label)
        # preds = np.concatenate(preds, axis=0)
        # targets = np.concatenate(targets, axis=0)
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            print(preds.shape)
            print(targets.shape)
            mean_iou = mean_iou_score(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())
        # mean_iou_loss = mean_iou_score(preds, targets)
        # mean_iou_loss /= len(validate_dataloader.dataset)
            print('Test set: IOU: {:.4f}'.format(mean_iou))

            if mean_iou > best_iou:
                best_iou = mean_iou
                save_checkpoint(os.path.join(save_dir, 'best.pth'), model, optimizer)
                print('saving model with acc {:.3f}'.format(best_iou))

    print('best.pt with iou {:.4f}'.format(best_iou))