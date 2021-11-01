from numpy.random.mtrand import rand
import torch
import argparse
import csv
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--test-dir', type=str)
    parser.add_argument('--output-csv', type=str)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionV3().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    load_checkpoint(opt.checkpoint, model, optimizer)

    # print("model:\n", model)

    csv_file = open(opt.output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_id', 'label'])
    model.eval()
    with torch.no_grad():
        filenames = glob.glob(os.path.join(opt.test_dir, '*.png'))
        filenames.sort()
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for filename in filenames:
            test_img = Image.open(filename)
            test_tensor = transform(test_img)
            input_batch = test_tensor.unsqueeze(0)

            output = model(input_batch.to(device))
            x = output[0].detach().cpu().numpy()
            pred = output[0].max(0, keepdim=True)[1].item()

            csv_writer.writerow([filename.split('/')[-1], pred])

    csv_file.close()

