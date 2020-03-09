import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import json
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from nltk.translate.bleu_score import sentence_bleu

'''Loading keypoints'''
videos_folder = os.path.join(os.getcwd(), "data", "json")


def read_json_2_openpose1dframe(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        people_dict = data['people']
        keypoints_dict = people_dict[0]
        keypoints_x = []
        keypoints_y = []
        keypoints_c = []
        keypoints = keypoints_dict['pose_keypoints_2d'][:8 * 3]
        keypoints += keypoints_dict['hand_left_keypoints_2d']
        keypoints += keypoints_dict['hand_right_keypoints_2d']
        for i, point in enumerate(keypoints):
            j = i % 3
            if j == 0:
                keypoints_x.append(point)
            elif j == 1:
                keypoints_y.append(point)
            elif j == 2:
                keypoints_c.append(point)
    return [keypoints_x, keypoints_y, keypoints_c]


def read_sample(sample_path):
    json_files = os.listdir(sample_path)
    sample = []
    for filename in json_files:
        frame_path = os.path.join(sample_path, filename)
        frame = read_json_2_openpose1dframe(frame_path)
        sample.append(frame)
    return sample


class Dataset(data.Dataset):

    def __init__(self, data_openpose_json_dir, labels_openpose_json_dir, padding=False):

        self.data_folders = os.listdir(data_openpose_json_dir)
        self.labels_folders = os.listdir(labels_openpose_json_dir)

        self.data_samples = []
        self.padding = padding
        self.sample_length = 0

        self.data_dirs = []
        self.labels_dirs = []

        for folder in self.data_folders:
            sample_data_path = os.path.join(data_openpose_json_dir, folder)
            self.data_dirs.append(sample_data_path)
            sample_labels_path = os.path.join(labels_openpose_json_dir, folder)
            self.labels_dirs.append(sample_labels_path)
            frames = os.listdir(sample_data_path)
            self.sample_length = len(frames) if len(frames) > self.sample_length else self.sample_length

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, index):

        data_sample_path = self.data_dirs[index]
        x = read_sample(data_sample_path)
        x = torch.tensor(x)
        label_sample_path = self.labels_dirs[index]

        if self.padding:
            n = self.sample_length - len(x)
            if n > 0:
                padding = torch.cat(n*[torch.zeros(x[0].shape).unsqueeze(0)])
                x = torch.cat((x, padding))
        # TODO
        # y = read_sample(label_sample_path)
        y = x
        return x, y
# def skp(prev, data):


def avg(data, threshold=0.0, windowing=1, overlap=0, skip=0.0, loss=nn.MSELoss(), device='cpu'):
    data.to(device)
    finaldata = []
    dta = None
    prev = None
    for i in range(0, len(data), windowing):
        window = []
        t = torch.tensor(0).to(device)
        for j in range(i - overlap, i + windowing + overlap):
            if 0 <= j < len(data) and data[j][2].mean() > threshold:
                if dta is not None:
                    prev = dta
                dta = data[j][:2]
                if prev is not None and loss(dta, prev) < skip:
                    continue
                weight = data[j][2]
                t = torch.add(t, weight)

                window.append(dta * weight)
        resu = torch.tensor(0).to(device)
        for part in window:
            resu = torch.add(resu, part)
        if t.max() > 0:
            resu = torch.div(resu, t)
            finaldata.append(resu)
    return torch.stack(finaldata)







# PRETRAIN

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2729, 1000, kernel_size=(3, 10)),
            nn.ReLU(True),
            nn.Conv2d(1000, 500, kernel_size=(1, 3)),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(500, 1000, kernel_size=(1, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(1000, 2729, kernel_size=(3, 10)),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

num_epochs = 20 #you can go for more epochs, I am using a mac
batch_size = 128

model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

trainset = Dataset(videos_folder, videos_folder, True)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cpu()
        # ===================forward=====================
        output = model(img)
        loss = distance(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
    # print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss))
