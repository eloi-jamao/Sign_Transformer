from torch.autograd import Variable
import os
import json
import torch
from torch.utils import data
from torch import nn
import dataLoaderUtils
# from nltk.translate.bleu_score import sentence_bleu

'''Loading keypoints'''
videos_folder = os.path.join(os.getcwd(), "data", "json")


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
        x = self.read_sample(data_sample_path)
        x = torch.tensor(x)
        label_sample_path = self.labels_dirs[index]

        if self.padding:
            n = self.sample_length - len(x)
            if n > 0:
                padding = torch.cat(n*[torch.zeros(x[0].shape).unsqueeze(0)])
                x = torch.cat((x, padding))
        y = x
        return x, y

    def read_sample(self, sample_path):
        json_files = os.listdir(sample_path)
        sample = []
        for filename in json_files:
            frame_path = os.path.join(sample_path, filename)
            frame = dataLoaderUtils.read_json_2_openpose1dframe(frame_path)
            sample.append(frame)
        return sample
# def skp(prev, data):

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


if __name__ == '__main__':
    num_epochs = 20 #you can go for more epochs, I am using a mac
    batch_size = 128

    model = Autoencoder().cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    trainset = Dataset(videos_folder, videos_folder, True)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=1)

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
