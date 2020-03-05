import os
import json
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

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

    def __init__(self, data_openpose_json_dir, labels_openpose_json_dir):

        self.data_folders = os.listdir(data_openpose_json_dir)
        self.labels_folders = os.listdir(labels_openpose_json_dir)

        self.data_samples = []
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

        # n = self.sample_length - len(x)
        # padding = torch.zeros(x[0].shape)
        # x = torch.cat((x, padding), 1)

        # TODO
        # y = read_sample(label_sample_path)
        y = x
        return x, y

#2729
ds = Dataset(videos_folder, videos_folder)
item = ds.__getitem__(0)
item2 = ds.__getitem__(1)

ds = [item[0], item2[0]]
res = torch.nn.utils.rnn.pad_sequence(ds, batch_first=True, padding_value=0)
res.data.numpy()[-1]