import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import copy
import csv
from DataLoader import Dictionary, process_sentence


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'images{:04d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='|')
        next(csv_reader)
        return {row[0]: row[-1] for row in csv_reader}


# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/datasets/kinetics.py
def make_dataset(root_path, annotation_path,
                 #n_samples_for_each_video,
                 sample_duration,
                 dictionary):
    video2text = load_annotation_data(annotation_path)

    dataset = []
    i = 0
    dataset_size = len(os.listdir(root_path))
    for i, video in enumerate(os.listdir(root_path)):
        if video == ".DS_Store":
            continue

        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, dataset_size))

        video_path = os.path.join(root_path, video)
        if not os.path.exists(video_path):
            print(f"Path {video_path} not found!")
            continue

        n_frames = len(os.listdir(video_path))

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'label': process_sentence(video2text[video], dictionary)
        }

        step = sample_duration
        for j in range(1, n_frames, step):
            sample_j = copy.deepcopy(sample)
            sample_j['frame_indices'] = list(
                range(j, min(n_frames + 1, j + sample_duration)))
            dataset.append(sample_j)
    return dataset


def make_dataset_deprecated(video_path, sample_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)

    return dataset


class Video(data.Dataset):
    def __init__(self,
                 video_path,
                 translation_path,  # our csv file
                 spatial_transform=None,
                 temporal_transform=None,
                 sample_duration=4,
                 get_loader=get_default_video_loader):
        self.dictionary = Dictionary()
        self.data = make_dataset(video_path,
                                 translation_path,
                                 sample_duration,
                                 self.dictionary)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:

        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            #self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, self.data[index]['label']

    def __len__(self):
        return len(self.data)
