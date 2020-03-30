import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import copy
import csv


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


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/datasets/kinetics.py
def make_dataset(root_path, annotation_path,
                 #n_samples_for_each_video,
                 sample_duration):
    video2text = load_annotation_data(annotation_path)

    dataset = []
    i = 0
    for video, text in video2text.items():
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video2text)))

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
            'text': text  # change to index
        }

        if n_samples_for_each_video > 1:
            step = max(1,
                       math.ceil((n_frames - 1 - sample_duration) /
                                 (n_samples_for_each_video - 1)))
        else:
            step = sample_duration
        for j in range(1, n_frames, step):
            sample_j = copy.deepcopy(sample)
            sample_j['frame_indices'] = list(
                range(j, min(n_frames + 1, j + sample_duration)))
            dataset.append(sample_j)

        i = i + 1

    return dataset, idx_to_class


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
        self.data = make_dataset(video_path,
                                 #translation_path,
                                 sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
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

        target = self.data[index]['segment']  # should return translation

        return clip, target

    def __len__(self):
        return len(self.data)
