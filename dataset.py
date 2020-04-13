import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import csv
# import utils
import torch
from PIL import Image


def transform_video(video):
    return video


class STDataset(Dataset):
    def __init__(self, split, root_path=os.path.join(os.getcwd(), "data"), frames_path="frames", csv_path="annotations", create_vocabulary=False):

        splits = ['train', 'test', 'dev']
        self.split = split
        if self.split not in splits:
            raise Exception('Split must be one of:', splits)

        self.samples = []

        # Paths
        self.img_dir = os.path.join(root_path, "features", "fullFrame-210x260px", self.split)
        self.csv_file = os.path.join(root_path, csv_path, self.split + ".csv")

        self.dictionary = Dictionary()

        img_dir_sent = os.listdir(self.img_dir)
        # Open the csv file and extract img_folder, gloss sentence and label sentence
        with open(self.csv_file) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                if row[0] in img_dir_sent:
                    self.samples.append({"sentence_name": row[0], "gloss": row[-2], "sentence": row[-1]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_sentence_path = os.path.join(self.img_dir, self.samples[idx].get("sentence_name"))
        video = []
        for image_name in os.listdir(img_sentence_path):
            image_path = os.path.join(img_sentence_path, image_name)
            image_np = io.imread(image_path)
            video.append(image_np)
        x = transform_video(video)
        sentence = self.samples[idx].get("sentence")
        y = self.process_sentence(sentence, self.dictionary)

        return x, y

    def process_sentence(self, sentence, dictionary_):
        # first four words are:
        start, end, unk, pad = [dictionary_.idx2word[i] for i in range(4)]
        tok_sent = []

        # tokenization using the dictionary
        for word in sentence.split():
            if word in dictionary_.idx2word:
                tok_sent.append(dictionary_.word2idx[word])
            else:
                tok_sent.append(dictionary_.word2idx[unk])

        # now introduce the start and end tokens
        tok_sent.insert(0, dictionary_.word2idx[start])
        tok_sent.append(1)  # 1 is the end token

        # padding sentence to max_seq
        max_seq = 17 if dictionary_.gloss else 27
        for i in range(max_seq - len(tok_sent)):
            tok_sent.append(dictionary_.word2idx[pad])

        return torch.LongTensor(tok_sent)


class Dictionary(object):
    def __init__(self, vocab_path='data/vocabulary.txt', gloss=False):
        self.gloss = gloss

        self.idx2word = self.read_vocab(vocab_path)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def read_vocab(self, path):
        vocabulary = []
        with open(path) as f:
            for i, line in enumerate(f):
                word = line.rstrip()
                vocabulary.append(word)

        return vocabulary

    def __len__(self):
        return len(self.idx2word)


def decode_sentence(index_sentence, dictionary):
    sentence = [dictionary.idx2word[i] for i in index_sentence]
    return sentence


if __name__ == '__main__':

    dataset = STDataset(split='train')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(len(dataset))

    for images, label in dataset:
        print('images', images)
        print('label', label)
        break

