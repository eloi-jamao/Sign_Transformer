import os
from torch.utils.data import Dataset, DataLoader
import csv
#import utils
import torch
from PIL import Image

class SNLT_Dataset(Dataset):
    def __init__(self, split, frames_path = "data/frames/", csv_path = "data/annotations/", gloss = False, create_vocabulary = False):

        splits = ['train','test','dev']
        self.split = split
        if self.split not in splits:
            raise Exception('Split must be one of:', splits)

        self.samples = []
        self.gloss = gloss
        #Paths
        self.img_dir = frames_path + self.split
        self.csv_path = csv_path
        self.csv_file = csv_path + self.split + ".csv"

        if self.gloss:
            self.gloss_dictionary = Dictionary(vocab_path='data/gloss_vocabulary.txt', gloss = True)
        self.dictionary = Dictionary()

        #Open the csv file and extract img_folder, gloss sentence and label sentence
        with open(self.csv_file) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                self.samples.append((row[0], row[-2], row[-1]))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_fold = os.path.join(self.img_dir, self.samples[idx][0])

        label = self.process_sentence(self.samples[idx][2], self.dictionary)

        if self.gloss:
            gloss_sent = self.process_sentence(self.samples[idx][1], self.gloss_dictionary)
            return (img_fold, gloss_sent, label)
        else:
            return (img_fold, label)

    def process_sentence(self, sentence, dictionary_ ):
        #first four words are:
        start, end, unk, pad = [dictionary_.idx2word[i] for i in range(4)]
        tok_sent = []

        #tokenization using the dictionary
        for word in sentence.split():
            if word in dictionary_.idx2word:
                tok_sent.append(dictionary_.word2idx[word])
            else:
                tok_sent.append(dictionary_.word2idx[unk])

        #now introduce the start and end tokens
        tok_sent.insert(0, dictionary_.word2idx[start])
        tok_sent.append(1) # 1 is the end token

        #padding sentence to max_seq
        max_seq = 17 if dictionary_.gloss else 27
        for i in range(max_seq - len(tok_sent)):
            tok_sent.append(dictionary_.word2idx[pad])

        return torch.LongTensor(tok_sent)

class Dictionary(object):
    def __init__(self, vocab_path='data/vocabulary.txt', gloss = False):

        self.gloss = gloss

        self.idx2word = self.read_vocab(vocab_path)
        self.word2idx = {word:i for i,word in enumerate(self.idx2word)}

    def read_vocab(self, path):
        vocabulary = []
        with open(path) as f:
            for i,line in enumerate(f):
                word = line.rstrip()
                vocabulary.append(word)

        return vocabulary

    def __len__(self):
        return len(self.idx2word)

def decode_sentence(index_sentence, dictionary):
    sentence = [dictionary.idx2word[i] for i in index_sentence]
    return sentence
    
if __name__ == '__main__':

    dataset = SNLT_Dataset(split = 'train', gloss = True)
    test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)

    print(len(dataset))

    for img_path, gsent, label in dataset:
        print('image_directory ', img_path)
        print('gloss_sentence ', len(gsent), gsent)
        print('label', len(label), label)
        break


    """
    - Removed all related with keypoints, saved locally in a Dataloader_old.py script (not pushed to the repo tho)
    - if gloss = True, output size will be 3, (img_folder, gloss sentence, label sentence), else, output will be (img_folder, label sentence)
    - todo, create a function flagged to create the vocabulary files when the dataset is instancied (?????? flagged??)
    """
