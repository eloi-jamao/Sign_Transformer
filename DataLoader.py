import os
from torch.utils.data import Dataset, DataLoader
import spacy
import csv
import utils
import torch
from PIL import Image

class SNLT_Dataset(Dataset):
    def __init__(self, split, frames_path, csv_path, gloss = False, padding = 475, kp_path = None):

        splits = ['train','test','dev']
        self.split = split
        if self.split not in splits:
            raise Exception('Split must be one of:', splits)
        #Read the CSV annotation file and creates a list with (Keypoint path, translation)
        self.samples = []
        self.cwd = os.getcwd()
        self.padding = padding
        self.dictionary = Dictionary()
        self.img_dir = os.path.join(frames_path, split)
        #self.kp_dir = os.path.join(kp_path, split)
        self.csv_file = csv_path + '/PHOENIX-2014-T.' + self.split + '.corpus.csv'

        with open(self.csv_file) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            col = -2 if gloss else -1
            for row in csv_reader:
                self.samples.append((row[0], row[col]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_fold = os.path.join(self.img_dir, self.samples[idx][0])
        label = self.process_sentence(self.samples[idx][1])

        return (img_fold, label)


    def process_sentence(self, sentence):

        start, end, unk, pad = [self.dictionary.idx2word[i] for i in range(4)]
        tok_sent = []
        for word in sentence.split():
            if word in self.dictionary.idx2word:
                tok_sent.append(self.dictionary.word2idx[word])
            else:
                tok_sent.append(self.dictionary.word2idx[unk])
        #print('sentence', sentence)
        #print('token sentence', tok_sent)
        #now introduce the start and end tokens
        tok_sent.insert(0,self.dictionary.word2idx[start])
        tok_sent.append(1) # 1 is the end token
        #padding sentence to max_seq
        for i in range(35 - len(tok_sent)):
            tok_sent.append(self.dictionary.word2idx[pad])
        return torch.LongTensor(tok_sent)

    '''
    def process_kps(self, kp_folder):

        kp_sentence = []
        #search the keypoints json
        for json_file in os.listdir(os.path.join(self.kp_dir, kp_folder)):
            #convert into a list
            kp = utils.json2keypoints(os.path.join(self.kp_dir, kp_folder, json_file))
            kp_sentence.append(kp)

        #add padding
        for i in range(self.padding-len(kp_sentence)):
        	kp_sentence.append([0 for x in range(len(kp_sentence[0]))])

        return torch.FloatTensor(kp_sentence)
    '''

class Dictionary(object):
    def __init__(self, vocab_path='./data/vocabulary.txt'):

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


def describe_row(name,translation):
    print(f'Data from iterator: \n Video name of type {type(name)}: {name}\n Translation of type {type(translation)}: {translation}')

def decode_sentence(sentence):
    pass


if __name__ == '__main__':

    frames_path = './../Sign_Transformer1/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px' #change to your paths
    csv_path = './../Sign_Transformer1/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

    dataset = SNLT_Dataset(split = 'test', frames_path = frames_path, csv_path = csv_path, gloss = False)
    test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)


    for img_path, sentence in test_loader:
        print('image_directory', img_path)
        print('sentence', sentence)
        break



#'./../Sign_Transformer1/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
