import os
from torch.utils.data import Dataset, DataLoader
import csv
from torchvision.transforms import transforms
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

        self.transform = transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.537,0.527,0.520],
                                                     std=[0.284,0.293,0.324])])



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_fold = os.path.join(self.img_dir, self.samples[idx][0])

        label = self.process_sentence(self.samples[idx][2], self.dictionary)

        if self.gloss:
            gloss_sent = self.process_sentence(self.samples[idx][1], self.gloss_dictionary)
            return (img_fold, gloss_sent, label)
        else:
            return (make_clip(img_fold), label)


    def process_sentence(self, sentence, dictionary_ ):
        #first four words are:
        pad, start, end, unk = [dictionary_.idx2word[i] for i in range(4)]
        tok_sent = []

        #tokenization using the dictionary
        for word in sentence.split():
            if word in dictionary_.idx2word:
                tok_sent.append(dictionary_.word2idx[word])
            else:
                tok_sent.append(dictionary_.word2idx[unk])

        #now introduce the start and end tokens
        tok_sent.insert(0, dictionary_.word2idx[start])
        tok_sent.append(2) # 2 is the end token

        #padding sentence to max_seq
        max_seq = 17 if dictionary_.gloss else 27
        for i in range(max_seq - len(tok_sent)):
            tok_sent.append(dictionary_.word2idx[pad])

        return torch.LongTensor(tok_sent)


    def make_clips(image_folder):

        tensors=[]
        for image in image_folder:
            img = Image.open(os.path.join(image_folder,image))
            tensor = self.transform(img).reshape(1,3,1,112,112)
            tensors.append(tensor)

        sequence = torch.cat(tensors,dim=2)
        #print(sequence.shape)
        sequence = torch.split(sequence,6,dim=2)
        #print(sequence[0].shape,sequence[1].shape)
        if sequence[-1].shape[2] < 6:
            sequenceA = torch.cat(sequence[:-1])
            #print('A',sequenceA.shape)
            sequenceB = torch.cat((sequence[-1],torch.zeros((1,3,6-sequence[-1].shape[2],112,112))),dim=2)
            #print('B',sequenceB.shape)
            sequence = torch.cat((sequenceA,sequenceB), dim = 0)
        else:
            sequence = torch.cat(sequence,dim=0)
        print(sequence.shape)

        return secuence

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

    dataset = SNLT_Dataset(split = 'train', frames_path = "/home/joaquims/dataset_tensor/", csv_path = "data/annotations/", gloss = False)
    #train_loader = DataLoader(dataset, batch_size = 4, shuffle = False)

    print(len(dataset))



    clip, label = dataset[0]

    print(clip)
      


