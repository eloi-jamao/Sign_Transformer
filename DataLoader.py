import os
from torch.utils.data import Dataset
import spacy
import csv
import utils



class SNLT_Dataset(Dataset):
    def __init__(self, train = False, padding = 475):

        #Read the CSV annotation file and creates a list with (Keypoint path, translation)
        self.samples = []
        self.cwd = os.getcwd()
        self.train = train
        self.padding = padding
        self.dictionary = Dictionary()

        if self.train:
            self.kp_dir = "/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/keypoints/train/"
            self.csv_path = "/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/train_annotations"
        else:
            self.kp_dir = "./data/keypoints/test/" #"/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/keypoints/test/"
            self.csv_path = "./data/annotations/test_annotations.csv"   #"/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/test_annotations.csv"

        with open(self.csv_path) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                self.samples.append((row[0], row[-1]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        kp_sentence = self.process_kps(self.samples[idx][0])
        label = self.process_sentence(self.samples[idx][1])

        return (kp_sentence, label)

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

        return kp_sentence

    def process_sentence(self, sentence):
        tok_sent = [self.dictionary.word2idx[word] for word in sentence.split()]
        #now introduce the start and end tokens
        start, end, unk, pad = [self.dictionary.idx2word[i] for i in range(4)]
        tok_sent.insert(0,self.dictionary.word2idx[start])
        tok_sent.append(self.dictionary.word2idx[end])
        #also might need to pad or add unknown tokens where necessary
        return tok_sent

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


def sentence_preprocessing(sentence):
    tokens = [token.text for token in nlp.tokenizer(sentence)]
    tokens = [vocab2int[token] for token in tokens]
    tokens.insert(0,1)
    tokens.append(2)
    return tokens


def decode_sentence(sentence):
    sentence = [int2vocab[integer] for integer in sentence]
    return sentence


def print_process(translation):
    print(translation)
    translation = sentence_preprocessing(translation)
    print(translation)
    translation = decode_sentence(translation)
    print(translation)


if __name__ == '__main__':

    dataset = SNLT_Dataset(train = False)

    for i in range(5):
        print(len(dataset[i][0]), len(dataset[i][0][0]), dataset[i][1] )
