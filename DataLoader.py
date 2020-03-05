import os
from torch.utils.data import Dataset
#import spacy
#from spacy.vocab import Vocab
#from spacy.language import Language
import csv
import utils



class SNLT_Dataset(Dataset):
    def __init__(self, train = False):

        #Read the CSV annotation file and creates a list with (Keypoint path, traduction)
        self.samples = []
        self.cwd = os.getcwd()
        self.train = train

        if self.train:
            self.keypoints_dir = "/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/keypoints/train/"
            self.csv_path = "/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/train_annotations"
        else:
            self.keypoints_dir = "./data/keypoints/test/" #"/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/keypoints/test/"
            self.csv_path = "./data/annotations/test_annotations.csv"   #"/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/test_annotations.csv"

        with open(self.csv_path) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                self.samples.append((row[0], row[-1]))
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        keypoint_sentence = []
        #search the keypoints json 
        keypoint_dir = os.path.join(self.keypoints_dir, self.samples[idx][0])
        for json_file in os.listdir(keypoint_dir):
            #convert into a list
            keypoint = utils.json2keypoints(os.path.join(keypoint_dir, json_file))

            #padding the keypoint
            utils.kp_padding(keypoint, max_len = 475)
            keypoint_sentence.append(keypoint)

        #label one hot
        label = self.samples[idx][1]

        return (keypoint_sentence, label)

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
    for i in range(1):
        print(len(dataset[i][0]), len(dataset[i][0][0]), dataset[i][1] )
    """
    '''This assumes you have a file named vocaulary.txt in the data folder with all the tokens,
    you can get it from https://github.com/neccam/nslt/blob/master/Data/phoenix2014T.vocab.de#L1 '''
    with open(root + '/data/vocabulary.txt') as f:
        vocab2int = {}
        int2vocab = {}
        for i,row in enumerate(f):
            row = row.rstrip()
            vocab2int[row] = i
            int2vocab[i] = row


    vocab = Vocab(strings = int2vocab.values())
    nlp = Language(vocab = vocab)

    #print(f'Custom language created with {len(nlp.vocab)} vocabulary size')
    '''Loading text'''
    text_file_path = root + '/data/test_de.txt'
    print_process(translation) #Uncomment to print results
    """