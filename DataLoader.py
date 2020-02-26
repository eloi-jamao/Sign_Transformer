import os
import json
import spacy
from spacy.vocab import Vocab
from spacy.language import Language
import csv

root = os.getcwd()

'''CSV'''
csv_folder = root + '/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/'

def get_samples(csv_file):
    '''This function yields the name of a video and
    the translation in german as an iterator'''
    first_line = True
    with open(csv_file) as file:
        csv_reader = csv.reader(file, delimiter='|')
        next(csv_reader)
        for row in csv_reader:
            yield row[0], row[-1]

def describe_row(name,translation):
    print(f'Data from iterator: \n Video name of type {type(name)}: {name}\n Translation of type {type(translation)}: {translation}')

'''JSON'''
def read_json(filename):
    '''This function takes the json file from openpose
    and returns a concatenated list of all keypoints'''
    keys = ['pose_keypoints_2d','face_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']
    with open(filename) as json_file:
        data = json.load(json_file)
        people_dict = data['people']
        keypoints_dict = people_dict[0]
        keypoints = []
        for key in keys:
            points = keypoints_dict[key]
            keypoints += points
    return keypoints

'''TEXT'''

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
print(f'Custom language created with {len(nlp.vocab)} vocabulary size')

def sentence_preprocessing(sentence):
    tokens = [token.text for token in nlp.tokenizer(sentence)]
    tokens = [vocab2int[str(token)] for token in tokens]
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

    '''Loading CSV'''
    csv_test = csv_folder + 'PHOENIX-2014-T.test.corpus.csv'
    data_iter = iter(get_samples(csv_test))
    for i in range(3):
        video, translation = next(data_iter)
        #describe_row(video, translation) #Uncomment to print results



    '''Loading keypoints'''
    videos_folder = root + '/data/How2Sign_samples 2/How2Sign_samples/openpose_output/json'
    videos = os.listdir(videos_folder)
    keypoints_files = os.listdir(videos_folder + '/' + videos[0])

    for i in range(2):
        frame_keypoints = read_json(videos_folder + '/' + videos[0] +'/' + keypoints_files[i])
        #print(f'Frame {i} with {len(frame_keypoints)} keypoints and type {type(frame_keypoints)}') #Uncomment to print results


    '''Loading text'''
    text_file_path = root + '/data/test_de.txt'
    print_process(translation) #Uncomment to print results
