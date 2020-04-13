import os
from torch.utils.data import Dataset, DataLoader
import csv
from torchvision.transforms import transforms
#import utils
import torch.nn.functional as F
import torch
from PIL import Image
import time
from multiprocessing import Pool


class SNLT_Dataset(Dataset):
    def __init__(self, split, dev = 'cpu',
                 frames_path = "data/frames/",
                 csv_path = "data/annotations/",
                 gloss = False,
                 create_vocabulary = False,
                 long_clips = 6, window_clips = 2):

        splits = ['train','test','dev']
        self.split = split
        if self.split not in splits:
            raise Exception('Split must be one of:', splits)

        self.samples = []
        self.gloss = gloss
        self.device = dev
        #Paths
        self.img_dir = frames_path #+ self.split
        self.csv_path = csv_path
        self.csv_file = csv_path + self.split + ".csv"

        #Clips
        self.long_clips = long_clips
        self.window_clips = window_clips

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
        #print(img_fold, type(img_fold))

        label = self.process_sentence(self.samples[idx][2], self.dictionary)

        if self.gloss:
            gloss_sent = self.process_sentence(self.samples[idx][1], self.gloss_dictionary)
            return (img_fold, gloss_sent, label)
        else:
            #clips = self.make_clips(img_fold, self.long_clips, self.window_clips)
            return (torch.load(img_fold), label)


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


    def make_clips(self, image_folder, long, window, max_len = 60):

        tensors=[]
        window_list = []
        i = 0
        #print(len(os.listdir(image_folder)))
        #with Pool(8) as p:
            #tensors = p.map(self.openimage, [ image for image in os.listdir(image_folder)], tensors)

        for image in os.listdir(image_folder):
            i += 1
            img = Image.open(os.path.join(image_folder,image))
            tensor = self.transform(img).reshape(1,3,1,112,112)
            #tensor.type(dtype=torch.int32)

            tensors.append(tensor)
            if long >= i and i > long-window:
                window_list.append(tensor)
            elif i > long:
                #print(len(window_list))
                tensors.extend(window_list)
                window_list = []
                i = 0

        #print(len(tensors))
        sequence = torch.cat(tensors,dim=2)
        #print(sequence.shape)
        sequence = torch.split(sequence, long, dim=2)
        #print(sequence[0].shape,sequence[1].shape)
        if sequence[-1].shape[2] < long:
            sequenceA = torch.cat(sequence[:-1])
            #print('A',sequenceA.shape)
            sequenceB = torch.cat((sequence[-1],torch.zeros((1, 3,long-sequence[-1].shape[2],112,112))),dim=2)
            #print('B',sequenceB.shape)
            sequence = torch.cat((sequenceA,sequenceB), dim = 0)
        else:
            sequence = torch.cat(sequence,dim=0)
        #print(sequence.shape)
        sequence = torch.cat((sequence,torch.zeros((max_len-sequence.size()[0],3,6,112,112))), dim = 0)
        return sequence

    def openimage(self, image, tensors):
        img = Image.open(os.path.join(image_folder,image))
        tensor = self.transform(img).reshape(1,3,1,112,112)
        tensors.append(tensor)

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
        
class Custom_iterator(object):
    def __init__(self, dataset, batch_size, num_workers = 1, shuff = True):
        self.dataset=dataset
        self.workers = num_workers
        self.batch_size = batch_size
        self.shuff = shuffle

    def mp_iterate(self):

        index = [i for i in range(len(dataset))]
        if shuff:
            shuffle(index)
        for i in range(0,batch_size*(len(index)//batch_size),batch_size):
            indexes = index[i:i+batch_size]
            #print(indexes)

            output = []
            q = mp.Queue()
            processes = []
            for w in range(num_workers):
                a = w*(batch_size//num_workers)
                b = a + (batch_size//num_workers)
                #print(a,b)
                chunk = indexes[a:b]
                proc = mp.Process(target=load_chunk, args=(dataset,chunk,q, num_workers))
                processes.append(proc)
                proc.start()
            for proc in processes:
                proc.join()
            #q.close()
            #time.sleep(2)
            while not q.empty():
                chunk = q.get()
                output.append(chunk)
            clips = torch.cat([chunk[0] for chunk in output], dim=0)
            target = torch.cat([chunk[1] for chunk in output], dim=0)
            yield clips, target

    def pool_iterate(self):
        index = [i for i in range(len(self.dataset))]
        if self.shuff:
            shuffle(index)
        for i in range(0,self.batch_size*(len(index)//self.batch_size),self.batch_size):
            indexes = index[i:i+self.batch_size]
            #print(indexes)
            #print('starting iterator')
            chunks = []
            for w in range(self.workers):
                a = w*(self.batch_size//self.workers)
                b = a + (self.batch_size//self.workers)
                chunks.append(indexes[a:b])
            with concurrent.futures.ProcessPoolExecutor(max_workers = self.workers) as executor:
                samples = executor.map(self.pool_chunks, chunks)
                samples = [sample for sample in samples]
                clips = torch.cat([sample[0] for sample in samples], dim=0)
                targets = torch.cat([sample[1] for sample in samples], dim=0)
            yield clips, targets


    def pool_chunks(self, chunk):
        #print('starting worker')
        samples = [self.dataset[i] for i in chunk]
        clips = torch.stack([sample[0] for sample in samples], dim=0)
        targets = torch.stack([sample[1] for sample in samples],dim=0)
        return clips,targets


    def load_chunk(self, chunk, queue):
        samples = [self.dataset[i] for i in chunk]
        #print(type(samples[0][0]),samples[0][0].shape)
        clips = torch.stack([sample[0] for sample in samples], dim=0)
        targets = torch.stack([sample[1] for sample in samples],dim=0)
        #print(clips.shape, targets.shape)
        queue.put((clips, targets))
        queue.close()
        time.sleep(2)
        #while queue.qsize() != self.workers:
            #time.sleep(.5)

def decode_sentence(index_sentence, dictionary):
    sentence = [dictionary.idx2word[i] for i in index_sentence]
    return sentence

if __name__ == '__main__':

    dataset = SNLT_Dataset(split = 'train', frames_path = "/home/joaquims/dataset_ssd/train_L6_W2/", csv_path = "data/annotations/", gloss = False)
    loader  = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 3)
    start = time.time()
    i = 0
    for video, target in loader:
        print(video.size())
        print(time.time()-start)
        start = time.time()
        i+=1
        if i==6:
            break
