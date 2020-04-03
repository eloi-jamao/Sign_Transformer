import torch
from torchvision.transforms import transforms
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Script to transform videos to batched tensors of clips')
parser.add_argument('-in', '--input', type=str, help='path to folder with subsequent frames folders')
parser.add_argument('-out', '--output', type=str, help='Empty folder to store the tensors')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

transform = transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.537,0.527,0.520],
                                                     std=[0.284,0.293,0.324])])

path = args.input #path to the folder with all directories containing images
out = args.output #path to the folder where we save all the tensors
i=0
with torch.cuda.device(device):
    for folder in os.listdir(path):
        tensors=[]
        for image in os.listdir(os.path.join(path,folder)):
            img = Image.open(os.path.join(path,folder,image))
            tensor = transform(img).reshape(1,3,1,112,112)
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
        #print(sequence.shape)
        torch.save(sequence, os.path.join(out,folder))
        if i % 100 == 0:
            print('Processing...',i,'folders done')
        i+=1
