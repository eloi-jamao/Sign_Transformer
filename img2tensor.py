import torch
from torchvision.transforms import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.537,0.527,0.520],
                                                     std=[0.284,0.293,0.324])])

path = 'data/frames' #path to the folder with all directories containing images
out = 'data/tensors' #path to the folder where we save all the tensors

for folder in os.listdir(path):
    tensors=[]
    for image in os.listdir(os.path.join(path,folder)):
        img = Image.open(os.path.join(path,folder,image))
        tensor = transform(img).reshape(1,3,1,112,112)
        tensors.append(tensor)


    sequence = torch.cat(tensors,dim=2)
    sequence = torch.split(sequence,6,dim=2)
    sequence = torch.cat(sequence,dim=0)
    os.mkdir(os.path.join(out,folder))
    torch.save(sequence, os.path.join(out,folder,folder))
