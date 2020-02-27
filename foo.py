import model
import utils
import os
import torch
root = os.getcwd()
videos_folder = root + '/data/How2Sign_samples 2/How2Sign_samples/openpose_output/json'
videos = os.listdir(videos_folder)
keypoints_files = os.listdir(videos_folder + '/' + videos[0])
frames_seq = []
for i in range(3):
    frame_keypoints = utils.json2keypoints(videos_folder + '/' + videos[0] +'/' + keypoints_files[i])
    frames_seq.append(frame_keypoints)

frames_seq = torch.tensor(frames_seq, dtype = torch.float)
'adding batch size'
frames_seq = frames_seq.unsqueeze(dim=1)
print(frames_seq.shape)

num_keypoints = 274
emb_size = 200
""""In our project ntoken = number of keypoints because that is our input, not words"""

model = model.TransformerModel(
                               ntoken = num_keypoints,
                               ninp = emb_size,
                               nhead = 2,
                               nhid = 100,
                               nlayers = 2,
                               dropout = 0.2
                               )
output = model(frames_seq)
print(output.shape)
output = output.squeeze(dim=1)
print(output[0])
'''
For now the model is designed for language modeling,
so we still have some things to change to get a reasonable output.
But for now, i have changed the first layer of the encoder to be a
nn.Linear instead of nn.embedding because our input won't be words
'''
