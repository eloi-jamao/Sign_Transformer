import model_end2end
import argparse
import os
import DataLoader as DL
import torch
from torch.utils.data import DataLoader
from dataset_end2end import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
import model_end2end as tf
from resnet3d import resnet34
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Transformer Training')
parser.add_argument('-e', '--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('-b', '--b_size', type=int, help='batch size', required = True)
parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to load the model')
parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default = 512)
parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default = 6)
parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default = 8)
parser.add_argument('-lr', '--learning_rate', type=float, help='number of attention heads per block', default = 0.0)
parser.add_argument('-d', '--sample_duration', type=float, help='sample duration', default = 4)
parser.add_argument('-s', '--sample_size', type=float, help='sample duration', default = 128)
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using device for training:', device)

batch_size = args.b_size
epochs = args.epochs
N_blocks = args.n_blocks
d_model = args.d_model
att_heads = args.att_heads
lr = args.learning_rate
sample_duration = args.sample_duration
sample_size = args.sample_size

root_dir = "data"
model_dir = "model"
mean = [114.7748, 107.7354, 99.4750]

spatial_transform = Compose([Scale(sample_size),#really needed?
                             CenterCrop(sample_size),
                             ToTensor(),
                             Normalize(mean, [1, 1, 1])])
temporal_transform = LoopPadding(sample_duration)

data_train = Video(os.path.join(root_dir, "train"),
                   os.path.join(root_dir, "annotations_train.csv"),
                   spatial_transform=spatial_transform,
                   temporal_transform=temporal_transform,
                   sample_duration=sample_duration)

src_vocab = 142  # dummy value, to be computed based on video length distribution
trg_vocab = len(data_train.dictionary.idx2word)  # dummy value
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=batch_size, shuffle=True)


data_dev = Video(os.path.join(root_dir, "dev"),
                 os.path.join(root_dir, "annotations_dev.csv"),
                 spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=sample_duration)

dev_loader = torch.utils.data.DataLoader(
    data_dev, batch_size=batch_size, shuffle=False)

# load pretrained model
cnn3d = resnet34(sample_size=sample_size,
                 sample_duration=sample_duration,
                 shortcut_type="A")
model_data = torch.load("pretrained/resnet-34-kinetics.pth", map_location=torch.device(device))

state_dict = {}
for key, value in model_data['state_dict'].items():
    key = key.replace("module.", "")
    state_dict[key] = value
cnn3d.load_state_dict(state_dict)
for i, child in enumerate(cnn3d.children()):
    if i < 7 or i == 9:
        for p in child.parameters():
            p.requires_grad = False

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(cnn3d, src_vocab, trg_vocab, N=N_blocks, d_model=d_model, h=att_heads)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded state_dict to the model before starting train')

model.to(device)
model_opt = tf.NoamOpt(d_model, 1, 2000,
                       # here add cnn parameters
                       torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))

train_losses = []
dev_losses = []
best_loss = None

try:
    for epoch in range(epochs):
        print('Starting epoch', epoch)

        model.train()
        train_loss = tf.run_epoch(train_loader, model,
                                  tf.SimpleLossCompute(model.generator, criterion, model_opt),
                                  device)
        print(train_loss)
        model.eval()
        dev_loss = tf.run_epoch(dev_loader, model,
                                tf.SimpleLossCompute(model.generator, criterion, None),
                                device)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        if not best_loss or (dev_loss < best_loss):
            torch.save(model.state_dict(), os.path.join(model_dir, f'cp_epoch_{epoch}'))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
torch.save(train_losses, os.path.join(model_dir, "loss/train"))
torch.save(dev_losses, os.path.join(model_dir, "loss/dev"))
plt.plot(train_losses, "r")
plt.plot(dev_losses, "b")
plt.show()
