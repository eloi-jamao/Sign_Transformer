import transformer as tf
import argparse
import os
import DataLoader as DL
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Transformer Training')
parser.add_argument('--epochs', '-e', type=int, default=50, help='upper epoch limit')
parser.add_argument('-b_size', '-b', type=int, help='batch size')
parser.add_argument('--checkpoint', '-cp', type=str, default=None, help='checkpoint to load the model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using device for training:', device)

model_dir = './models/G2T' #folder to save the model state

train_dataset = DL.SNLT_Dataset(split='train', gloss = True)
dev_dataset = DL.SNLT_Dataset(split='dev', gloss = True)

src_vocab = len(train_dataset.gloss_dictionary.idx2word)
trg_vocab = len(train_dataset.dictionary.idx2word)
print('Gloss vocab of',src_vocab,'german vocab of',trg_vocab)

batch_size = args.b_size
epochs = args.epochs
N_blocks = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(src_vocab, trg_vocab, N=N_blocks, d_model=512, h=8)
if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded state_dict to the model before starting train')
model.to(device)
model_opt = tf.NoamOpt(model.src_embed[0].d_model, 1, 400,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

train_losses = []
dev_losses = []
best_loss = None

try:
    for epoch in range(epochs):
        print('Starting epoch', epoch)

        model.train()
        train_loss = tf.run_epoch(train_loader, model,
                                  tf.SimpleLossCompute(model.generator, criterion, model_opt), device)

        model.eval()
        dev_loss = tf.run_epoch(dev_loader, model,
                                  tf.SimpleLossCompute(model.generator, criterion, None), device)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        if not best_loss or dev_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f'cp_epoch_{epoch}'))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

torch.save(train_losses, 'models/G2T/train_losses')
torch.save(dev_losses, 'models/G2T/dev_losses')
