import transformer as tf
import argparse
import os
import DataLoader as DL
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Transformer Training')
parser.add_argument('-e', '--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('-b', '--b_size', type=int, help='batch size', required = True)
parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to load the model')
parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default = 512)
parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default = 6)
parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default = 8)
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using device for training:', device)

model_cp = './models/G2T/best_model' #to save the model state

train_dataset = DL.SNLT_Dataset(split='train', gloss = True)
dev_dataset = DL.SNLT_Dataset(split='dev', gloss = True)
test_dataset = DL.SNLT_Dataset(split='test', gloss = True)

src_vocab = len(train_dataset.gloss_dictionary.idx2word)
trg_vocab = len(train_dataset.dictionary.idx2word)
print('Gloss vocab of',src_vocab,'german vocab of',trg_vocab)

batch_size = args.b_size
epochs = args.epochs
N_blocks = args.n_blocks
d_model = args.d_model
att_heads = args.att_heads

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(src_vocab, trg_vocab, N=N_blocks, d_model=d_model, h= att_heads)

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
                                  tf.SimpleLossCompute(model.generator, criterion, model_opt),
                                  device)

        model.eval()
        dev_loss = tf.run_epoch(dev_loader, model,
                                  tf.SimpleLossCompute(model.generator, criterion, None),
                                  device)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        if not best_loss or (dev_loss < best_loss):
            torch.save(model.state_dict(), model_cp)

        torch.save(train_losses, 'models/G2T/train_losses')
        torch.save(dev_losses, 'models/G2T/dev_losses')

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


model.load_state_dict(torch.load(model_cp, map_location=torch.device(device)))
model.eval()

tf.evaluate_model(model,
                  test_loader,
                  device,
                  max_seq = 27,
                  dictionary = train_dataset.dictionary)
