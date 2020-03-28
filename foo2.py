import transformer as tf
import argparse
import os
import DataLoader as DL
import torch
from torch.utils.data import DataLoader
import opts
import seq2seq


args = opts.parse_opts()

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
args.src_vocab = src_vocab
args.trg_vocab = trg_vocab
print('Gloss vocab of',src_vocab,'german vocab of',trg_vocab)

train_loader = DataLoader(train_dataset, batch_size=args.b_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.b_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = seq2seq.Seq2Seq(args)
# model = tf.make_model(src_vocab, trg_vocab, N=N_blocks, d_model=d_model, h= att_heads)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded state_dict to the model before starting train')

# model.to(device)
model_opt = tf.NoamOpt(model.src_embed[0].d_model, 1, 2000,
                       torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9))

train_losses = []
dev_losses = []
best_loss = None

try:
    for epoch in range(args.epochs):
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
