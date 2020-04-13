import transformer as tf
import argparse
import os
import DataLoader as DL
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import bleu

parser = argparse.ArgumentParser(description='PyTorch Transformer Training')
parser.add_argument('-s2t', '--sign2text', action='store_true', default = False, help = 'Train end to end model')
parser.add_argument('-e', '--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('-b', '--b_size', type=int, help='batch size', required = True)
parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to load the model')
parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default = 512)
parser.add_argument('-df', '--d_ff', type=int, help='size of feed forward representations', default = 2048)
parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default = 6)
parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default = 8)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default = 0.0)
parser.add_argument('-w', '--workers', type=int, help='number of workers to load data', default = 2)
parser.add_argument('--frames_path', type=str, default='data/tensors', help='checkpoint to load the model')
args = parser.parse_args()

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using device for training: ', device)

frames_path = args.frames_path

train_dataset = DL.SNLT_Dataset(split='train',dev=device, frames_path = frames_path , create_vocabulary = True )
dev_dataset = DL.SNLT_Dataset(split='dev', dev=device, frames_path = frames_path, create_vocabulary = True)
test_dataset = DL.SNLT_Dataset(split='test', dev=device, frames_path = frames_path, create_vocabulary = True)

model_cp = os.path.join('models','best_model') #to save the model state

if args.sign2text:
    import adapted_transformer as tf
    src_vocab = args.d_model
    print('Training end to end model')

else:
    import transformer as tf
    print('Training gloss to text model')
    src_vocab = len(train_dataset.gloss_dictionary.idx2word)

trg_vocab = len(train_dataset.dictionary.idx2word)


train_loader = DataLoader(train_dataset, batch_size=args.b_size, shuffle=True, num_workers = args.workers)
dev_loader = DataLoader(dev_dataset, batch_size=args.b_size, shuffle=True, num_workers = args.workers)
test_loader = DataLoader(test_dataset, batch_size=1)

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(src_vocab, trg_vocab, N=args.n_blocks, d_model=args.d_model, d_ff=args.d_ff, h=args.att_heads)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded state_dict to the model before starting train')

model.to(device)
model_opt = tf.NoamOpt(args.d_model, 1, 2000,
                       torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9))
if __name__ == '__main__':
    mp.set_start_method('spawn')

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

            if epoch > (args.epochs // 3) and epoch % 25 == 0:
                try:
                    bleu.score_model(model, test_loader, device, train_dataset.dictionary)
                except:
                    print('Bleu score error occurred, continuing with training')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

torch.save(train_losses, 'models/train_losses')
torch.save(dev_losses, 'models/dev_losses')
