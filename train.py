import transformer as tf
import argparse
import SNLT_Dataset
import resource
import os
import SNLT_Dataset as DL
import torch
from torch.utils.data import DataLoader
# import torch.multiprocessing as mp
import time
import opts
from torchvision.transforms import transforms

args = opts.parse_training_opts()

device = args.device if torch.cuda.is_available() else 'cpu'
print('Using device for training: ', device)

frames_path = os.path.join(args.root, args.path_frames)
annotations_path = os.path.join(args.root, args.path_annotations)

frame_size = args.frame_size
transform = transforms.Compose([transforms.Resize((frame_size, frame_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.537, 0.527, 0.520],
                                                     std=[0.284, 0.293, 0.324])])

train_dataset = DL.SNLT_Dataset(split='train', dev=device, frames_path=frames_path, csv_path=annotations_path,
                                long_clips=args.clips_long, window_clips=args.clips_overlap, transform=transform)
dev_dataset = DL.SNLT_Dataset(split='dev', dev=device, frames_path=frames_path, csv_path=annotations_path,
                              long_clips=args.clips_long, window_clips=args.clips_overlap, transform=transform)
test_dataset = DL.SNLT_Dataset(split='test', dev=device, frames_path=frames_path, csv_path=annotations_path,
                               long_clips=args.clips_long, window_clips=args.clips_overlap, transform=transform)

model_cp = os.path.join(args.root, args.path_state) #to save the model state


if args.end2end:
    import transformer_adapted as tf
    src_vocab = 128
    print('Training end to end model')

else:
    import transformer as tf
    print('Training gloss to text model')
    src_vocab = len(train_dataset.gloss_dictionary.idx2word)

trg_vocab = len(train_dataset.dictionary.idx2word)


train_loader = DataLoader(train_dataset, batch_size=args.b_size, shuffle=True, num_workers=args.workers)
dev_loader = DataLoader(dev_dataset, batch_size=args.b_size, shuffle=True, num_workers=args.workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

criterion = tf.LabelSmoothing(size=trg_vocab, padding_idx=0, smoothing=0.0)
model = tf.make_model(trg_vocab, N=args.n_blocks, d_model=args.d_model, d_ff=args.d_ff, h=args.att_heads)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded state_dict to the model before starting train')

model.to(device)
model_opt = tf.NoamOpt(args.d_model, 1, 2000,
                       torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9))

if __name__ == '__main__':
    # mp.set_start_method('spawn')

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
