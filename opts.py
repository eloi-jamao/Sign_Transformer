import argparse
import os


def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    parser.add_argument('--data', type=str, default=os.path.join(os.getcwd(), 'data'),
                        help='location of the data corpus')

    # parser.add_argument('--emsize', type=int, default=200,
    #                     help='size of word embeddings')
    # parser.add_argument('--nhid', type=int, default=200,
    #                     help='number of hidden units per layer')
    # parser.add_argument('--nlayers', type=int, default=2,
    #                     help='number of layers')
    # parser.add_argument('--lr', type=float, default=20,
    #                     help='initial learning rate')
    # parser.add_argument('--clip', type=float, default=0.25,
    #                     help='gradient clipping')
    # parser.add_argument('--epochs', type=int, default=40,
    #                     help='upper epoch limit')
    # parser.add_argument('--batch_size', type=int, default=20, metavar='N',
    #                     help='batch size')
    # parser.add_argument('--bptt', type=int, default=35,
    #                     help='sequence length')
    # parser.add_argument('--dropout', type=float, default=0.2,
    #                     help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    # parser.add_argument('--seed', type=int, default=1111,
    #                     help='random seed')
    # parser.add_argument('--cuda', action='store_true',
    #                     help='use CUDA')
    # parser.add_argument('--log-interval', type=int, default=200, metavar='N',
    #                     help='report interval')
    # parser.add_argument('--save', type=str, default='model.pt',
    #                     help='path to save the final model')
    # parser.add_argument('--onnx-export', type=str, default='',
    #                     help='path to export the final model in onnx format')
    # parser.add_argument('--nhead', type=int, default=2,
    #                     help='the number of heads in the encoder/decoder of the transformer model')

    # # Resnet params
    parser.add_argument('--n_classes', default=400, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')

    #
    # # Transformer params
    parser.add_argument('-e', '--epochs', type=int, default=500, help='upper epoch limit')
    parser.add_argument('-b', '--b_size', type=int, help='batch size', default=8)
    parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to load the model')
    parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default=512)
    parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default=6)
    parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default=8)
    parser.add_argument('-lr', '--learning_rate', type=float, help='number of attention heads per block', default=0.0)
    parser.add_argument('-sb', '--src_vocab', type=int, default=None)
    parser.add_argument('-tb', '--trg_vocab', type=int, default=None)
    parser.add_argument('-dff', '--d_ff', type=int, default=2048)
    parser.add_argument('-do', '--dropout', type=float, default=0.1)

    args = parser.parse_args()

    return args
