import argparse
import os


def parse_training_opts():

    parser = argparse.ArgumentParser(description='PyTorch Transformer Training')

    # Folders
    parser.add_argument('-r', '--root', type=str, help='data root folder', default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument('-pf', '--path_frames', type=str, help='frames folder', default='features')
    parser.add_argument('-pa', '--path_annotations', type=str, help='annotations path', default='annotations')
    parser.add_argument('-ps', '--path_state', type=str, help='state folder', default=os.path.join('models', 'G2T', 'best_model'))

    parser.add_argument('-dv', '--device', type=str, help='device', default='cuda')
    parser.add_argument('-e2e', '--end2end', action='store_true', default=True, help='Train end to end model')

    parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to load the model')
    # Dimensions
    parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default=128)
    parser.add_argument('-df', '--d_ff', type=int, help='size of feed forward representations', default=512)
    parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default=2)
    parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default=8)
    # Model params
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', default=0.0)
    parser.add_argument('-w', '--workers', type=int, help='learning rate', default=2)
    parser.add_argument('-b', '--b_size', type=int, help='batch size', default=32)
    # Data params
    parser.add_argument('-cl', '--clips_long', type=int, help='clips size', default=6)
    parser.add_argument('-co', '--clips_overlap', type=int, help='overlap size', default=2)
    parser.add_argument('-fs', '--frame_size', type=int, help='frame size', default=112)

    parser.add_argument('-e', '--epochs', type=int, default=500, help='upper epoch limit')


    args = parser.parse_args()

    return args
