# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import os

from video_loader import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
from slt_transformer import evaluate_model, make_model
from resnet3d import resnet34


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length

def reference_corpus(loader, dictionary, references):
    """ speed up so that frames are not loaded unnecessarily """
    test_corpus = []
    for sentence in references:
        sentence.squeeze_(dim=0)
        sent = []
        for i in sentence:
            if i == 2:
                break
            else:
                sent.append(i)
        test_corpus.append([[dictionary.idx2word[i] for i in sent[1:]]])
        #break
    return test_corpus

def write_corpus(corpus, path):
    with open(path, 'w') as f:
        for line in corpus:
            line = ' '.join(line)+'.\n'
            f.write(line)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

if __name__ == '__main__':

    import DataLoader as DL
    import torch
    from torch.utils.data import DataLoader
    import argparse
    parser = argparse.ArgumentParser(description='Bleu score')
    parser.add_argument('-m', '--model', type=str, help='path to model file', default=None)
    parser.add_argument('-dm', '--d_model', type=int, help='size of intermediate representations', default = 512)
    parser.add_argument('-df', '--d_ff', type=int, help='size of feed forward representations', default = 2048)
    parser.add_argument('-n', '--n_blocks', type=int, help='number of blocks for the encoder and decoder', default = 6)
    parser.add_argument('-at', '--att_heads', type=int, help='number of attention heads per block', default = 8)
    parser.add_argument('-d', '--sample_duration', type=float, help='sample duration', default = 4)
    parser.add_argument('-s', '--sample_size', type=float, help='sample size', default = 128)
    args = parser.parse_args()

    root_dir = "../data"

    model_cp = args.model
    N_blocks = args.n_blocks
    d_model = args.d_model
    d_ff = args.d_ff
    att_heads = args.att_heads
    sample_duration = args.sample_duration
    sample_size = args.sample_size

    mean = [114.7748, 107.7354, 99.4750]

    spatial_transform = Compose([Scale(sample_size),#really needed?
                                 CenterCrop(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data_test = Video(os.path.join(root_dir, "train"),
                      "annotations_train.csv",
                      spatial_transform=spatial_transform,
                      temporal_transform=temporal_transform,
                      sample_duration=sample_duration)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=1, shuffle=False)

    trg_vocab = len(data_test.dictionary.idx2word)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    cnn3d = resnet34(sample_size=sample_size,
                     sample_duration=sample_duration,
                     shortcut_type="A")
    model = make_model(cnn3d, trg_vocab, N=N_blocks, d_model=d_model, d_ff=d_ff, h=att_heads)
    model.load_state_dict(torch.load(model_cp, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    print('Generating corpus with model...')
    pred_corpus, references = evaluate_model(model,
                                 test_loader,
                                 device,
                                 max_seq=32,
                                 dictionary=data_test.dictionary)

    print('Loading reference corpus...')
    test_corpus = reference_corpus(test_loader, data_test.dictionary, references)

    print('------Example sample-------')
    for i in range(10):
        print('Reference sentence:\n',test_corpus[i][0],'\nGenerated equivalent:\n',pred_corpus[i])

    for n in [1,2,3,4]:
        results = compute_bleu(test_corpus, pred_corpus, max_order=n, smooth=True)
        print('Bleu score with n_grams =',n, ':',results[0])
