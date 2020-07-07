import os
import argparse
import re
from collections import namedtuple
import nltk
import numpy as np
import torch
import random
import math

from xml.etree import ElementTree
from latent_rationale.common.util import get_alphas
from latent_rationale.agnews.constants import UNK_TOKEN, PAD_TOKEN
from latent_rationale.agnews.plotting import plot_heatmap
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn


def bag_of_probas(words, probas, vocab_size, mask=None):
    # set probas=1.0 for bag of words
    bs, ts = words.shape
    device = words.device
    bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()
    bids = bids.to(device)
    idxs = torch.stack((bids, words.flatten()), dim=0)
    mask = torch.ones_like(words) if mask is None else mask
    vals = mask.to(device).float() * probas
    vals = vals.flatten()
    size = torch.Size([bs, vocab_size])
    bow = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
    bow = bow.to(device)
    return bow


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('model'):
            return os.path.join(path, f)
    print("Could not find ckpt in {}".format(path))


def filereader(path):
    """read file lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.findall(r"\([0-9] ([^\(\)]+)\)", s)


def token_labels_from_treestring(s):
    """extract token labels from sentiment tree"""
    return list(map(int, re.findall(r"\(([0-9]) [^\(\)]", s)))


Example = namedtuple("Example", ["tokens", "label"])


def agnews_reader(path, lower=False):
    """
    Reads in examples
    :param path:
    :param lower:
    :return:
    """
    tokenizer = nltk.WordPunctTokenizer()
    root = ElementTree.parse(path).getroot()
    categories = [x.text for x in root.iter('category')]
    descriptions = [x.text for x in root.iter('description')]
    l2i = {'Business': 0, 'World': 1}
    for text, label in zip(descriptions, categories):
        if text is None or label is None:
            continue
        # business vs world (binary classification)
        if label not in ['Business', 'World']:
            continue
        text = text.lower() if lower else text
        # text = text.replace("\\", "")
        text = re.sub("\\\\", "", text)  # fix escape
        tokens = tokenizer.tokenize(text)
        label = l2i[label]
        yield Example(tokens=tokens, label=label)


def print_parameters(model):
    """Prints model parameters"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))


def load_glove(glove_path, vocab, glove_dim=300):
    """
    Load Glove embeddings and update vocab.
    :param glove_path:
    :param vocab:
    :param glove_dim:
    :return:
    """
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, glove_dim).astype(np.float32))
    w2i[UNK_TOKEN] = 0
    i2w.append(UNK_TOKEN)

    # Zero vector for padding
    vectors.append(np.zeros(glove_dim).astype(np.float32))
    w2i[PAD_TOKEN] = 1
    i2w.append(PAD_TOKEN)

    # vectors should be a dict mapping str keys to numpy arrays
    import pickle
    with open(glove_path, 'rb') as f:
        d = pickle.load(f)
        tokens = list(d.keys())
        vecs = list(d.values())
        for word in tokens:
            w2i[word] = len(w2i)
        i2w.extend(tokens)
        vectors.extend(vecs)

    # fix brackets
    w2i[u'-LRB-'] = w2i.pop(u'(')
    w2i[u'-RRB-'] = w2i.pop(u')')

    i2w[w2i[u'-LRB-']] = u'-LRB-'
    i2w[w2i[u'-RRB-']] = u'-RRB-'

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)


def get_minibatch(data, batch_size=25, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    reverse_map = None
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.label for ex in mb]

    x = np.array(x)
    y = np.array(y)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # to put back into the original order
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map


def plot_dataset(model, data, batch_size=100, device=None, save_path=".",
                 ext="pdf"):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout
    sent_id = 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)

        with torch.no_grad():
            logits = model(x)

            alphas = model.alphas if hasattr(model, "alphas") else None
            z = model.z if hasattr(model, "z") else None

        # reverse sort
        alphas = alphas[reverse_map] if alphas is not None else None
        z = z.squeeze(1).squeeze(-1)  # make [B, T]
        z = z[reverse_map] if z is not None else None

        for i, ex in enumerate(mb):
            tokens = ex.tokens

            if alphas is not None:
                alpha = alphas[i][:len(tokens)]
                alpha = alpha[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.alphas.{}".format(sent_id, ext))
                plot_heatmap(alpha, column_labels=tokens, output_path=path)

            # print(tokens)
            # print(" ".join(["%4.2f" % x for x in alpha]))

            # z is [batch_size, num_samples, time]
            if z is not None:

                zi = z[i, :len(tokens)]
                zi = zi[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.z.{}".format(sent_id, ext))
                plot_heatmap(zi, column_labels=tokens, output_path=path)

            sent_id += 1


def expl_dataset(classifier, layperson, data, batch_size=100, device=None, save_path="."):
    """Save explanations of a model on given data set (using minibatches)"""

    classifier.eval()  # disable dropout
    layperson.eval()

    sent_id = 0
    vocab_size = len(layperson.vocab.w2i)
    expl_path = os.path.join(save_path, 'explanations.txt')
    expl_file = open(expl_path, 'w', encoding='utf8')

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, classifier.vocab, device=device, sort=True)
        mask = (x != 1)

        with torch.no_grad():
            clf_logits = classifier(x)  # classifier forward pass (to compute alphas)
            clf_targets = classifier.predict(clf_logits).detach()
            clf_alphas = get_alphas(classifier)
            unit_probas = (clf_alphas > 0).float()
            message = bag_of_probas(x, probas=unit_probas, vocab_size=vocab_size, mask=mask)
            message = message.detach()

            logits = layperson(message)  # forward pass
            predictions = layperson.predict(logits)
            alphas = unit_probas

        # reverse sort
        alphas = alphas[reverse_map]

        for i, ex in enumerate(mb):
            tokens = ex.tokens
            true_label = targets[i].item()
            pred_l = predictions[i].item()
            pred_c = clf_targets[i].item()

            alpha = alphas[i][:len(tokens)].tolist()

            # get words associated with nonzero alphas
            words = [tk for tk, a in zip(tokens, alpha) if a > 0]
            words = ' '.join(words)

            txt = '{}\t{}\t{}\t{}'.format(true_label, pred_l, pred_c, words)
            expl_file.write(txt + '\n')

            sent_id += 1


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    # Custom initialization
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def get_predict_args():
    parser = argparse.ArgumentParser(description='Agnews prediction')
    parser.add_argument('--ckpt', type=str, default="path_to_checkpoint",
                        required=True)
    parser.add_argument('--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args


def get_args():
    parser = argparse.ArgumentParser(description='Agnews')
    parser.add_argument('--save_path', type=str, default='agnews_results/default')
    parser.add_argument('--resume_snapshot', type=str, default='')

    parser.add_argument('--num_iterations', type=int, default=-25)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--eval_batch_size', type=int, default=25)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.)

    parser.add_argument('--model',
                        choices=["baseline", "rl", "attention",
                                 "latent"],
                        default="baseline")
    parser.add_argument('--dist', choices=["", "hardkuma"],
                        default="")

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--layer', choices=["lstm"], default="lstm")
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')

    parser.add_argument('--dependent-z', action='store_true',
                        help="make dependent decisions for z")

    # rationale settings for RL model
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--coherence', type=float, default=0.0)

    # rationale settings for HardKuma model
    parser.add_argument('--selection', type=float, default=1.,
                        help="Target text selection rate for Lagrange.")
    parser.add_argument('--lasso', type=float, default=0.0)

    # lagrange settings
    parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
    parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                        help="alpha for computing the running average")
    parser.add_argument('--lambda_init', type=float, default=1e-4,
                        help="initial value for lambda")

    # misc
    parser.add_argument('--word_vectors', type=str,
                        default='data_spec/embs/glove/glove.840B.300d.small.raw.pickle.agnews')
    args = parser.parse_args()
    return args


def get_comm_args():
    parser = argparse.ArgumentParser(description='Communication')
    parser.add_argument('--ckpt', type=str, default="path to classifier checkpoint", required=True)
    parser.add_argument('--save_explanations', type=int, default=1)

    parser.add_argument('--save_path', type=str, default='agnews_comm_results/default')
    parser.add_argument('--resume_snapshot', type=str, default='')

    parser.add_argument('--num_iterations', type=int, default=-10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.)

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--layperson', choices=["linear"], default="linear")
    parser.add_argument('--explainer', choices=["embedded", "topk"], default="embedded")

    # misc
    parser.add_argument('--word_vectors', type=str,
                        default='data_spec/embs/glove/glove.840B.300d.small.raw.pickle.agnews')
    args = parser.parse_args()
    return args
