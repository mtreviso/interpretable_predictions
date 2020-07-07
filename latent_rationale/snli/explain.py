import os

import torch
import torch.optim
import torchtext
from torchtext.data import Dataset

from latent_rationale.snli.constants import UNK_TOKEN, PAD_TOKEN, INIT_TOKEN
from latent_rationale.snli.models.layperson import LinearLayperson
from latent_rationale.snli.models.model_helper import build_model
from latent_rationale.snli.text import SNLI
from latent_rationale.snli.util import print_config, print_parameters, get_device, get_data_fields, \
    load_glove_words, find_ckpt_in_directory, get_comm_args, expl_dataset


def read_human_corpus(path, fields_dict, lower=False):
    exs = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.lower() if lower else line
            label, words = line.strip().split('\t')
            prem, hypo = words.split('|||')
            prem = prem.strip()
            hypo = hypo.strip()
            label = label.strip().lower()
            ex = {'premise': prem, 'hypothesis': hypo, 'label': label}
            exs.append(torchtext.data.Example.fromdict(ex, fields_dict))
    return exs


def explain():

    comm_cfg = get_comm_args()
    device = get_device()
    print(device)

    # load checkpoint
    ckpt_path = find_ckpt_in_directory(comm_cfg.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # to know which words to UNK we need to know the Glove vocabulary
    glove_words = load_glove_words(cfg.word_vectors)

    # load data sets
    print("Loading data... ", end="")
    input_field, label_field, not_in_glove = get_data_fields(glove_words)
    train_data, dev_data, test_data = SNLI.splits(input_field, label_field)
    print("Done")
    print("Words not in glove:", len(not_in_glove))

    fields_tuples = [
        ('premise', input_field),
        ('hypothesis', input_field),
        ('label', label_field)
    ]
    fields_dict = dict(fields_tuples)
    fields_dict = dict(zip(fields_dict.keys(), fields_dict.items()))
    # human_examples = read_human_corpus('data_spec/human-corpus/snli.txt', fields_dict)
    human_examples = read_human_corpus('data_spec/human-corpus-dev/snli.txt', fields_dict)
    human_data = torchtext.data.Dataset(human_examples, fields_tuples)

    # import ipdb; ipdb.set_trace()

    # build vocabulary (deterministic so no need to load it)
    input_field.build_vocab(train_data, dev_data, test_data,
                            # vectors=None, vectors_cache=None)
                            unk_init=lambda x: x.normal_(mean=0, std=1.0),
                            vectors=comm_cfg.word_vectors, vectors_cache=None)
    label_field.build_vocab(train_data)

    # construct model
    model = build_model(cfg, input_field.vocab)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["model"])
    print("Done")

    train_iter = torchtext.data.BucketIterator(
        train_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    dev_iter = torchtext.data.BucketIterator(
        dev_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    test_iter = torchtext.data.BucketIterator(
        human_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
        shuffle=False, sort=False, sort_within_batch=False,
        device=device if torch.cuda.is_available() else -1)

    print_config(comm_cfg)

    print("Embedding variance:", torch.var(model.embed.weight).item())
    model.to(device)

    print_parameters(model)
    print(model)

    # from now on I will call the original model as "classifier"
    classifier = model
    input_field.vocab.vectors = model.embed.weight.clone()

    comm_cfg.n_embed = len(input_field.vocab)
    comm_cfg.output_size = len(label_field.vocab)
    comm_cfg.n_cells = cfg.n_layers
    comm_cfg.pad_idx = input_field.vocab.stoi[PAD_TOKEN]
    comm_cfg.unk_idx = input_field.vocab.stoi[UNK_TOKEN]
    comm_cfg.init_idx = input_field.vocab.stoi[INIT_TOKEN]

    # normalize word embeddings (each word embedding has L2 norm of 1.)
    if comm_cfg.normalize_embeddings:
        with torch.no_grad():
            input_field.vocab.vectors /= input_field.vocab.vectors.norm(2, dim=-1, keepdim=True)

    # zero out padding
    with torch.no_grad():
        input_field.vocab.vectors[cfg.pad_idx].zero_()

    # communication
    for k, v in vars(comm_cfg).items():
        print("{:20} : {:10}".format(k, v))

    # switch model to evaluation mode
    classifier.eval()
    train_iter.init_epoch()
    dev_iter.init_epoch()
    test_iter.init_epoch()

    # define layperson
    layperson = LinearLayperson(comm_cfg, input_field.vocab)
    if comm_cfg.word_vectors:
        with torch.no_grad():
            layperson.embed.weight.data.copy_(input_field.vocab.vectors)
            # layperson.embed.weight.data.copy_(classifier.embed.weight.data)

    # load layperson first
    # load checkpoint
    ckpt_comm_path = find_ckpt_in_directory(comm_cfg.ckpt_comm)
    ckpt_comm = torch.load(ckpt_comm_path, map_location=device)
    # cfg_comm = ckpt_comm["cfg"]
    layperson.load_state_dict(ckpt_comm["layperson"])
    layperson = layperson.to(device)
    print(layperson)
    print_parameters(layperson)

    # load layperson
    # test_eval = evaluate_comm(classifier, layperson, layperson.criterion, test_iter, comm_cfg)
    # print(test_eval)

    print("Saving explanations")
    if not os.path.exists(comm_cfg.save_path):
        os.mkdir(comm_cfg.save_path)
    expl_dataset(classifier, layperson, test_iter,
                 input_field.vocab, label_field.vocab,
                 comm_cfg.save_path, comm_cfg)


if __name__ == "__main__":
    explain()
