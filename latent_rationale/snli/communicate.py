import time

import torch
import torch.optim
import torch.nn as nn

import os
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchtext import data

from latent_rationale.snli.constants import UNK_TOKEN, PAD_TOKEN, INIT_TOKEN
from latent_rationale.snli.models.layperson import LinearLayperson
from latent_rationale.snli.text import SNLI
from latent_rationale.snli.models.model_helper import build_model
from latent_rationale.snli.util import print_config, print_parameters, get_device, get_data_fields, \
    load_glove_words, get_predict_args, find_ckpt_in_directory, get_comm_args, get_n_correct, save_checkpoint, \
    bag_of_probas, get_n_correct_comm, expl_dataset
from latent_rationale.common.util import make_kv_string, get_alphas
from latent_rationale.snli.util import print_examples, extract_attention
from latent_rationale.snli.evaluate import evaluate, evaluate_comm


def communicate():

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

    train_iter = data.BucketIterator(
        train_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    dev_iter = data.BucketIterator(
        dev_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    test_iter = data.BucketIterator(
        test_data, batch_size=comm_cfg.batch_size, train=False, repeat=False,
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
    layperson.to(device)

    trainable_parameters = list(filter(lambda p: p.requires_grad, layperson.parameters()))
    opt = AdamW(trainable_parameters, lr=comm_cfg.lr, weight_decay=comm_cfg.weight_decay)

    scheduler = ReduceLROnPlateau(opt, "max", patience=comm_cfg.patience,
                                  factor=comm_cfg.lr_decay, min_lr=comm_cfg.min_lr,
                                  verbose=True)

    if comm_cfg.eval_every == -1:
        comm_cfg.eval_every = int(np.ceil(len(train_data) / comm_cfg.batch_size))
        print("Eval every: %d" % comm_cfg.eval_every)

    if comm_cfg.save_every == -1:
        comm_cfg.save_every = int(np.ceil(len(train_data) / comm_cfg.batch_size))
        print("Save every: %d" % comm_cfg.save_every)

    vocab_size = len(input_field.vocab)
    iterations = 0
    start = time.time()
    best_dev_acc = -1
    best_dev_acc_true = -1
    train_iter.repeat = False
    writer = SummaryWriter(log_dir=comm_cfg.save_path)  # TensorBoard

    # test_eval = evaluate_comm(classifier, layperson, layperson.criterion, test_iter, comm_cfg)
    # import ipdb; ipdb.set_trace()

    for epoch in range(comm_cfg.epochs):
        train_iter.init_epoch()
        n_correct, n_correct_true, n_total = 0, 0, 0
        for batch_idx, batch in enumerate(train_iter):

            # switch layperson to training mode, clear gradient accumulators
            layperson.train()
            opt.zero_grad()

            iterations += 1

            # forward pass
            clf_logits = classifier(batch)  # classifier forward pass (to compute alphas)
            clf_targets = clf_logits.argmax(dim=-1).detach()

            clf_alphas = get_alphas(classifier)
            unit_probas = (clf_alphas > 0).float()

            prem_input, prem_lengths = batch.premise
            prem_mask = (prem_input != comm_cfg.pad_idx)
            message = bag_of_probas(prem_input, probas=unit_probas, vocab_size=comm_cfg.n_embed, mask=prem_mask)
            message = message.detach()
            output = layperson(message, batch)

            # calculate accuracy of predictions in the current batch
            n_correct += get_n_correct_comm(clf_targets, output)
            n_correct_true += get_n_correct_comm(batch.label, output)
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            train_acc_true = 100. * n_correct_true / n_total

            # calculate loss of the network output with respect to clf labels
            loss, optional = layperson.get_loss(output, clf_targets)

            # backpropagate and update optimizer learning rate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layperson.parameters(), comm_cfg.max_grad_norm)
            opt.step()

            # checkpoint layperson periodically
            if iterations % comm_cfg.save_every == 0:
                ckpt = {
                    "layperson": layperson.state_dict(),
                    "cfg": comm_cfg,
                    "iterations": iterations,
                    "epoch": epoch,
                    "best_dev_acc": best_dev_acc,
                    "best_dev_acc_true": best_dev_acc_true,
                    "optimizer": opt.state_dict()
                }
                save_checkpoint(ckpt, comm_cfg.save_path, iterations,
                                delete_old=True)

            # print progress message
            if iterations % comm_cfg.print_every == 0:
                writer.add_scalar('train/loss', loss.item(), iterations)
                writer.add_scalar('train/acc', train_acc, iterations)
                writer.add_scalar('train/acc_true', train_acc_true, iterations)
                for k, v in optional.items():
                    writer.add_scalar('train/' + k, v, iterations)

                opt_s = make_kv_string(optional)
                elapsed = int(time.time() - start)
                print("{:02d}:{:02d}:{:02d} epoch {:03d} "
                      "iter {:08d} loss {:.4f} {}".format(
                        elapsed // 3600, elapsed % 3600 // 60, elapsed % 60,
                        epoch, iterations, loss.item(), opt_s))

            # evaluate performance on validation set periodically
            if iterations % comm_cfg.eval_every == 0:

                # switch layperson to evaluation mode
                layperson.eval()
                dev_iter.init_epoch()
                test_iter.init_epoch()

                # calculate accuracy on validation set
                dev_eval = evaluate_comm(classifier, layperson, layperson.criterion, dev_iter, comm_cfg)
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/%s' % k, v, iterations)

                dev_eval_str = make_kv_string(dev_eval)
                print("# Evaluation dev : epoch {:2d} iter {:08d} {}".format(
                    epoch, iterations, dev_eval_str))

                # calculate accuracy on test set
                test_eval = evaluate_comm(classifier, layperson, layperson.criterion, test_iter, comm_cfg)
                for k, v in test_eval.items():
                    writer.add_scalar('test/%s' % k, v, iterations)

                test_eval_str = make_kv_string(test_eval)
                print("# Evaluation test: epoch {:2d} iter {:08d} {}".format(
                    epoch, iterations, test_eval_str))

                # update learning rate scheduler
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                else:
                    scheduler.step(dev_eval["acc"])

                # update best validation set accuracy
                if dev_eval["acc"] > best_dev_acc:

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/%s' % k, v, iterations)

                    for k, v in test_eval.items():
                        writer.add_scalar('best/test/%s' % k, v, iterations)

                    print("# New highscore {} iter {}".format(dev_eval["acc"], iterations))

                    # found a layperson with better validation set accuracy
                    best_dev_acc = dev_eval["acc"]
                    best_dev_acc_true = dev_eval["acc_true"]

                    # save layperson, delete previous 'best_*' files
                    ckpt = {
                        "layperson": layperson.state_dict(),
                        "cfg": comm_cfg,
                        "iterations": iterations,
                        "epoch": epoch,
                        "best_dev_acc": best_dev_acc,
                        "best_dev_acc_true": best_dev_acc_true,
                        "best_test_acc": test_eval["acc"],
                        "best_test_acc_true": test_eval["acc_true"],
                        "optimizer": opt.state_dict()
                    }
                    save_checkpoint(
                        ckpt, comm_cfg.save_path, iterations, prefix="best_ckpt",
                        dev_acc=dev_eval["acc"], dev_acc_true=dev_eval["acc_true"],
                        test_acc=test_eval["acc"], test_acc_true=test_eval["acc_true"],
                        delete_old=True)
                    break

                if opt.param_groups[0]["lr"] < comm_cfg.stop_lr_threshold:
                    print("Learning rate too low, stopping")
                    # writer.close()
                    # exit()
                    break

    writer.close()

    if comm_cfg.save_explanations:
        print("Saving explanations")
        expl_dataset(classifier, layperson, test_iter,
                     input_field.vocab, label_field.vocab,
                     comm_cfg.save_path, comm_cfg)


if __name__ == "__main__":
    communicate()
