import os
from collections import OrderedDict
import time
import json

import torch
import torch.optim
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from latent_rationale.common.util import make_kv_string, get_alphas
from latent_rationale.yelp.models.layperson import LinearLayperson
from latent_rationale.yelp.vocabulary import Vocabulary
from latent_rationale.yelp.models.model_helpers import build_model
from latent_rationale.yelp.util import get_predict_args, yelp_reader, \
    load_glove, print_parameters, get_device, find_ckpt_in_directory, \
    get_comm_args, initialize_model_, prepare_minibatch, get_minibatch, bag_of_probas, expl_dataset
from latent_rationale.yelp.evaluate import evaluate, evaluate_comm


def communicate():
    comm_cfg = get_comm_args()
    comm_cfg = vars(comm_cfg)
    device = get_device()
    print(device)

    # load checkpoint
    ckpt_path = find_ckpt_in_directory(comm_cfg["ckpt"])
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    batch_size = cfg.get("eval_batch_size", 25)

    # Let's load the data into memory.
    train_data = list(yelp_reader("data_spec/corpus/yelp/review_train.json"))
    dev_data = list(yelp_reader("data_spec/corpus/yelp/review_dev.json"))
    test_data = list(yelp_reader("data_spec/corpus/yelp/review_test.json"))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    example = dev_data[0]
    print("First train example:", example)
    print("First train example tokens:", example.tokens)
    print("First train example label:", example.label)

    vocab = Vocabulary()
    vectors = load_glove(cfg["word_vectors"], vocab)  # this populates vocab

    # Map the sentiment labels 0-4 to a more readable form (and the opposite)
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

    # Build model
    model = build_model(cfg["model"], vocab, t2i, cfg)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print("Done")

    # print model
    print(model)
    print_parameters(model)

    # from now on I will call the original model as "classifier"
    classifier = model
    classifier.eval()

    # communication
    for k, v in comm_cfg.items():
        print("{:20} : {:10}".format(k, v))

    num_iterations = comm_cfg["num_iterations"]
    print_every = comm_cfg["print_every"]
    eval_every = comm_cfg["eval_every"]
    batch_size = comm_cfg["batch_size"]
    eval_batch_size = comm_cfg.get("eval_batch_size", batch_size)
    iters_per_epoch = len(train_data) // comm_cfg["batch_size"]

    if comm_cfg["eval_every"] == -1:
        eval_every = iters_per_epoch
        print("Set eval_every to {}".format(iters_per_epoch))

    if comm_cfg["num_iterations"] < 0:
        num_iterations = iters_per_epoch * -1 * comm_cfg["num_iterations"]
        print("Set num_iterations to {}".format(num_iterations))

    # Build model
    vocab_size = len(vocab.w2i)
    output_size = len(t2i)
    layperson = LinearLayperson(vocab, vocab_size, output_size)
    initialize_model_(layperson)

    # linear layperson doesnt have an embedding layer for text classification
    # with torch.no_grad():
    #     layperson.embed.weight.data.copy_(torch.from_numpy(vectors))
    #     if comm_cfg["fix_emb"]:
    #         print("fixed word embeddings")
    #         layperson.embed.weight.requires_grad = False
    #     layperson.embed.weight[1] = 0.  # padding zero

    optimizer = AdamW(layperson.parameters(), lr=comm_cfg["lr"],
                     weight_decay=comm_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=comm_cfg["lr_decay"], patience=comm_cfg["patience"],
        verbose=True, cooldown=comm_cfg["cooldown"], threshold=comm_cfg["threshold"],
        min_lr=comm_cfg["min_lr"])

    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 0
    best_iter = 0
    writer = SummaryWriter(log_dir=comm_cfg["save_path"])  # TensorBoard

    layperson = layperson.to(device)
    print(layperson)
    print_parameters(layperson)

    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            layperson.train()
            x, targets, _ = prepare_minibatch(batch, layperson.vocab, device=device)
            mask = (x != 1)

            clf_logits = classifier(x)  # classifier forward pass (to compute alphas)
            clf_targets = classifier.predict(clf_logits).detach()
            clf_alphas = get_alphas(classifier)
            unit_probas = (clf_alphas > 0).float()
            # if comm_cfg["explainer"] == "topk":
            #     unit_probas = zero_out_non_topk_probas(unit_probas, k=comm_cfg["topk"])
            message = bag_of_probas(x, probas=unit_probas, vocab_size=vocab_size, mask=mask)
            message = message.detach()

            logits = layperson(message)  # forward pass

            loss, loss_optional = layperson.get_loss(logits, clf_targets, mask=mask)
            layperson.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(layperson.parameters(),
                                           max_norm=comm_cfg["max_grad_norm"])
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:

                train_loss = train_loss / print_every
                writer.add_scalar('train/loss', train_loss, iter_i)
                for k, v in loss_optional.items():
                    writer.add_scalar('train/'+k, v, iter_i)

                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print("Epoch %r Iter %r time=%dm loss=%.4f %s" %
                      (epoch, iter_i, min_elapsed, train_loss, print_str))
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate_comm(classifier, layperson, dev_data,
                                         batch_size=eval_batch_size, device=device)
                accuracies.append(dev_eval["acc"])
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/'+k, v, iter_i)

                print("# epoch %r iter %r: dev %s" % (
                    epoch, iter_i, make_kv_string(dev_eval)))

                # save best layperson parameters
                compare_score = dev_eval["acc"]
                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score > (best_eval * (1-comm_cfg["threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print("***highscore*** %.4f" % compare_score)
                    best_eval = compare_score
                    best_iter = iter_i

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/' + k, v, iter_i)

                    if not os.path.exists(comm_cfg["save_path"]):
                        os.makedirs(comm_cfg["save_path"])

                    ckpt = {
                        "state_dict": layperson.state_dict(),
                        "cfg": comm_cfg,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    path = os.path.join(comm_cfg["save_path"], "layperson.pt")
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("# Done training")

                # evaluate on test with best layperson
                print("# Loading best layperson")
                path = os.path.join(comm_cfg["save_path"], "layperson.pt")
                if os.path.exists(path):
                    ckpt = torch.load(path)
                    layperson.load_state_dict(ckpt["state_dict"])
                else:
                    print("No layperson found.")

                print("# Evaluating")
                dev_eval = evaluate_comm(
                    classifier, layperson, dev_data, batch_size=eval_batch_size,
                    device=device)
                test_eval = evaluate_comm(
                    classifier, layperson, test_data, batch_size=eval_batch_size,
                    device=device)

                print("best layperson iter {:d}: "
                      "dev {} test {}".format(
                        best_iter,
                        make_kv_string(dev_eval),
                        make_kv_string(test_eval)))

                # save result
                result_path = os.path.join(comm_cfg["save_path"], "results.json")

                comm_cfg["best_iter"] = best_iter

                for k, v in dev_eval.items():
                    comm_cfg["dev_" + k] = v
                    writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    comm_cfg["test_" + k] = v
                    writer.add_scalar('best/test/' + k, v, iter_i)

                writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(comm_cfg, f)

                if comm_cfg["save_explanations"]:
                    print("Saving explanations")
                    expl_dataset(classifier, layperson, test_data, batch_size=batch_size,
                                 device=device, save_path=comm_cfg["save_path"])

                return losses, accuracies


if __name__ == "__main__":
    communicate()
