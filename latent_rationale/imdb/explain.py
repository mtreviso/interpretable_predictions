import os
from collections import OrderedDict

import torch
import torch.optim

from latent_rationale.imdb.models.layperson import LinearLayperson
from latent_rationale.imdb.models.model_helpers import build_model
from latent_rationale.imdb.util import imdb_reader, \
    load_glove, print_parameters, get_device, find_ckpt_in_directory, \
    get_comm_args, initialize_model_, expl_dataset, find_ckpt_in_directory_comm
from latent_rationale.imdb.vocabulary import Vocabulary


def explain():
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
    train_data = list(imdb_reader("data_spec/corpus/imdb/train/data.txt"))
    dev_data = list(imdb_reader("data_spec/corpus/imdb/dev/data.txt"))
    # test_data = list(imdb_reader("data_spec/human-corpus/imdb.txt"))
    test_data = list(imdb_reader("data_spec/human-corpus-dev/imdb.txt"))

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
    i2t = ["negative", "positive"]
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

    # load layperson first
    # load checkpoint
    ckpt_comm_path = find_ckpt_in_directory_comm(comm_cfg["ckpt_comm"])
    ckpt_comm = torch.load(ckpt_comm_path, map_location=device)
    # cfg_comm = ckpt_comm["cfg"]
    layperson.load_state_dict(ckpt_comm["state_dict"])
    layperson = layperson.to(device)
    print(layperson)
    print_parameters(layperson)

    # test_eval = evaluate_comm(
    #     classifier, layperson, test_data, batch_size=eval_batch_size,
    #     device=device)
    # print('Test:', make_kv_string(test_eval))

    print("Saving explanations")
    if not os.path.exists(comm_cfg["save_path"]):
        os.mkdir(comm_cfg["save_path"])
    expl_dataset(classifier, layperson, test_data, batch_size=batch_size,
                 device=device, save_path=comm_cfg["save_path"])


if __name__ == "__main__":
    explain()
