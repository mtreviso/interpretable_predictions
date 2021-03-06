import torch
from collections import defaultdict
from itertools import count
import numpy as np

from latent_rationale.sst.util import get_minibatch, prepare_minibatch, bag_of_probas
from latent_rationale.common.util import get_z_stats, get_alphas


def get_histogram_counts(z=None, mask=None, mb=None):
    counts = np.zeros(5).astype(np.int64)

    for i, ex in enumerate(mb):

        tokens = ex.tokens
        token_labels = ex.token_labels

        if z is not None:
            ex_z = z[i][:len(tokens)]

        if mask is not None:
            assert mask[i].sum() == len(tokens), "mismatch mask/tokens"

        for j, tok, lab in zip(count(), tokens, token_labels):
            if z is not None:
                if ex_z[j] > 0:
                    counts[lab] += 1
            else:
                counts[lab] += 1

    return counts


def evaluate(model, data, batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)
    histogram_totals = np.zeros(5).astype(np.int64)
    z_histogram_totals = np.zeros(5).astype(np.int64)

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(mb, model.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            logits = model(x)
            predictions = model.predict(logits)

            loss, loss_optional = model.get_loss(logits, targets, mask=mask)

            if isinstance(loss, dict):
                loss = loss["main"]

            totals['loss'] += loss.item() * batch_size

            for k, v in loss_optional.items():
                if not isinstance(v, float):
                    v = v.item()

                totals[k] += v * batch_size

            if hasattr(model, "z"):
                n0, nc, n1, nt = get_z_stats(model.z, mask)
                z_totals['p0'] += n0
                z_totals['pc'] += nc
                z_totals['p1'] += n1
                z_totals['total'] += nt

                # histogram counts
                # for this need to sort z in original order
                z = model.z.squeeze(1).squeeze(-1)[reverse_map]
                mask = mask[reverse_map]
                z_histogram = get_histogram_counts(z=z, mask=mask, mb=mb)
                z_histogram_totals += z_histogram
                histogram = get_histogram_counts(mb=mb)
                histogram_totals += histogram

        # add the number of correct predictions to the total correct
        totals['acc'] += (predictions == targets.view(-1)).sum().item()
        totals['total'] += batch_size

    result = {}

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    # z scores
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            result[k] = v / z_totals["total"]

    if "p0" in result:
        result["selected"] = 1 - result["p0"]

    return result


def evaluate_comm(classifier, layperson, data, batch_size=25, device=None):
    """Accuracy of a layperson on given data set (using minibatches)"""

    classifier.eval()  # disable dropout
    layperson.eval()  # disable dropout

    totals = defaultdict(float)
    vocab_size = len(layperson.vocab.w2i)

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(mb, layperson.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            clf_logits = classifier(x)  # classifier forward pass (to compute alphas)
            clf_targets = classifier.predict(clf_logits).detach()
            clf_alphas = get_alphas(classifier)
            unit_probas = (clf_alphas > 0).float()
            message = bag_of_probas(x, probas=unit_probas, vocab_size=vocab_size, mask=mask)
            message = message.detach()

            logits = layperson(message)  # forward pass
            predictions = layperson.predict(logits)
            loss, loss_optional = layperson.get_loss(logits, clf_targets, mask=mask)

            if isinstance(loss, dict):
                loss = loss["main"]

            totals['loss'] += loss.item() * batch_size

            for k, v in loss_optional.items():
                if not isinstance(v, float):
                    v = v.item()
                totals[k] += v * batch_size

        # add the number of correct predictions to the total correct
        totals['acc'] += (predictions == clf_targets.view(-1)).sum().item()
        totals['true_acc'] += (predictions == targets.view(-1)).sum().item()
        totals['total'] += batch_size

    result = {}

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    return result
