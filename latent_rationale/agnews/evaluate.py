import torch
from collections import defaultdict

from latent_rationale.agnews.util import get_minibatch, prepare_minibatch, bag_of_probas
from latent_rationale.common.util import get_z_stats, get_alphas


def evaluate(model, data, batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)

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
