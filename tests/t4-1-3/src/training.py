import time

import torch

from config import DEVICE
from src.history import History
from src.statistics import get_ngram_stats, get_statistics


def SGD(model, optimizer, data_loader, test_loader, num_epochs=5, log_every=1, test_every=1, c=10,
        history={'err_rate': [], 'loss': [], 'test_err_rate': []}):
    model.train()
    iter_ = 0
    epoch_ = 0
    try:
        while epoch_ < num_epochs:
            if epoch_ % test_every == 0:
                print("Minibatch |  loss  | err rate | steps/s |")
            epoch_ += 1
            stime = time.time()
            siter = iter_
            for x, y in data_loader:
                iter_ += 1
                optimizer.zero_grad()
                x = x.to(DEVICE).view(x.size(0), -1)
                y = y.to(DEVICE)
                out = model.forward(x)
                loss = model.loss(out, y)
                loss.backward()
                optimizer.step()

                _, predictions = out.max(dim=1)
                a = predictions != y.view(-1)
                err_rate = 100.0 * a.sum().item() / out.size(0)
                history['err_rate'].append(err_rate)
                history['loss'].append(loss)

                if iter_ % log_every == 0:
                    num_iter = iter_ - siter
                    print("{:>8}  | {:>6.2f} | {:>7.2f}% | {:>7.2f} |".format(iter_, loss, err_rate, num_iter / (time.time() - stime)))
                    siter = iter_
                    stime = time.time()
            if epoch_ % test_every == 0:
                test_err_rate = model.compute_error_rate(test_loader)
                history['test_err_rate'].append(test_err_rate)
                msg = "Epoch {:>10} | Test error rate: {:.2f}".format(epoch_, test_err_rate)
                print('{0}\n{1}\n{0}'.format('----------------------------------------+', msg))
    except KeyboardInterrupt:
        pass
    return history


def SPDG(model, optimizer_primal, optimizer_dual, sequence_loader, data_loader, test_loader, num_epochs=5, log_every=1, test_every=1,
         predictions_on_data=False, predictions_on_sequences=False, show_dual=False, history=None, sequence_test_loader=None,
         ngram_data_stats=False, ngram_test_stats=False, loss_on_test=False):
    if history is None:
        history = History()
        for idx in model.dual:
            history['dual ' + str(idx)] = []
    model.train()
    iter_ = 0
    epoch_ = 0
    try:
        while epoch_ < num_epochs:
            if epoch_ % test_every == 0:
                msg = "Minibatch |   p-loss   |    loss    | err rate | steps/s |"
                if show_dual:
                    for i in model.dual:
                        msg += " {:>7d} |".format(i)
                print(msg)
            epoch_ += 1
            stime = time.time()
            siter = iter_
            for x, y in sequence_loader:
                iter_ += 1
                x = x.to(DEVICE).view(-1, model.n, 28*28).float()
                y = y.to(DEVICE)
                out = model.forward_sequences(x)
                ploss = model.loss_primal(out)
                dloss = model.loss_dual(out)
                ploss_ = ploss.item()
                loss_ = -dloss.item()

                optimizer_primal.zero_grad()
                ploss.backward(retain_graph=True)
                optimizer_primal.step()

                optimizer_dual.zero_grad()
                dloss.backward()
                optimizer_dual.step()

                with torch.no_grad():
                    _, predictions = out.max(dim=2)
                    a = predictions.view(-1) != y.view(-1)
                    err_rate = 100.0 * a.sum().item() / (out.size(1) * out.size(0))
                    history.err_rate.append(err_rate)
                    history.primal_loss.append(ploss_)
                    history.loss.append(loss_)
                    for idx in model.dual:
                        history['dual ' + str(idx)].append(model.dual[idx].item())

                    if iter_ % log_every == 0:
                        num_iter = iter_ - siter
                        msg = "{:>8}  | {:>10.2f} | {:>10.2f} | {:>7.2f}% | {:>7.2f} |".format(iter_, ploss_, loss_, err_rate,
                                                                                               num_iter / (time.time() - stime))
                        if show_dual:
                            for idx in model.dual:
                                msg += " {:>7.2f} |".format(model.dual[idx].item())
                        print(msg)
                        siter = iter_
                        stime = time.time()
            if epoch_ % test_every == 0:
                epmsg = "Epoch {:>3} | Test errors for: ".format(epoch_ + history.epochs_done)
                history.predictions_test.append(get_statistics(model, data_loader=test_loader))
                for i in range(model.output_size):
                    accuracy = 100.0 - 100.0 * history.predictions_test[-1][i, i] / history.predictions_test[-1][i].sum()
                    epmsg += " {}: {:.2f}, ".format(i, accuracy)
                epmsg = epmsg[:-2]
                if predictions_on_data:
                    epmsg += "\n          | Data errors for: ".format(epoch_)
                    history.predictions_data.append(get_statistics(model, data_loader=data_loader))
                    for i in range(model.output_size):
                        accuracy = 100.0 - 100.0 * history.predictions_data[-1][i, i] / history.predictions_data[-1][i].sum()
                        epmsg += " {}: {:.2f}, ".format(i, accuracy)
                    epmsg = epmsg[:-2]
                if predictions_on_sequences:
                    epmsg += "\n          | Seqs errors for: ".format(epoch_)
                    history.predictions_sequences.append(get_statistics(model, data_loader=sequence_loader, sequences=True))
                    for i in range(model.output_size):
                        accuracy = 100.0 - 100.0 * history.predictions_sequences[-1][i, i] / history.predictions_sequences[-1][i].sum()
                        epmsg += " {}: {:.2f}, ".format(i, accuracy)
                    epmsg = epmsg[:-2]
                if ngram_data_stats:
                    history.ngram_data_stats.append(get_ngram_stats(model, sequence_loader))
                if ngram_test_stats:
                    history.ngram_test_stats.append(get_ngram_stats(model, sequence_test_loader))
                if loss_on_test:
                    for x, y in sequence_test_loader:
                        with torch.no_grad():
                            x = x.to(DEVICE).view(-1, model.n, 28*28).float()
                            out = model.forward_sequences(x)
                            history.test_primal_loss.append(model.loss_primal(out).item())
                            history.test_loss.append(-model.loss_dual(out).item())
                print('{0}+\n{1}\n{0}+'.format('-' * (len(msg) - 1), epmsg))
    except KeyboardInterrupt:
        pass
    history.epochs_done += num_epochs
    return history
