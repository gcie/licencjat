# %% IMPORTS
import argparse
import os
# %% TEST
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DEVICE
from src.data_processing import (Ngram, randomized_ngram,
                                 sequence_loader_MNIST, test_loader_MNIST,
                                 train_loader_MNIST)
from src.history import History
from src.model import Model
from src.remote import mpl
from src.statistics import get_statistics
from src.training import SGD, SPDG

# torch.manual_seed()
# np.random.seed()

torch.manual_seed(40872390)
np.random.seed(89237482)

# torch.manual_seed(89127349)
# np.random.seed(479238478)



def save(history, model, ngram, optimizer_primal, optimizer_dual, primal_lr, dual_lr, comment=''):
    if not os.path.exists('data'):
        os.mkdir('data')
    fname = 'data/t3-1-4'

    np.save(fname + '_hist', history)
    np.save(fname + '_model', model)
    np.save(fname + '_ngram', ngram)
    np.save(fname + '_opt_primal', optimizer_primal)
    np.save(fname + '_opt_dual', optimizer_dual)
    with open(fname + '_doc', "w+") as doc:
        doc.write("primal_lr: {}\ndual_lr: {}\nn: {}\n{}".format(primal_lr, dual_lr, ngram.n, comment))


def load():
    fname = 'data/t3-1-4'
    hist = np.load(fname + '_hist.npy').item()
    model = np.load(fname + '_model.npy').item()
    ngram = np.load(fname + '_ngram.npy').item()
    opt_primal = np.load(fname + '_opt_primal.npy').item()
    opt_dual = np.load(fname + '_opt_dual.npy').item()
    return hist, model, ngram, opt_primal, opt_dual


# if __name__ == "__main__" and False:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cont", action="store_true")
#     parser.add_argument("--epochs", default=1000, type=int)
#     parser.add_argument("--save-every", default=100, type=int)
#     parser.add_argument("--log-every", default=100, type=int)
#     parser.add_argument("--test-every", default=5, type=int)
#     parser.add_argument("--primal-lr", default=1e-6, type=float)
#     parser.add_argument("--dual-lr", default=1e-4, type=float)
#     args = parser.parse_args()
#     continuation = args.cont
#     num_epochs = args.epochs
#     save_every = args.save_every
#     log_every = args.log_every
#     test_every = args.test_every
#     primal_lr = args.primal_lr
#     dual_lr = args.dual_lr
# else:

continuation = False
num_epochs = 1000
save_every = 200
log_every = 100
test_every = 5
primal_lr = 1e-6
dual_lr = 1e-4

# %% GENERATING DATASET
# ngram = randomized_ngram(3, 2, out_dim=4, min_var=5e-2)
ngram = Ngram(3)
ngram[(0, 1, 2)] = 9.
ngram[(1, 2, 3)] = 1.
ngram.norm()
ngram.show()

# %% CREATING MODEL
data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=50000)
sequence_test_loader = sequence_loader_MNIST(ngram, num_samples=10000)


# %% REGULAR TRAINING (SGD)
# model = Model(ngram)
# model.to(DEVICE)
# model.init_weights()

# optimizer = torch.optim.Adam(model.primal.parameters())
# history = SGD(model, optimizer, data_loader, test_loader, num_epochs=1, log_every=50, test_every=1)

# %% DUAL TRAINING
if continuation:
    history, model, ngram, optimizer_primal, optimizer_dual = load()
    print(model.ngram.n)
else:
    model = Model(ngram, output_size=4)
    model.to(DEVICE)
    model.init_weights()

    optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
    optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

    history = History()
    for idx in model.dual:
        history['dual ' + str(idx)] = []


epochs_done = 0
while epochs_done < num_epochs:
    history = SPDG(model, optimizer_primal, optimizer_dual, 
                    sequence_loader,
                   data_loader, test_loader, save_every, log_every,
                   test_every,
                   sequence_test_loader=sequence_test_loader,
                    show_dual=False, eval_predictions_on_sequences=True,
                   history=history)
    save(history, model, ngram, optimizer_primal, optimizer_dual, primal_lr, dual_lr)
    epochs_done += save_every


# %% PLOTTING TEST

xs = np.arange(len(history['predictions'])) * test_every
ys = [[100.0 - preds[i, i] / preds[i].sum() * 100 for preds in history['predictions']] for i in range(model.output_size)]
mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
for i in range(model.output_size):
    plt.plot(xs, ys[i], label=str(i))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.savefig("predictions_test_error")
plt.close()

# %% PLOTTING DATA

ys = [[100.0 - preds[i, i] / preds[i].sum() * 100 for preds in history['predictions_data']] for i in range(model.output_size)]
mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
for i in range(model.output_size):
    plt.plot(xs, ys[i], label=str(i))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.savefig("predictions_data_error")
plt.close()

# %% STATISTICS
stats = history['predictions'][-1]
print(stats)
print("\nn | acc\n--+------")
for i, x in zip(range(10), np.pad(np.diag(stats), (0, 10 - model.output_size), 'constant') / stats.sum(axis=1) * 100.0):
    print("{} | {:>5.2f}".format(i, x))

# %% RESTORE

fname = 't3-1-4'
history = np.load(fname + '_hist.npy').item()
