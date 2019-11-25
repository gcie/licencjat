# %% IMPORTS
import numpy as np
import torch

from config import DEVICE
from src.data_processing import (Ngram, randomized_ngram,
                                 sequence_loader_MNIST, test_loader_MNIST,
                                 train_loader_MNIST)
from src.model import Model
from src.remote import mpl
from src.training import SGD, SPDG
from src.statistics import get_statistics
import matplotlib.pyplot as plt

torch.manual_seed(50314896)
np.random.seed(301623220)

# %% GENERATING DATASET
ngram = Ngram(1)
ngram[(0)] = 5.
ngram[(1)] = 95.
ngram.norm()

data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=40000)

# %% REGULAR TRAINING (SGD)
# model = Model(ngram)
# model.to(DEVICE)
# model.init_weights()

# optimizer = torch.optim.Adam(model.primal.parameters())
# history = SGD(model, optimizer, data_loader, test_loader, num_epochs=1, log_every=50, test_every=1)

# %% DUAL TRAINING
model = Model(ngram, output_size=2)
model.to(DEVICE)
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-4

num_epochs = 1000
log_every = 100     # batches
test_every = 5      # epochs

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               data_loader, test_loader, num_epochs, log_every,
               test_every, eval_predictions_on_data=True, show_dual=True)


# %% DUAL TRAINING (CONTINUATION)
# history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
#                data_loader, test_loader, num_epochs=10, log_every=100,
#                test_every=1, eval_predictions_on_data=True, show_dual=True, history=history)


# # %% DUAL TRAINING (CONTINUATION VERY LONG)
# history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
#                test_loader, num_epochs=1000, log_every=50, test_every=1, history=history)


# %% SAVE
fname = 't1-3-1'
comment = ''

np.save(fname + '_hist', history)
np.save(fname + '_model', model)
np.save(fname + '_ngram', ngram)

with open(fname + '_doc', "w+") as doc:
    doc.write("primal_lr: {}\ndual_lr: {}\nn: {}\n{}".format(primal_lr, dual_lr, ngram.n, comment))


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

# %% STATISTICS
stats = history['predictions'][-1]
print(stats)
print("\nn | acc\n--+------")
for i, x in zip(range(10), np.pad(np.diag(stats), (0, 10 - model.output_size), 'constant') / stats.sum(axis=1) * 100.0):
    print("{} | {:>5.2f}".format(i, x))

# %% RESTORE

fname = 't1-3-1'
history = np.load(fname + '_hist.npy').item()
