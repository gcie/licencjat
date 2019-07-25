# IMPORTS
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from src.data_processing import Ngram, train_loader_MNIST, test_loader_MNIST, sequence_loader_MNIST
from src.model import Model
from config import DEVICE
from src.training import SGD, SPDG

torch.manual_seed(413862)
np.random.seed(8752643)

# GENERATING DATASET
n = 3
ngram = Ngram(n)  # create_ngram(np.array([[0, 1, 2]]), n)
# ngram[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)] = 1.
ngram[(0, 1, 2)] = 11.
ngram[(1, 2, 3)] = 3.
ngram[(4, 5, 6)] = 1.
ngram[(5, 6, 7)] = 1.
ngram[(6, 7, 8)] = 1.
ngram[(7, 8, 9)] = 1.
ngram[(8, 9, 0)] = 1.
ngram[(9, 0, 1)] = 1.
# ngram[(1, 3, 5)] = 0.
# ngram[(5, 7, 9)] = 0.
# ngram[(9, 5, 1)] = 0.
ngram.norm()

data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=20000)

# REGULAR TRAINING (SGD)
model = Model(ngram)
model.to(DEVICE)
model.init_weights()

optimizer = torch.optim.Adam(model.primal.parameters())
history = SGD(model, optimizer, data_loader, test_loader, num_epochs=3, log_every=50, test_every=1)

# DUAL TRAINING
model = Model(ngram, output_size=10)
model.to(DEVICE)
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-2

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=10, log_every=20, test_every=1)


# DUAL TRAINING (CONTINUATION)
history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=20, log_every=100, test_every=1, history=history)


# DUAL TRAINING (CONTINUATION VERY LONG)
history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=1000, log_every=20, test_every=1, history=history)

# PLOTTING

xs = range(len(history['loss']))
ys = signal.savgol_filter(history['loss'], 51, 7)

mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
plt.plot(xs, ys)
plt.savefig("fig_3")

# %% STATISTICS
stats = history['predictions'][-1]
print(stats)
print("\nn | acc\n--+------")
for i, x in zip(range(10), np.diag(stats) / stats.sum(axis=1) * 100.0):
    print("{} | {:>5.2f}".format(i, x))

# %% SAVE
fname = 'example'
comment = ''

np.save(fname + '_hist', history)
np.save(fname + '_model', model)
np.save(fname + '_ngram', ngram)

with open(fname + '_doc', "w+") as doc:
    doc.write("primal_lr: {}\ndual_lr: {}\nn: {}\n{}".format(primal_lr, dual_lr, ngram.n, comment))


# %% RESTORE

# TODO

# %%
