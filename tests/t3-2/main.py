# %% IMPORTS
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.data_processing import Ngram, train_loader_MNIST, test_loader_MNIST, sequence_loader_MNIST
from src.model import Model
from config import DEVICE
from src.training import SGD, SPDG

torch.manual_seed(514579)
np.random.seed(9173537)

# %% GENERATING DATASET
n = 3
ngram = Ngram(n)
ngram[(0, 1, 2)] = 6.
ngram[(1, 2, 3)] = 4.
ngram.norm()

data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=20000)

# %% DUAL TRAINING
model = Model(ngram, output_size=4)
model.to(DEVICE)
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-4

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               data_loader, test_loader, num_epochs=50, log_every=20, test_every=1,
               eval_predictions_on_data=True)

# %% PLOTTING TEST

xs = range(len(history['predictions']))
ys = [[100.0 - preds[i, i] / preds[i].sum() * 100 for preds in history['predictions']] for i in range(4)]
mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
for i in range(4):
    plt.plot(xs, ys[i], label=str(i))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.savefig("predictions_test_error_1")

# %% PLOTTING DATA

ys = [[100.0 - preds[i, i] / preds[i].sum() * 100 for preds in history['predictions_data']] for i in range(4)]
mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
for i in range(4):
    plt.plot(xs, ys[i], label=str(i))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.savefig("predictions_data_error_1")

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
