# %% IMPORTS
import torch
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from src.data_processing import Ngram, train_loader_MNIST, test_loader_MNIST, sequence_loader_MNIST, randomized_ngram
from src.model import Model
from config import DEVICE
from src.training import SGD, SPDG

torch.manual_seed(403249064)
np.random.seed(235084161)

# %% GENERATING DATASET
n = 1
ngram = Ngram(n)
ngram[(0)] = 1.
ngram[(1)] = 9.
ngram.norm()

data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=20000)

# %% DUAL TRAINING
model = Model(ngram, output_size=2)
model.to(DEVICE)
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-4

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader, sequence_loader,
               test_loader, num_epochs=1000, log_every=100, test_every=5,
               eval_predictions_on_data=True, show_dual=True)


# %% SAVE
fname = 't1-1-1'
comment = ''

np.save(fname + '_hist', history)
np.save(fname + '_model', model)
np.save(fname + '_ngram', ngram)

with open(fname + '_doc', "w+") as doc:
    doc.write("primal_lr: {}\ndual_lr: {}\nn: {}\n{}".format(primal_lr, dual_lr, ngram.n, comment))

# %% PLOTTING TEST

xs = np.arange(len(history['predictions'])) * 5
ys0 = [100.0 - preds[0, 0] / preds[0].sum() * 100 for preds in history['predictions']]
ys1 = [100.0 - preds[1, 1] / preds[1].sum() * 100 for preds in history['predictions']]

mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
plt.plot(xs, ys0, label='0')
plt.plot(xs, ys1, label='1')
plt.legend()
plt.savefig("predictions_test_error")
plt.close()
# %% PLOTTING DATA

xs = np.arange(len(history['predictions_data'])) * 5
ys0 = [100.0 - preds[0, 0] / preds[0].sum() * 100 for preds in history['predictions_data']]
ys1 = [100.0 - preds[1, 1] / preds[1].sum() * 100 for preds in history['predictions_data']]

mpl.rc('savefig', format='svg')
mpl.rc('lines', linewidth=0.5)
mpl.style.use('seaborn')
plt.plot(xs, ys0, label='0')
plt.plot(xs, ys1, label='1')
plt.legend()
plt.savefig("predictions_data_error")
plt.close()
# %% STATISTICS
# stats = history['predictions'][-1]
# print(stats)
# print("\nn | acc\n--+------")
# for i, x in zip(range(10), np.diag(stats) / stats.sum(axis=1) * 100.0):
#     print("{} | {:>5.2f}".format(i, x))


# %% RESTORE

fname = 't1-1-1'
history = np.load(fname + '_hist.npy').item()
