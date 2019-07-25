# %% IMPORTS
import numpy as np
import torch

from config import DEVICE
from src.data_processing import (Ngram, sequence_loader_MNIST,
                                 test_loader_MNIST, train_loader_MNIST)
from src.model import Model
from src.training import SPDG

torch.manual_seed(413862)
np.random.seed(8752643)

# GENERATING DATASET
n = 1
ngram = Ngram(n)
ngram[(0)] = 9.
ngram[(1)] = 1.
ngram.norm()

data_loader = train_loader_MNIST()
test_loader = test_loader_MNIST()
sequence_loader = sequence_loader_MNIST(ngram, num_samples=20000)

# DUAL TRAINING
model = Model(ngram, output_size=2)
model.to(DEVICE)
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-2

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=10, log_every=20, test_every=1)


# %% SAVE
fname = 't1-1-1'
comment = ''

np.save(fname + '_hist', history)
np.save(fname + '_model', model)
np.save(fname + '_ngram', ngram)

with open(fname + '_doc', "w+") as doc:
    doc.write("primal_lr: {}\ndual_lr: {}\nn: {}\n{}".format(primal_lr, dual_lr, ngram.n, comment))



# %%
