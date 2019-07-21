# %% [markdown]
# ## Sformułowanie problemu
#
# Rozważamy problem uczenia klasyfikatora danych sekwencyjnych, który ciągowi
# $ (x_1,\dots x_{T_0}) $ przyporządkowuje ciąg $(y_1,\dots y_{T_0})$, ale bez dostępu
# do podpisanego zbioru treningowego $\mathcal{D}_{XY} = \lbrace(x_1^n, \dots, x_{T_n}^n),
# (y_1^n, \dots,y_{T_n}^n) \ : \ n=1,\dots, M \rbrace $ (tutaj $T_n$ oznacza długość $n$-tego ciągu), a jedynie do:
# * zbioru niepodpisanych danych: $\mathcal{D}_X = \lbrace(x_1^n, \dots, x_{T_n}^n)\ : \ n=1,\dots, M \rbrace$,
# * modelu n-gram: $p_{LM}(i_1,\dots ,i_N) = p_{LM}(y_{t-N+1}^n=i_1, \dots, y_t^n=i^N)$,
#
# gdzie $i_1, \dots, i_N$ są elementami sekwencji (np. słowami/literami), a subskrypt $LM$ oznacza model językowy (*Language Model*).

# %% IMPORTS, MODELS AND UTILITY FUNCTIONS
import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from modules.data_processing import Ngram, train_loader_MNIST, test_loader_MNIST, sequence_loader_MNIST

device = 'cuda'


def get_statistics(model, output_size=10):
    model.eval()
    num_errs = 0.0
    num_examples = 0
    results = np.zeros((10, output_size), dtype='int32')
    with torch.no_grad():
        for x, y in test_loader:
            # x = x.to(device).view(-1, 1, 28, 28).float()
            x = x.to(device).view(-1, 28*28).float()
            y = y.to(device)
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            for i in range(10):
                x_ = predictions[y.data == i].cpu().numpy()
                x_unique, x_unique_count = np.unique(x_, return_counts=True)
                # x_unique_count = torch.stack([(x_ == x_u).sum() for x_u in x_unique])
                for idx, occ in zip(x_unique, x_unique_count):
                    results[i, idx] += occ
            num_errs += (predictions != y.data).sum().item()
            num_examples += x.size(0)
    model.train()
    return results


class Model(nn.Module):
    """Neural net with dual parameters"""
    def __init__(self, ngram, architecture=None, output_size=10):
        super(Model, self).__init__()
        self.ngram = ngram
        self.cnt = 1
        for idx in ngram:
            ngram[idx]
        if architecture is None:
            self.primal = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, output_size)
            ).to(device)
        else:
            self.primal = architecture.to(device)
        self.dual = Ngram(self.ngram.n)
        for idx in self.ngram:
            self.dual[idx] = torch.tensor(0.).uniform_(-1, 0).to(device).requires_grad_()
        self.cuda()
        self.init_weights()

    def forward_sequences(self, x):
        return torch.nn.functional.softmax(self.primal.forward(x), dim=2)

    def forward(self, x):
        return torch.nn.functional.softmax(self.primal.forward(x), dim=0)

    def loss(self, output, target):
        return torch.mean(-torch.log(torch.gather(output, 1, target.unsqueeze(1))))

    def loss_primal(self, output, target):
        loss = torch.tensor(0, dtype=torch.float32).to(device)
        for i in self.ngram:
            loss += torch.sum(output[:, np.arange(self.ngram.n), i].prod(dim=1) * self.dual[i] * self.ngram[i])
        return loss / batch_size

    def loss_dual(self, output, target):
        loss = self.loss_primal(output, target)
        for i in self.ngram:
            loss += torch.log(-self.dual[i]) * self.ngram[i]
        return -loss

    def clamp_dual(self):
        for idx in self.dual:
            if self.dual[idx].item() < -1.:
                self.dual[idx] = torch.clamp(self.dual[idx], -1., 0.).to(device).requires_grad_()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith('primal') and name.endswith('weight'):
                with torch.no_grad():
                    param.data.uniform_(-1.0/28,  1.0/28)
            if name.startswith('primal') and name.endswith('bias'):
                with torch.no_grad():
                    param.zero_()

    def compute_error_rate(self, test_loader):
        self.eval()
        num_errs = 0.0
        num_examples = 0
        with torch.no_grad():
            for x, y in data_loader:
                # x = x.to(device).view(-1, 1, 28, 28).float()
                x = x.to(device).view(-1, 28*28).float()
                y = y.to(device)
                outputs = self.forward(x)
                _, predictions = outputs.data.max(dim=1)
                num_errs += (predictions != y.data).sum().item()
                num_examples += x.size(0)
        self.train()
        return 100.0 * num_errs / num_examples


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
                x = x.cuda().view(x.size(0), -1)
                y = y.cuda()
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


def SPDG(model, optimizer_primal, optimizer_dual, data_loader, test_loader, num_epochs=5, log_every=1, test_every=1,
         history=None):
    if history is None:
        history = dict(err_rate=[], ploss=[], loss=[], test_err_rate=[], dual=[], predictions=[])
        for idx in model.dual:
            history['dual ' + str(idx)] = []
    model.train()
    iter_ = 0
    epoch_ = 0
    try:
        while epoch_ < num_epochs:
            if epoch_ % test_every == 0:
                msg = "Minibatch |   p-loss   |    loss    | err rate | steps/s |"
                for i in model.dual:
                    msg += " {} |".format(i)
                print(msg)
            epoch_ += 1
            stime = time.time()
            siter = iter_
            for x, y in data_loader:
                iter_ += 1
                x = x.cuda().view(-1, model.ngram.n, 28*28).float()
                y = y.cuda()
                out = model.forward_sequences(x)
                ploss = model.loss_primal(out, y)
                dloss = model.loss_dual(out, y)
                ploss_ = ploss.item()
                loss_ = -dloss.item()

                optimizer_primal.zero_grad()
                ploss.backward(retain_graph=True)
                optimizer_primal.step()

                optimizer_dual.zero_grad()
                dloss.backward()
                optimizer_dual.step()

#                 with torch.no_grad():
#                     model.clamp_dual()

                _, predictions = out.max(dim=2)
                a = predictions.view(-1) != y.view(-1)
                err_rate = 100.0 * a.sum().item() / (out.size(1) * out.size(0))
                history['err_rate'].append(err_rate)
                history['ploss'].append(ploss_)
                history['loss'].append(loss_)
                for idx in model.dual:
                    history['dual ' + str(idx)].append(model.dual[idx])

                if iter_ % log_every == 0:
                    num_iter = iter_ - siter
                    msg = "{:>8}  | {:>10.2e} | {:>10.2e} | {:>7.2f}% | {:>7.2f} |".format(iter_, ploss_, loss_, err_rate,
                                                                                           num_iter / (time.time() - stime))
                    for idx in model.dual:
                        msg += " {:>8.2f}  |".format(model.dual[idx])
                    print(msg)
                    siter = iter_
                    stime = time.time()
            if epoch_ % test_every == 0:
                history['predictions'].append(get_statistics(model))
                test_err_rate = model.compute_error_rate(test_loader)
                history['test_err_rate'].append(test_err_rate)
                msg = "Epoch {:>10} | Test error rate: {:.2f}".format(epoch_, test_err_rate)
                print('{0}\n{1}\n{0}'.format('---------------------------------------------------------+', msg))
    except KeyboardInterrupt:
        pass
    return history


# %% GENERATING DATASET
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

batch_size = 128
data_loader = train_loader_MNIST(batch_size)
test_loader = test_loader_MNIST(batch_size)
sequence_loader = sequence_loader_MNIST(batch_size, ngram, num_samples=20000)

# %% REGULAR TRAINING (SGD)
model = Model(ngram)
model.cuda()
model.init_weights()

optimizer = torch.optim.Adam(model.primal.parameters())
history = SGD(model, optimizer, data_loader, test_loader, num_epochs=3, log_every=50, test_every=1)

# %% DUAL TRAINING
model = Model(ngram, output_size=10)
model.cuda()
model.init_weights()

primal_lr = 1e-6
dual_lr = 1e-2

optimizer_primal = torch.optim.Adam(model.primal.parameters(), lr=primal_lr)
optimizer_dual = torch.optim.Adam(model.dual.parameters(), lr=dual_lr)

history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=10, log_every=20, test_every=1)


# %% DUAL TRAINING (CONTINUATION)
history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=20, log_every=100, test_every=1, history=history)


# %% DUAL TRAINING (CONTINUATION VERY LONG)
history = SPDG(model, optimizer_primal, optimizer_dual, sequence_loader,
               test_loader, num_epochs=1000, log_every=20, test_every=1, history=history)

# %% PLOTTING

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
