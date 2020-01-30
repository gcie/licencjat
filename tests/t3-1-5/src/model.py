import numpy as np
import torch
from torch import nn

from config import BATCH_SIZE, DEVICE
from src.data_processing import Ngram


class Model(nn.Module):
    """Neural net with dual parameters"""
    def __init__(self, ngram, architecture=None, output_size=10):
        super(Model, self).__init__()
        self.ngram = ngram
        self.n = ngram.n
        self.cnt = 1
        self.output_size = output_size
        for idx in ngram:
            ngram[idx]
        if architecture is None:
            self.primal = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, output_size)
            ).to(DEVICE)
        else:
            self.primal = architecture.to(DEVICE)
        self.dual = Ngram(self.ngram.n)
        for idx in self.ngram:
            self.dual[idx] = torch.tensor(-1. / self.ngram[idx]).to(DEVICE).requires_grad_()
            # self.dual[idx] = torch.tensor(0.).uniform_(-1, 0).to(DEVICE).requires_grad_()
        self.to(DEVICE)
        self.init_weights()

    def forward_sequences(self, x):
        return torch.nn.functional.softmax(self.primal.forward(x), dim=2)

    def forward(self, x):
        return torch.nn.functional.softmax(self.primal.forward(x), dim=0)

    def loss(self, output, target):
        return torch.mean(-torch.log(torch.gather(output, 1, target.unsqueeze(1))))

    def loss_primal(self, output):
        loss = torch.tensor(0, dtype=torch.float32).to(DEVICE)
        for i in self.ngram:
            loss += torch.sum(output[:, np.arange(self.n), i].prod(dim=1) * self.dual[i] * self.ngram[i])
        return loss / BATCH_SIZE

    def loss_dual(self, output):
        loss = self.loss_primal(output)
        for i in self.ngram:
            loss += torch.log(-self.dual[i]) * self.ngram[i]
        return -loss

    def clamp_dual(self):
        for idx in self.dual:
            if self.dual[idx].item() < -1.:
                self.dual[idx] = torch.clamp(self.dual[idx], -1., 0.).to(DEVICE).requires_grad_()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith('primal') and name.endswith('weight'):
                with torch.no_grad():
                    param.data.uniform_(-1.0/28,  1.0/28)
            if name.startswith('primal') and name.endswith('bias'):
                with torch.no_grad():
                    param.zero_()

    def compute_error_rate(self, data_loader):
        self.eval()
        num_errs = 0.0
        num_examples = 0
        with torch.no_grad():
            for x, y in data_loader:
                # x = x.to(device).view(-1, 1, 28, 28).float()
                x = x.to(DEVICE).view(-1, 28*28).float()
                y = y.to(DEVICE)
                outputs = self.forward(x)
                _, predictions = outputs.data.max(dim=1)
                num_errs += (predictions != y.data).sum().item()
                num_examples += x.size(0)
        self.train()
        return 100.0 * num_errs / num_examples
