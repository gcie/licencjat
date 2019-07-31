import numpy as np
import torch

from config import DEVICE
from src.data_processing import test_loader_MNIST


def get_statistics(model, data_loader=None):
    if data_loader is None:
        data_loader = test_loader_MNIST()
    model.eval()
    num_errs = 0.0
    num_examples = 0
    results = np.zeros((10, model.output_size), dtype='int32')
    with torch.no_grad():
        for x, y in data_loader:
            # x = x.to(DEVICE).view(-1, 1, 28, 28).float()
            x = x.to(DEVICE).view(-1, 28*28).float()
            y = y.to(DEVICE)
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
