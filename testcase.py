
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cont", action="store_true")
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--save-every", default=100, type=int)
parser.add_argument("--log-every", default=100, type=int)
parser.add_argument("--test-every", default=5, type=int)
parser.add_argument("--primal-lr", default=1e-6, type=float)
parser.add_argument("--dual-lr", default=1e-4, type=float)