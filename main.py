import sys
import torch
from trainer import train_model
from runner import run_model

if not 3 <= len(sys.argv) <= 4:
    print('Usage: `python3 main.py [-train {csv_path}] [-load {model_path} {test}]')
    sys.exit(0)

# DEVICE detection
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

match sys.argv[1]:
    case '-train':
        train_model(sys.argv[2], DEVICE)
    case '-load':
        run_model(sys.argv[2], sys.argv[3], DEVICE)