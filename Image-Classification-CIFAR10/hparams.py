import os
import torch

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

config = {
    "RANDOM_SEED": 42,
    "AVAIL_GPUS": AVAIL_GPUS,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_WORKERS": NUM_WORKERS,
    "LEARNING_RATE": 0.05,
    "EarlyStoppingPatience": 20,
}
