import torch

BATCH_SIZE = 4
IMG_SIZE = 256
NUM_EPOCHS = 10
NUM_WORKERS = 1
OPTIMIZER = "Adam"
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data/train/"
VAL_DIR = "data/valid/"
TEST_DIR = "data/test/"

LABELS = {
    "background": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "10": 11,  # +4 card
    "11": 12,  # +2 card
    "12": 13,  # Double arrow card
    "13": 14,  # Phi card
    "14": 15,  # Wildcard (colored circle)
}

NUM_LABELS = len(LABELS)

NEPTUNE_API_TOKEN = "your_api_key"
OUT_DIR = "./outputs"
