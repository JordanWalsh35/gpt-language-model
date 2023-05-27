import os
import numpy as np
from trainer import Trainer, TrainConfig
from model import GPTLanguageModel, GPTConfig

# Create directory where the binary files are stored
bin_dir = os.path.join(os.path.join(os.getcwd(), "data"), "bin")

# Define training and validation sets
train_data = np.memmap(os.path.join(bin_dir, "train.bin"), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(bin_dir, "val.bin"), dtype=np.uint16, mode='r')
