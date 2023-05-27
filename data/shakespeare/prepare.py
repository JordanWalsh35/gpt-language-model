import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(home_dir)

from encoder import BPETokenizer

with open("data/shakespeare/shakespeare.txt", "r") as file:
    data = file.read()

# Create datasets
l = len(data)
n = int(0.9 * l)
train_data = data[:n]
val_data = data[n:]


# Encode the data
tokenizer = BPETokenizer()
train_ids = tokenizer(train_data).reshape(-1)
val_ids = tokenizer(val_data).reshape(-1)
print(f"the training data has {len(train_ids)} tokens")
print(f"the validation data has {len(val_ids)} tokens")


# Export as bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(current_dir, "train.bin"))
val_ids.tofile(os.path.join(current_dir, "val.bin"))
