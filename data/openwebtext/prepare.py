import os
import sys
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Put home directory on path to import encoder
home_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(home_dir)
from encoder import BPETokenizer
tokenizer = BPETokenizer()

# Load data from Hugging Face
data = load_dataset("openwebtext")
split_data = data["train"].train_test_split(test_size=0.0005, seed=3535, shuffle=True)
split_data["val"] = split_data.pop("test")

# Create function to apply tokenization to each dataset and add eot token after each example
def encode(input_dict):
    ids = tokenizer(input_dict["text"])
    ids.append(50256)
    out = {'ids': ids, 'len': len(ids)}
    return out

# Map function to datasets to perform tokenization
tokenized = split_data.map(function=encode, remove_columns=["text"], num_proc=8)

# Concat all ids in each dataset into one large file that we can use for training
for split, dataset in tokenized.items():
    arr_len = np.sum(dataset["len"])
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches)):
        # Batch together samples for faster write
        batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx:idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()