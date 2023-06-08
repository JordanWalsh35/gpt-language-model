import os
import sys
import numpy as np
from datasets import load_dataset

from encoder import BPETokenizer
 
current_dir = os.path.dirname(__file__)
home_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(home_dir)


data = load_dataset("openwebtext")
split_data = data["train"].train_test_split(test_size=0.0005, seed=3535, shuffle=True)

tokenizer = BPETokenizer()

def encode(example):
    pass

tokenized = split_data.map(function=encode, remove_columns=["text"], num_proc=8)

for split, dataset in tokenized.items():
    arr_len = np.sum(dataset["len"])
    filename = os.path.join