import os
import sys
import numpy as np

from datasets import load_dataset
 
current_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(home_dir)

from encoder import BPETokenizer

data = load_dataset("openwebtext")
split_data = data["train"]
