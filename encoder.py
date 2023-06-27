import os
import requests
import json
import regex as re
import torch


def bytes_to_unicode():
    """ 
    One-to-One mapping of bytes to unicode characters.
    """
    # List of clean characters
    ints = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = ints[:]

    # Map remaining by shifting them by 2**8
    n = 0
    for i in range(2**8):
        if i not in ints:
            ints.append(i)
            chars.append(2**8 + n)
            n += 1
    chars = [chr(n) for n in chars]
    enc = dict(zip(ints, chars))

    return enc



def get_pairs(word):
    """
    Returns all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs



class Encoder:
    def __init__(self, encoder, bpe_merges):
        # Byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}

        # BPE token encoder/decoder
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}

        # BPE merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # Regex pattern used to split tokens
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}
    

    def bpe(self, token):
        """
        This function uses self.bpe_ranks to iteratively merge all the possible bpe tokens up the tree. 'token' is a string
        of one individual 'word' (after regex tokenization) and after byte encoding.
        """
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)      # individual characters that make up the token, in a tuple
        pairs = get_pairs(word)  # get all bigrams

        if not pairs:
            return token
        
        while True:
            # Find the next lowest rank bigram that can be merged
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break   # no more bigrams are eligible to be merged
            first, second = bigram

            # We now replace all occurences of (first, second) in the list of current words into one merged token first_second
            new_word = []
            i = 0
            while i < len(word):
                # Find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # If this occurence is also followed by a second, then merge them into one
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            # All occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # Concat all words into a string and cache the result
        word = ' '.join(word)
        self.cache[token] = word

        return word


    def encode(self, text):
        """ String goes in, list of integers comes out. """
        bpe_idx = []

        # Pre-tokenize the input text into string tokens
        tokens = re.findall(self.pattern, text)

        # Process each token into BPE integers
        for token in tokens:
            # Encode the token as a byte object
            token_bytes = token.encode('utf-8')
            # Translate all bytes to their unicode string representation and flatten
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # Perform all the applicable bpe merges according to self.bpe_ranks
            token_merged = self.bpe(token_translated).split(' ')
            # Translate all bpe tokens to integers
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # Extend our running list of all output integers
            bpe_idx.extend(token_ix)
        
        return bpe_idx


    def decode(self, bpe_idx):
        """ List of integers come in, string comes out. """
        # Inverse map the integers to get the tokens
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # Inverse the byte encoder
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # Recover the full utf-8 string
        text = tokens_bytes.decode('utf-8', errors="replace")

        return text
    


def get_encoder():
    """ Returns an instance of the GPT BPE Encoder/Decoder and handles caching of "database" files. """

    # Load gpt_encoder.json and vocab.bpe (from OpenAI)
    inputs_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), "encoder_input")
    encoder_path = os.path.join(inputs_dir, "gpt_encoder.json")
    vocab_path = os.path.join(inputs_dir, "vocab.bpe")
    with open(encoder_path, 'r') as f:
        encoder = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        bpe_data = f.read()
        
    # Light postprocessing: strip the version on first line and the last line is a blank
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    # Construct the Encoder object and return
    enc = Encoder(encoder, bpe_merges)
    
    return enc



class BPETokenizer:
    """ This class wraps the Encoder above """

    def __init__(self):
        self.encoder = get_encoder()   

    def __call__(self, text):
        # Encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # Wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # Ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # Decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text