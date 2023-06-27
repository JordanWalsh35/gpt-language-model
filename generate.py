import torch
from model import GPTLanguageModel
from encoder import BPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BPETokenizer()


def generate(model, prompt='', num_samples=1, tokens=20):
    """ A function for generating text from our model. """

    if prompt == '':
        x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    else:
        x = tokenizer(prompt).to(device)

    x = x.expand(num_samples, -1)
    y = model.generate(x, max_new_tokens=tokens, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('_' * 80)
        print(out)