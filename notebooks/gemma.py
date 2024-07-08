#!python

from pathlib import Path

import torch
import thunder

import litgpt
from litgpt import Config, GPT, Tokenizer
from litgpt.generate.base import generate

# open_llama_7b would be better
cfg = Config.from_name("open_llama_3b")

device = "cpu"
# gemma = GPT(cfg).to(device)

# Run litgpt download openlm-research/open_llama_3b

tokenizer = Tokenizer('/Users/mruberry/Documents/git/lightning-thunder/checkpoints/openlm-research/open_llama_3b')

# Point to .pth file
state_dict = torch.load('/Users/mruberry/Documents/git/lightning-thunder/checkpoints/openlm-research/open_llama_3b/lit_model.pth')
model = GPT(cfg)
model.load_state_dict(state_dict)
model = model.to(device)
model.set_kv_cache(batch_size=1)

prompt = "What is 5 + 7?"
encoded = tokenizer.encode(prompt, bos=True, eos=False, device=device)
prompt_length = encoded.size(0)

max_new_tokens = 50
y = generate(model, encoded, max_new_tokens, top_k=1)
print(tokenizer.decode(y))


th_model = thunder.jit(model)
y = generate(model, encoded, max_new_tokens, top_k=1)
print(tokenizer.decode(y))


# options
#  * start by running the 7B run on a second GPU
#  * start with just an MLP
#    * show it executing with thunder
#  * here's how to do the same thing in t.c
#  * then:
#    * add in TE
#    * show the traces in both cases, highlight included TE
#  * go back to the initial run of the 7B. three options:
#    * eager running it
#    * thunder
#    * thunder + TE
