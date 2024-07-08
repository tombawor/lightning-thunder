#!python
from pathlib import Path
import torch
import thunder
import litgpt
from litgpt import Config, GPT, Tokenizer
from litgpt.generate.base import generate

cfg = Config.from_name("open_llama_7b")

model_path = Path("../../../checkpoints/openlm-research/open_llama_7b")
tokenizer = Tokenizer(model_path)
state_dict = torch.load(model_path / "lit_model.pth", weights_only=False)
device = torch.device("cuda:7")
model = GPT(cfg).to(device)
model.load_state_dict(state_dict)
model.set_kv_cache(batch_size=1, device=device)
model = model.to(device)

try:
  prompt = input("Model loaded.\n--> ").strip()
  while not prompt == "":
    encoded = tokenizer.encode(prompt, bos=False, eos=True, device=device)
    prompt_length = encoded.size(0)

    max_new_tokens = 256
    y = generate(model, encoded, max_new_tokens, top_k=1)
    print("eager:", tokenizer.decode(y))

    th_model = thunder.jit(model)
    y = generate(model, encoded, max_new_tokens, top_k=1)
    print("thunder.jit:", tokenizer.decode(y))

    prompt = input("--> ").strip()
except EOFError:
  print("")
