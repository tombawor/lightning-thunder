#!python
import time

import torch
import thunder

from thunder.executors.transformer_engineex import transformer_engine_ex
from transformer_engine.common import recipe

from litgpt import Config

import torch.nn as nn
device = "cuda"
dtype = torch.bfloat16

# LLaMA MLP module
class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

# Contructs the mlp
config = Config.from_name("open_llama_3b")
mlp = LLaMAMLP(config).to(device=device, dtype=dtype).requires_grad_(True)

# Generates input
batch_size = 1 # 16
shape = (1,) + (config.block_size, config.n_embd)
x = torch.testing.make_tensor(shape, device=device, dtype=dtype, requires_grad=True)
torch.cuda.synchronize()

# Runs eagerly
start_eager = time.time_ns()
eager_result = mlp(x)
torch.cuda.synchronize()
end_eager = time.time_ns()
print(f"eager time: {(end_eager-start_eager) / 1e+6}ms")

# Runs using default thunder executors (uses nvFuser executor)
thmlp  = thunder.jit(mlp)
thmlp(x) # compile first
torch.cuda.synchronize()
start_th = time.time_ns()
thmlp(x)
torch.cuda.synchronize()
end_th = time.time_ns()
print(f"thunder default: {(end_th-start_th) / 1e+6}ms")
#print(thunder.last_traces(thmlp)[-1])
del thmlp

# Runs using the TransformerEnginer executor
# (automatically transforms the MLP to use transformer engine layers)
fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
executors = [transformer_engine_ex] + list(thunder.get_default_executors())
te_mlp = thunder.jit(
  mlp,
  executors=executors,
  fp8_recipe=fp8_recipe,
)
te_mlp(x) # compile first
torch.cuda.synchronize()
start_th = time.time_ns()
te_mlp(x)
torch.cuda.synchronize()
end_th = time.time_ns()
print(f"thunder FP8 w/ TE: {(end_th-start_th) / 1e+6}ms")
print(thunder.last_traces(te_mlp)[-1])


#def fn(x, w1, w2):
#  o = torch.nn.functional.linear(x, w1)
#  return torch.nn.functional.linear(o + x, w2)
#
#dtype = torch.bfloat16
#device = "cuda"

# TE inputs (3D input)
#x_te = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=False)
#
#te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
#te_linear2 = te.Linear(4096, 4096, params_dtype=dtype)
# 
## thunder inputs

##x = x_te.detach().clone()
#x.requires_grad_(False)
#w1 = te_linear1.weight.detach().clone()
#w1.requires_grad_(False)
#w2 = te_linear2.weight.detach().clone()
#w2.requires_grad_(False)
# 
#cfn = thunder.jit(fn, executors=[transformer_engine_ex], fp8_recipe=fp8_recipe)
#cfn(x, w1, w2)
#print(thunder.last_traces(cfn)[-1])

#fn(*args, **kwargs)

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
