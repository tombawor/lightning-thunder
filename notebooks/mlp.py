#!python
import time
import torch
import thunder
from thunder.executors.transformer_engineex import transformer_engine_ex
from transformer_engine.common import recipe
from litgpt import Config
import torch.nn as nn

# LLaMA MLP module
class LLaMAMLP(nn.Module):
    def __init__(self, n_embd: int, intermediate_size: int):
        super().__init__()
        bias = False
        self.fc_1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.fc_2 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.proj = nn.Linear(intermediate_size, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

# Contructs the mlp
device = "cuda:0"
dtype = torch.bfloat16
n_embd = 3200*2
intermediate_size = 8640*8
block_size = 2048
mlp = LLaMAMLP(n_embd, intermediate_size).to(device=device, dtype=dtype).requires_grad_(True)

# Generates input
batch_size = 1 # 16
shape = (1,) + (block_size, n_embd)
x = torch.testing.make_tensor(shape, device=device, dtype=dtype, requires_grad=True)
torch.cuda.synchronize()

def benchmark(fqn: callable) -> float:
  """Returns the time (in ms) to run the given function)"""
  for i in range(3):  # do a few warmup runs (for stable timings)
    foo = fqn(x)
  torch.cuda.synchronize()
  start = time.time_ns()
  fqn(x)
  torch.cuda.synchronize()
  end = time.time_ns()
  return ((end - start) / 1e+6)

# First run eager mode to establish a baseline:
eager_runtime = benchmark(mlp)
print(f"eager: {eager_runtime}ms")

# Runs using default thunder executors (uses nvFuser executor)
thmlp = thunder.jit(mlp)
th_default_runtime = benchmark(thmlp)
print(f"thunder default: {th_default_runtime}ms")
th_trace = thunder.last_traces(thmlp)[-1]
del thmlp
input("Waiting... ")
print("""
  self.fc_1 = nn.Linear(n_embd, intermediate_size, bias=bias)
  self.fc_2 = nn.Linear(n_embd, intermediate_size, bias=bias)
  self.proj = nn.Linear(intermediate_size, n_embd, bias=bias)
  ...
  x_fc_1 = self.fc_1(x)
  x_fc_2 = self.fc_2(x)
  x = torch.nn.functional.silu(x_fc_1) * x_fc_2
  return self.proj(x)
""")
input("")
print(th_trace)

# Runs using the TransformerEnginer executor
# (automatically transforms the MLP to use transformer engine layers)
fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
te_mlp = thunder.jit(mlp, fp8_recipe=fp8_recipe)
th_fp8_runtime = benchmark(te_mlp)
fp8_trace = thunder.last_traces(te_mlp)[-1]
input("")
print(fp8_trace)
input("")
print(f"eager: {eager_runtime}ms")
print(f"thunder default: {th_default_runtime}ms")
print(f"thunder with TransformerEngine: {th_fp8_runtime}ms")
del te_mlp
del th_fp8_runtime
