#!python
import time
import torch
import thunder
from thunder.benchmarks.benchmark_litgpt import Benchmark_litGPT
from thunder.benchmarks import LlamaMLPBenchmark
from thunder.tests.litgpt_model import Config, GPT, Block
from thunder.executors.transformer_engineex import transformer_engine_ex
from transformer_engine.common import recipe
from lightning.fabric.plugins.precision.transformer_engine import TransformerEnginePrecision
import transformer_engine.pytorch as te
import thunder.executors.torchex

#  model_name="Llama-3-70B",
#  distributed_mode="fsdp",
#b = Benchmark_litGPT(
#  compile="thunder+transformerengine+nvfuser",
#  low_precision_mode = "fp8-delayed-te",
#)
#
#b.train()

b = LlamaMLPBenchmark(
  config="Llama-2-7b-hf",
  batchdims=(16,),
  device="cuda:0",
  dtype=torch.bfloat16,
  requires_grad=True,
)

args, kwargs = b.make_batch()
eager_fqn = b.fn()
for i in range(5): eager_fqn(*args, **kwargs)
torch.cuda.synchronize()
start_eager = time.time_ns()
eager_result = eager_fqn(*args, **kwargs)
torch.cuda.synchronize()
end_eager = time.time_ns()
print(f"eager time: {(end_eager-start_eager) / 1e+6}ms")
del eager_fqn

torch_fqn = thunder.jit(
  b.fn(),
  executors = [thunder.get_executor("torch")]
)
torch_fqn(*args, **kwargs) # compile first
torch.cuda.synchronize()
start_th = time.time_ns()
torch_fqn(*args, **kwargs)
torch.cuda.synchronize()
end_th = time.time_ns()
print(f"thunder torch-only: {(end_th-start_th) / 1e+6}ms")
#print(thunder.last_traces(torch_fqn)[-1])
del torch_fqn

thfqn  = thunder.jit(b.fn())
thfqn(*args, **kwargs) # compile first
torch.cuda.synchronize()
start_th = time.time_ns()
thfqn(*args, **kwargs)
torch.cuda.synchronize()
end_th = time.time_ns()
print(f"thunder default: {(end_th-start_th) / 1e+6}ms")
#print(thunder.last_traces(thfqn)[-1])
del thfqn


fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
executors = [transformer_engine_ex] + list(thunder.get_default_executors())
th_te_fqn = thunder.jit(
  b.fn(),
  executors=executors,
  fp8_recipe=fp8_recipe,
)
th_te_fqn(*args, **kwargs) # compile first
torch.cuda.synchronize()
start_th = time.time_ns()
th_te_fqn(*args, **kwargs)
torch.cuda.synchronize()
end_th = time.time_ns()
print(f"thunder FP8 w/ TE: {(end_th-start_th) / 1e+6}ms")
#print(thunder.last_traces(th_te_fqn)[-1])


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
