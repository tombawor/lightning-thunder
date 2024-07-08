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
tc_fqn = torch.compile(b.fn())
for i in range(5): tc_fqn(*args, **kwargs)
torch.cuda.synchronize()
start_eager = time.time_ns()
eager_result = tc_fqn(*args, **kwargs)
torch.cuda.synchronize()
end_eager = time.time_ns()
print(f"tc time: {(end_eager-start_eager) / 1e+6}ms")
