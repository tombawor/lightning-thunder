#!python
import torch
import thunder
from thunder.benchmarks.benchmark_litgpt import Benchmark_litGPT
#from thunder.benchmarks.targets import benchmark
from thunder.benchmarks import LitGPTBenchmark
from thunder.benchmarks import NanoGPTBenchmark
from thunder.tests.litgpt_model import Config, GPT, Block

cfg: Config = Config.from_name("Llama-2-7b-hf")
b = LitGPTBenchmark(
  cfg,
  batchdims=(2,),
  device="cuda",
  dtype=torch.bfloat16,
  requires_grad=True,
)

args, kwargs = b.make_batch()
th_fqn = thunder.jit(b.fn())

def benchmark(fqn: callable) -> float:
  """Returns the time (in ms) to run the given function)"""
  for i in range(3):  # do a few warmup runs (for stable timings)
    foo = fqn(*args, **kwargs)
  torch.cuda.synchronize()
  start = time.time_ns()
  fqn(*args, **kwargs)
  torch.cuda.synchronize()
  end = time.time_ns()
  return ((end - start) / 1e+6)

eager_rt = benchmark(b.fn())
thunder_rt = benchmark(th_fqn)

print(f"Eager: {eager_rt}, thunder: {thunder_rt}")
