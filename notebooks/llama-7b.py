#!python
import torch
import thunder
from thunder.benchmarks.benchmark_litgpt import Benchmark_litGPT
#from thunder.benchmarks.targets import benchmark
from thunder.benchmarks import LitGPTBenchmark
from thunder.benchmarks import NanoGPTBenchmark
from thunder.tests.litgpt_model import Config, GPT, Block

#  model_name="Llama-3-70B",
#  distributed_mode="fsdp",
#b = Benchmark_litGPT(
#  compile="thunder+transformerengine+nvfuser",
#  low_precision_mode = "fp8-delayed-te",
#)
#
#b.train()

cfg: Config = Config.from_name("Llama-2-7b-hf")
b = LitGPTBenchmark(
#b = NanoGPTBenchmark(
  #config="Llama-2-7b-hf",
  cfg,
  #batchdims=(2,),
  device="cuda:0",
  dtype=torch.bfloat16,
  requires_grad=True,
)

args, kwargs = b.make_batch()
fn = thunder.jit(b.fn())

fn(*args, **kwargs)

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
