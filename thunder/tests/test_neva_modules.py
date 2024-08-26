import torch
import torch.distributed

import thunder
from thunder.core import devices
from thunder.tests.distributed.helper import init_per_process_distributed, distributed_wrapper
from thunder.tests.framework import instantiate, TorchExecutor

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import init_method_normal, scaled_init_method_normal, make_viewless_tensor

# Config extracted from running NeMo with instructions: https://github.com/Lightning-AI/lightning-thunder/issues/343
transformer_config = TransformerConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    sequence_parallel=False,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    moe_extended_tp=False,
    perform_initialization=True,
    use_cpu_initialization=False,
    fp16=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    timers=None,
    finalize_model_grads_func=None,
    grad_scale_func=None,
    no_sync_func=None,
    grad_sync_func=None,
    param_sync_func=None,
    deterministic_mode=False,
    enable_autocast=False,
    autocast_dtype=torch.bfloat16,
    num_microbatches_with_partial_activation_checkpoints=None,
    gradient_accumulation_fusion=False,
    async_tensor_model_parallel_allreduce=False,
    use_te_rng_tracker=False,
    tp_comm_overlap=False,
    tp_comm_bulk_wgrad=True,
    tp_comm_bulk_dgrad=True,
    tp_comm_overlap_ag=True,
    tp_comm_overlap_rs=True,
    tp_comm_overlap_rs_dgrad=False,
    tp_comm_split_ag=True,
    tp_comm_atomic_ag=False,
    tp_comm_split_rs=True,
    tp_comm_atomic_rs=False,
    cross_entropy_loss_fusion=False,
    # tp_comm_overlap_disable_qkv=False,
    # tp_comm_overlap_disable_fc1=False,
    pipeline_dtype=torch.bfloat16,
    variable_seq_lengths=False,
    overlap_p2p_comm=False,
    batch_p2p_comm=True,
    batch_p2p_sync=True,
    use_ring_exchange_p2p=False,
    deallocate_pipeline_outputs=True,
    defer_embedding_wgrad_compute=False,
    wgrad_deferral_limit=0,
    pipeline_model_parallel_split_rank=None,
    cpu_offloading=False,
    cpu_offloading_num_layers=0,
    _cpu_offloading_context=None,
    cpu_offloading_activations=True,
    cpu_offloading_weights=True,
    barrier_with_L1_time=True,
    num_layers=2,
    hidden_size=5120,
    num_attention_heads=40,
    num_query_groups=40,
    ffn_hidden_size=13824,
    kv_channels=128,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    fp32_residual_connection=False,
    apply_residual_connection_post_layernorm=False,
    layernorm_epsilon=1e-05,
    layernorm_zero_centered_gamma=False,
    add_bias_linear=False,
    add_qkv_bias=False,
    gated_linear_unit=True,
    activation_func=torch.nn.functional.silu,
    activation_func_fp8_input_store=False,
    num_moe_experts=None,
    rotary_interleaved=False,
    window_size=None,
    normalization="RMSNorm",
    qk_layernorm=False,
    test_mode=False,
    calculate_per_token_loss=False,
    init_method=init_method_normal(0.014),
    output_layer_init_method=scaled_init_method_normal(0.014, 2),
    init_method_std=0.014,
    apply_query_key_layer_scaling=False,
    attention_softmax_in_fp32=False,
    bias_activation_fusion=False,
    masked_softmax_fusion=True,
    persist_layer_norm=True,
    memory_efficient_layer_norm=False,
    bias_dropout_fusion=False,
    apply_rope_fusion=False,
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    distribute_saved_activations=False,
    fp8=None,
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1,
    fp8_amax_compute_algo="most_recent",
    fp8_wgrad=True,
    fp8_dot_product_attention=False,
    fp8_multi_head_attention=False,
    moe_router_load_balancing_type="aux_loss",
    moe_router_topk=2,
    moe_router_pre_softmax=False,
    moe_grouped_gemm=False,
    moe_aux_loss_coeff=0,
    moe_z_loss_coeff=None,
    moe_input_jitter_eps=None,
    moe_token_dropping=False,
    moe_token_dispatcher_type="allgather",
    moe_per_layer_logging=False,
    moe_expert_capacity_factor=None,
    moe_pad_expert_input_to_capacity=False,
    moe_token_drop_policy="probs",
    moe_layer_recompute=False,
    clone_scatter_output_in_embedding=True,
    disable_parameter_transpose_cache=False,
    enable_cuda_graph=False,
    # config_logger_dir="",
)


def init_megatron_module_test(input_data):
    ps.destroy_model_parallel()
    init_method, world_size, rank, executor, device, dtype, kwargs = input_data
    devicetype = devices.device_from_string(device).devicetype

    pg = init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.distributed.barrier(pg)

    ps.initialize_model_parallel(1, 1, None)
    model_parallel_cuda_manual_seed(0)

    return init_method, world_size, rank, executor, device, dtype, kwargs



def _test_megatron_transformer_block(input_data):
    init_method, world_size, rank, executor, device, dtype, kwargs = init_megatron_module_test(input_data)
    device = devices.device_from_string(device).type
    import traceback

    block = TransformerBlock(transformer_config, get_gpt_layer_with_transformer_engine_spec())

    block.to(device)
    jblock = thunder.jit(block)
    hidden_states = torch.ones((4096, 1, transformer_config.hidden_size))
    hidden_states = hidden_states.cuda()

    attention_mask = torch.ones((1, 1, 4096, 4096), dtype=bool).cuda()

    # Comment this function out to repro https://github.com/Lightning-AI/lightning-thunder/issues/753
    @thunder.core.jit_ext.register_general_jit_lookaside(make_viewless_tensor)
    @thunder.core.jit_ext.interpreter_needs_wrap
    def make_viewless_tensor_lookaside(inp, requires_grad, keep_graph):
        return inp

    exception_list = []
    try:
        hidden_states = jblock(hidden_states=hidden_states, attention_mask=attention_mask)
    except Exception as e:
        if rank == 0:
            print(*traceback.format_exception(e))
            exception_list.append(e)

    return exception_list


@instantiate(
    dtypes=(thunder.bfloat16,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA, devices.DeviceType.CPU),
    executors=(TorchExecutor,),
)
@distributed_wrapper("test_megatron_transformer_block", _test_megatron_transformer_block)
def test_megatron_transformer_block(executor, devices, dtype):
    pass

