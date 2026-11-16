import torch
import triton
import triton.language as tl
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from .pi0_infer import (
    pdl_kernel, pdl_scope, pdl_wait, pdl_trigger,
    vision_encoder,
    rms_norm_kernel,
    matmul_n_2048_2560_qkv_rope,
    rms_matmul_packed_gate,
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    matmul_n_2048_2048_res,
    matmul_n_16384_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_small_bias,
    matmul_small_bias_res,
    matmul_small_bias_silu,
    matmul_small_gate,
    matmul_k8_n_256,
    matmul_abT_scale,
)

@pdl_kernel
@triton.jit
def matmul_small_res_gate(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr, USE_PDL : tl.constexpr = False):
    pdl_wait(USE_PDL)
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)
        
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)

        acc += matmul_acc * gate
        
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )
    pdl_trigger(USE_PDL)

def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )

@pdl_kernel
@triton.jit
def adarms_norm_kernel(
    x_ptr,
    style_ptr,
    normed_x_ptr, 
    gate_ptr, 
    seq_len: tl.constexpr, 
    features: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += x_val * x_val
        
        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)

        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_val * rms_factor
            s_scale = tl.load(style_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            s_shift = tl.load(style_ptr + features + cols, mask=mask, other=0.0).to(tl.float32)
            s_gate = tl.load(style_ptr + 2 * features + cols, mask=mask, other=0.0).to(tl.float32)

            output_val = x_norm * (1.0 + s_scale) + s_shift

            tl.store(normed_x_ptr + row_x_offset + cols, output_val.to(tl.bfloat16), mask=mask)
            tl.store(gate_ptr + row_x_offset + cols, s_gate.to(tl.bfloat16), mask=mask)
    pdl_trigger(USE_PDL)

def matmul_1_1024_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )

@pdl_kernel
@triton.jit
def matmul_rope_qkv(
    inp_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    weight_qkv_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = (offs_i < seq_len) & (offs_k[None, :] < features),
                other = 0,
            )
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = (offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                other = 0
            )
            accumulator = tl.dot(x, w, accumulator)

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize
    pdl_trigger(USE_PDL)

def matmul_k_1024_2560_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V, optimized=False):
    if optimized:
        seq_len = x_normed.shape[0]
        matmul_rope_qkv[(triton.cdiv(seq_len, 32) * (2560 // 32),)](
            x_normed, seq_len, 1024, 256, 8, weight_qkv, rope_weight, Q, K, V,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, num_warps=4, num_stages=3
        )
        return
    seq_len = x_normed.shape[0]
    matmul_rope_qkv[(128,)](
        x_normed, seq_len, 1024, 256, 8,
        weight_qkv, rope_weight, Q, K, V,
    )

def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]
    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 512
    )
def adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]

    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 512
    )

def adarms_matmul_k_1024_32_bias_res(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
    res,
):
    adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    seq_len = x.shape[0]
    matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),] (
        x_normed,
        weight,
        out,
        bias,
        res,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )

def matmul_k_2048_1024_gate(x, weight, out, gate, optimized=False):
    if optimized:
        seq_len = x.shape[0]
        matmul_grouped_res_gate[(triton.cdiv(seq_len, 16) * (1024 // 32),)](
            x, weight, out, gate, out, seq_len, 2048, 1024,
            16, 32, 256, 2, num_warps=4, num_stages=4
        )
        return
    seq_len = x.shape[0]
    matmul_small_res_gate[(128,)](
        x,
        weight,
        out,
        out, 
        gate,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 128
    )

def matmul_k_4096_1024_gate(x, weight, out, gate, optimized=False):
    if optimized:
        seq_len = x.shape[0]
        matmul_grouped_res_gate[(triton.cdiv(seq_len, 16) * (1024 // 32),)](
            x, weight, out, gate, out, seq_len, 4096, 1024,
            16, 32, 256, 2, num_warps=4, num_stages=4
        )
        return
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 256
    )

@pdl_kernel
@triton.jit
def softmax_kernel_masklen(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys, f"BLOCK_SIZE must be >= keys, got {BLOCK_SIZE} < {keys}"

    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16),
                 mask=(offs_i < queries) & (offs_j < keys))
    pdl_trigger(USE_PDL)

@pdl_kernel
@triton.jit
def matmul_grouped_res_gate(
    inp_ptr, weight_ptr, res_ptr, gate_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr,
    hidden : tl.constexpr, BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr, GROUP_M : tl.constexpr,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    gm = tl.cdiv(seq_len, BLOCK_SIZE_N)
    gn = tl.cdiv(hidden, BLOCK_SIZE_M)
    for pid in range(tl.program_id(0), gm * gn, tl.num_programs(0)):
        group = pid // (GROUP_M * gn)
        first = group * GROUP_M
        sz = tl.minimum(gm - first, GROUP_M)
        mi = first + pid % (GROUP_M * gn) % sz
        ni = pid % (GROUP_M * gn) // sz
        r = mi * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c = ni * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        kk = tl.arange(0, BLOCK_SIZE_K)
        res = tl.load(res_ptr + r[:, None] * hidden + c[None, :], r[:, None] < seq_len, other=0).to(tl.float32)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), tl.float32)
        for off in tl.range(0, features, BLOCK_SIZE_K, loop_unroll_factor=1):
            a = tl.load(inp_ptr + r[:, None] * features + off + kk[None, :], r[:, None] < seq_len, other=0)
            b = tl.trans(tl.load(weight_ptr + c[:, None] * features + off + kk[None, :]))
            acc = tl.dot(a, b, acc)
        g = tl.load(gate_ptr + r[:, None] * hidden + c[None, :], r[:, None] < seq_len, other=0).to(tl.float32)
        acc = res + acc * g
        tl.store(out_ptr + r[:, None] * hidden + c[None, :], acc.to(tl.bfloat16), r[:, None] < seq_len)
    pdl_trigger(USE_PDL)

@pdl_kernel
@triton.jit
def matmul_abT_scale_masked(
    q_ptr, k_ptr, out_ptr, M : tl.constexpr, N : tl.constexpr, K : tl.constexpr, scale_factor : tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_K : tl.constexpr,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    pid = tl.program_id(0)
    gn = tl.cdiv(N, BLOCK_SIZE_N)
    r = pid // gn * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c = pid % gn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    kk = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for off in range(0, K, BLOCK_SIZE_K):
        a = tl.load(q_ptr + r[:, None] * K + off + kk[None, :], (r[:, None] < M) & (off + kk[None, :] < K), other=0)
        b = tl.load(k_ptr + c[:, None] * K + off + kk[None, :], (c[:, None] < N) & (off + kk[None, :] < K), other=0)
        acc = tl.dot(a, tl.trans(b), acc)
    value = (acc * scale_factor).to(tl.bfloat16)
    tl.store(out_ptr + r[:, None] * N + c[None, :], value, (r[:, None] < M) & (c[None, :] < N))
    pdl_trigger(USE_PDL)

@triton.jit
def prepare_inputs(
    images_ptr, noise_ptr, rope_table_ptr, images_out_ptr, noise_out_ptr, rope_out_ptr, valid_len_ptr,
    image_size : tl.constexpr, noise_size : tl.constexpr, rope_size : tl.constexpr,
    rope_start : tl.constexpr, BLOCK_SIZE : tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_SIZE)
    IMAGE_BLOCKS: tl.constexpr = (image_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    NOISE_BLOCKS: tl.constexpr = (noise_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid < IMAGE_BLOCKS:
        ii = pid * BLOCK_SIZE + r
        iv = tl.load(images_ptr + ii, ii < image_size, other=0)
        tl.store(images_out_ptr + ii, iv, ii < image_size)
    elif pid < IMAGE_BLOCKS + NOISE_BLOCKS:
        ni = (pid - IMAGE_BLOCKS) * BLOCK_SIZE + r
        nv = tl.load(noise_ptr + ni, ni < noise_size, other=0)
        tl.store(noise_out_ptr + ni, nv, ni < noise_size)
    else:
        ri = (pid - IMAGE_BLOCKS - NOISE_BLOCKS) * BLOCK_SIZE + r
        rv = tl.load(rope_table_ptr + rope_start * 256 + ri, ri < rope_size, other=0)
        tl.store(rope_out_ptr + ri, rv, ri < rope_size)
    if pid == 0:
        tl.store(valid_len_ptr, rope_start)

def num_snapshot_tokens(buffers):
    return buffers['vision_prefix_snapshot'].shape[0]

def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm']
    )
    if 'vision_prefix_snapshot' in buffers:
        buffers['vision_prefix_snapshot'].copy_(buffers['encoder_x'][:num_snapshot_tokens(buffers)])
    for i in range(18):
        if buffers.get('optimized', False) and i == 17:
            rms_norm_kernel[(encoder_seq_len,)](
                buffers['encoder_x'], buffers['encoder_x_norm'], encoder_seq_len, 2048
            )
            matmul_n_2048_2560_qkv_rope[(triton.cdiv(encoder_seq_len, 64), 8)](
                buffers['encoder_x_norm'], weights['encoder_attn_kv_w'],
                buffers['encoder_rope_weights'], buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len], buffers['encoder_V'][i, :encoder_seq_len],
                encoder_seq_len, 2048, 256, 0
            )
        else:
            rms_matmul_n_2048_2560_qkv_rope(
                buffers['encoder_x'],
                weights['encoder_attn_qkv_w'][i],
                buffers['encoder_rope_weights'],
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_x_norm']
            )
        if 'encoder_events' in buffers:
            buffers['encoder_events'][i].record()
        if i != 17:
            scale = 1.0 / (256 ** 0.5)
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            optimized = buffers.get('optimized', False)
            qk_kernel = matmul_abT_scale_masked if optimized else matmul_abT_scale
            block_m = 64 if optimized else 32
            block_n = 64 if optimized else 32
            qk_kernel[(triton.cdiv(total_queries, block_m) * triton.cdiv(total_keys, block_n),)](
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                scale,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=64,
                num_warps=4, num_stages=3,
            )
            softmax_kernel_masklen[((total_queries + 3) // 4,)](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_k8_n_256(
                buffers['encoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_ctx_buf'],
                tile=(64, 64, 32, 4, 3) if buffers.get('optimized', False) else None,
            )
            
            matmul_n_2048_2048_res(
                buffers['encoder_ctx_buf'].view(-1, 2048),
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x']
            )
        
            if buffers.get('optimized', False):
                rms_matmul_packed_gate(
                    buffers['encoder_x'], weights['encoder_ffn_packed_w'][i],
                    buffers['encoder_hidden'], buffers['encoder_x_norm'],
                    (128, 32, 32, 4, 3)
                )
            else:
                rms_matmul_n_2048_16384_gate(
                    buffers['encoder_x'],
                    weights['encoder_ffn_gate_w'][i],
                    weights['encoder_ffn_up_w'][i],
                    buffers['encoder_hidden'],
                    buffers['encoder_x_norm']
                )

            matmul_n_16384_2048_res(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x'],
                group=2 if buffers.get('optimized', False) else None,
            )

@pdl_kernel
@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
    USE_PDL : tl.constexpr = False,
):
    pdl_wait(USE_PDL)
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)
    pdl_trigger(USE_PDL)

def transformer_decoder(weights, buffers, encoder_seq_len, num_steps=10, start_step=0):
    for step in range(start_step, start_step + num_steps):
        matmul_k_32_1024_bias(
            buffers['diffusion_noise'].to(torch.bfloat16),
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x']
        )
        seq_len = buffers['decoder_x'].shape[0]
        for i in range(18):
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_pre_attn_norm_mod_w'][i],
                weights['decoder_pre_attn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_attn'][step, i]
            )
            matmul_k_1024_2560_qkv_rope(
                buffers['x_normed_buf'], 
                weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                optimized=buffers.get('optimized', False),
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            if step == 0 and 'encoder_events' in buffers:
                torch.cuda.current_stream().wait_event(buffers['encoder_events'][i])
            optimized = buffers.get('optimized', False)
            qk_kernel = matmul_abT_scale_masked if optimized else matmul_abT_scale
            block_m = 32
            block_n = 64 if optimized else 32
            qk_kernel[(triton.cdiv(total_queries, block_m) * triton.cdiv(total_keys, block_n),)](
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                buffers['decoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                256 ** -0.5,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=64,
                num_warps=4, num_stages=3,
            )

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )

            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
                tile=(32, 32, 128, 4, 3) if buffers.get('optimized', False) else None,
            )
            matmul_k_2048_1024_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_packed_w'][i] if buffers.get('optimized', False) else weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf'],
                optimized=buffers.get('optimized', False),
            )
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_pre_ffn_norm_mod_w'][i],
                weights['decoder_pre_ffn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_ffn'][step, i]
            )
            seq_len = buffers['decoder_x'].shape[0]
            block_n = 32 if buffers.get('optimized', False) else 128
            block_k = 64 if buffers.get('optimized', False) else 32
            matmul_small_gate[(triton.cdiv(seq_len, block_n), 4096 // 64)](
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                seq_len,
                1024,
                4096,
                BLOCK_SIZE_N=block_n, BLOCK_SIZE_M=64, BLOCK_SIZE_K=block_k,
                num_warps=4, num_stages=3,
            )
            matmul_k_4096_1024_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_packed_w'][i] if buffers.get('optimized', False) else weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf'],
                optimized=buffers.get('optimized', False),
            )

        if 'action_update' in buffers:
            buffers['action_update'](weights, buffers, step)
            continue
        adarms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'],
            buffers['decoder_time_emb'][step],
            weights['decoder_final_norm_mod_w'],
            weights['decoder_final_norm_mod_b'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style_final'][step],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['diffusion_noise'],
        )

def pi05_model(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len, num_steps)

class Pi05Inference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
        optimized: bool = True,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        print(f"max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")
        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),

            "decoder_time_embeds":                torch.zeros(10, 1024,                  dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_w":              torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_b":              torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_w":             torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_b":             torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_w":           torch.empty(32, 1024,                      dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                          dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_w":        torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_attn_norm_mod_b":        torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_w":         torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_ffn_norm_mod_b":         torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_w":          torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_b":          torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),
            "decoder_final_norm_mod_w":           torch.empty(1024, 3 * 1024,         dtype=torch.bfloat16, device="cuda"), 
            "decoder_final_norm_mod_b":           torch.empty(3 * 1024,               dtype=torch.bfloat16, device="cuda"), 
            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),

            
        }

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size 

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'diffusion_noise':                    torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'vision_x':                           torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_norm':                      torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_QKV':                         torch.empty(num_views, 256, 3 * 1152,        dtype = torch.bfloat16, device = "cuda"),
            'vision_hidden':                      torch.empty(num_views, 256, 4304,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_split_k_buf':               torch.empty((num_views * 256 * 1152 * 4,),   dtype = torch.float32, device = "cuda"),
            'encoder_rope_weights':               torch.empty(encoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'encoder_x':                          torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_x_norm':                     torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_K':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_V':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_Q':                          torch.empty(encoder_seq_len * 8, 256,        dtype = torch.bfloat16, device = "cuda"),
            'encoder_hidden':                     torch.empty(encoder_seq_len, 16384,          dtype = torch.bfloat16, device = "cuda"),
            'valid_encoder_len':                  torch.empty((1,),                           dtype = torch.int32, device = "cuda"),
            'encoder_logits_buf':                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32,  device="cuda"),
            'encoder_attn_buf':                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            'encoder_ctx_buf':                    torch.empty((encoder_seq_len * 8, 256),     dtype=torch.bfloat16, device="cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((decoder_seq_len, 1024),         dtype=torch.bfloat16,  device = "cuda"),
            'decoder_action_buf':                 torch.empty((decoder_seq_len, 32),           dtype = torch.bfloat16, device = "cuda"),
            'decoder_time_emb':                   torch.empty((10, decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_attn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_ffn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_final':                      torch.empty((10, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),            
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_logits_buf':                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.float32, device="cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
            'x_normed_buf':                       torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'gate_buf':                           torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers['encoder_rope_weights'].copy_(self._rope_table[:prefix_alloc])

        self.buffers['valid_encoder_len'].fill_(self.num_views * 256 + 1)
        for k, v in checkpoint.items():
            if k != "embedding_weight":
                self.weights[k].copy_(v)
        num_steps = 10
        self.weights['decoder_action_out_proj_w'] *= -1.0 / num_steps
        self.weights['decoder_action_out_proj_b'] *= -1.0 / num_steps 

        for step in range(num_steps):
            matmul_1_1024_1024_bias_silu(
                self.weights['decoder_time_embeds'][step].view(1, -1),
                self.weights['decoder_time_mlp_in_w'],
                self.weights['decoder_time_mlp_in_b'],
                self.buffers['decoder_x_buf']
            )
            matmul_1_1024_1024_bias_silu(
                self.buffers['decoder_x_buf'],
                self.weights['decoder_time_mlp_out_w'],
                self.weights['decoder_time_mlp_out_b'],
                self.buffers['decoder_time_emb'][step]
            )
            for i in range(18):
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_attn_norm_mod_w'][i],
                    self.buffers['decoder_style_attn'][step, i],
                    self.weights['decoder_pre_attn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_ffn_norm_mod_w'][i],
                    self.buffers['decoder_style_ffn'][step, i],
                    self.weights['decoder_pre_ffn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
            matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                self.buffers['decoder_time_emb'][step],
                self.weights['decoder_final_norm_mod_w'],
                self.buffers['decoder_style_final'][step],
                self.weights['decoder_final_norm_mod_b'],
                seq_len = decoder_seq_len,
                features = 1024, 
                hidden = 3072,
                BLOCK_SIZE_N = 32,
                BLOCK_SIZE_M = 32,
                BLOCK_SIZE_K = 32
                )
            
        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w_t = torch.from_numpy(emb_w)
            else:
                emb_w_t = emb_w
            emb_w_t = emb_w_t.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        self.optimized = (
            optimized and torch.cuda.get_device_capability() == (12, 0)
            and num_views in (2, 3) and chunk_size == 50
            and not discrete_state_input and self.max_prompt_len == 0
        )
        self.buffers['optimized'] = self.optimized
        if self.optimized:
            self.weights['encoder_ffn_packed_w'] = torch.stack((
                self.weights['encoder_ffn_gate_w'], self.weights['encoder_ffn_up_w']
            ), dim=-1).reshape(18, 2048, 32768)
            self.weights['encoder_attn_kv_w'] = self.weights['encoder_attn_qkv_w'][-1, :, 2048:].contiguous()
            for key in ('decoder_attn_o', 'decoder_ffn_down'):
                self.weights[key + '_packed_w'] = self.weights[key + '_w'].transpose(1, 2).contiguous()
            self.buffers['vision_qkv_tile'] = (128, 64, 64, 4, 3)
            self.encoder_stream = torch.cuda.Stream()
            self.buffers['encoder_events'] = [torch.cuda.Event() for _ in range(18)]

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        with pdl_scope(self.optimized):
            if self.optimized:
                main = torch.cuda.current_stream()
                self.encoder_stream.wait_stream(main)
                with torch.cuda.stream(self.encoder_stream):
                    vision_encoder(self.weights, self.buffers, self.num_views)
                    transformer_encoder(self.weights, self.buffers, self.encoder_seq_len)
                transformer_decoder(self.weights, self.buffers, self.encoder_seq_len)
                main.wait_stream(self.encoder_stream)
            else:
                pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def record_infer_graph(self):
        if not self.optimized:
            for _ in range(3):
                self.record_run()
            main = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                self.infer_graph.capture_begin()
                self.record_run()
                self.infer_graph.capture_end()
            main.wait_stream(stream)
            return
        main = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.record_run()
        stream.synchronize()
        with torch.cuda.graph(self.infer_graph, stream=stream):
            self.record_run()
        main.wait_stream(stream)

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
    ) -> torch.Tensor:
        if (self.optimized and observation_images_normalized.is_cuda and diffusion_noise.is_cuda
            and observation_images_normalized.device == self.buffers['observation_images_normalized'].device
            and diffusion_noise.device == self.buffers['diffusion_noise'].device
            and observation_images_normalized.is_contiguous() and diffusion_noise.is_contiguous()
            and observation_images_normalized.shape == self.buffers['observation_images_normalized'].shape
            and diffusion_noise.shape == self.buffers['diffusion_noise'].shape
            and observation_images_normalized.dtype in (torch.bfloat16, torch.float32)
            and diffusion_noise.dtype in (torch.bfloat16, torch.float32)):
            image_size = self.buffers['observation_images_normalized'].numel()
            noise_size = self.buffers['diffusion_noise'].numel()
            rope_size = self.buffers['decoder_rope_weights'].numel()
            prepare_inputs[(triton.cdiv(image_size, 1024) + triton.cdiv(noise_size, 1024) + triton.cdiv(rope_size, 1024),)](
                observation_images_normalized, diffusion_noise, self._rope_table,
                self.buffers['observation_images_normalized'], self.buffers['diffusion_noise'],
                self.buffers['decoder_rope_weights'], self.buffers['valid_encoder_len'],
                image_size, noise_size, rope_size, self.num_views * 256, 1024, num_warps=4
            )
            self.infer_graph.replay()
            return self.buffers['diffusion_noise']
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']
