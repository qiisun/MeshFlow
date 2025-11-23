import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    FLASH_ATTN_AVAILABLE = True
except:
    print('[WARN] flash_attn not available, using naive implementation')
    FLASH_ATTN_AVAILABLE = False


# flashattn 2.7.0 changes unpad_input API... we are overriding it here
def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def attention_flashattn(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1), backend='flash-attn'):
    # q: (B, N, H, D)
    # k: (B, M, H, D)
    # v: (B, M, H, D)
    # mask_q: (B, N)
    # mask_kv: (B, M)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = k.shape[1]

    # kiui.lo(q, k, v)

    if causal: 
        assert N == 1 or N == M, 'Causal mask only supports self-attention'

    ### unmasked case (usually inference)
    ### will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask_q is None and mask_kv is None:
        if backend == 'flash-attn' and FLASH_ATTN_AVAILABLE:
            return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size) # [B, N, H, D]
        else: # naive implementation
            q = q.transpose(1, 2).reshape(B * H, N, D)
            k = k.transpose(1, 2).reshape(B * H, M, D)
            v = v.transpose(1, 2).reshape(B * H, M, D)
            w = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5) # [B*H, N, M]
            if causal and N > 1:
                causal_mask = torch.full((N, M), float('-inf'), device=w.device, dtype=w.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                w = w + causal_mask.unsqueeze(0)
            w = F.softmax(w, dim=-1)
            if dropout > 0:
                w = F.dropout(w, p=dropout)
            out = torch.bmm(w, v) # [B*H, N, D]
            out = out.reshape(B, H, N, D).transpose(1, 2).contiguous() # [B, N, H, D]
            return out
    
    ### at least one of q or kv is masked (training)
    ### only support flash-attn for now...
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    elif mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)

    if FLASH_ATTN_AVAILABLE:
        # unpad (gather) input
        # mask_q: [B, N], first row has N1 1s, second row has N2 1s, ...
        # indices: [Ns,], Ns = N1 + N2 + ...
        # cu_seqlens_q: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
        # max_len_q: scalar, max(N1, N2, ...)
        q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
        k, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(k, mask_kv)
        v = index_first_axis(v.reshape(-1, H, D), indices_kv) # same indice as k

        # call varlen_func
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=dropout,
            causal=causal,
            window_size=window_size,
        )

        # pad (put back) output
        out = pad_input(out, indices_q, B, N)
        return out
    else:
        raise NotImplementedError('masked attention requires flash_attn!')


def attention_pytorch_sdpa(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False):
    """
    使用 PyTorch 的 scaled_dot_product_attention 实现的高效注意力机制。

    参数:
    q (torch.Tensor): 查询张量，形状为 (B, N, H, D)。
                      B=批次大小, N=目标序列长度, H=头数, D=每个头的维度。
    k (torch.Tensor): 键张量，形状为 (B, M, H, D)。M=源序列长度。
    v (torch.Tensor): 值张量，形状为 (B, M, H, D)。
    mask_q (torch.Tensor, optional): 查询的布尔掩码，形状为 (B, N)。
                                     True 表示有效token（不被遮蔽），False 表示padding（被遮蔽）。默认为 None。
    mask_kv (torch.Tensor, optional): 键/值的布尔掩码，形状为 (B, M)。
                                      True 表示有效token（不被遮蔽），False 表示padding（被遮蔽）。默认为 None。
    dropout (float, optional): Attention权重的dropout概率。仅在模型训练时且dropout > 0时生效。默认为 0.0。
    causal (bool, optional): 是否启用因果遮罩（例如 Decoder 中的自注意力）。默认为 False。
                               如果为True，通常要求 N == M (自注意力)。对于 N=1（单步解码）的情况，
                               is_causal=True 在 scaled_dot_product_attention 中通常没有额外遮罩效应。

    返回:
    torch.Tensor: 输出张量，形状为 (B, N, H, D)。

    注意:
    - 此实现利用了 PyTorch 2.0+ 中引入的 scaled_dot_product_attention。
    - 为了正确组合 attn_mask (用于padding) 和 is_causal=True (用于因果遮罩), 推荐使用 PyTorch 2.1 或更高版本。
      在旧版本中，如果同时提供了 attn_mask，is_causal 参数可能会被忽略。
    """
    B, N_target, H, D_head = q.shape
    _B_k, M_source, _H_k, _D_k = k.shape
    _B_v, _M_v, _H_v, _D_v = v.shape

    # 基本维度校验
    if not (H == _H_k == _H_v and D_head == _D_k == _D_v):
        raise ValueError(f"查询、键、值的头数(H)和头部维度(D)必须一致。收到的 H: q={H}, k={_H_k}, v={_H_v}; D: q={D_head}, k={_D_k}, v={_D_v}")
    if not (B == _B_k == _B_v):
        raise ValueError(f"查询、键、值的批次大小(B)必须一致。收到的 B: q={B}, k={_B_k}, v={_B_v}")
    if not (M_source == _M_v):
        raise ValueError(f"键和值的源序列长度(M)必须一致。收到的 M: k={M_source}, v={_M_v}")

    # torch.nn.functional.scaled_dot_product_attention 需要 (B, H, Seq, Dim_head) 格式的输入
    # 而我们函数接收的格式是 (B, Seq, H, Dim_head)，所以需要进行维度转换
    q_sdpa = q.transpose(1, 2)  # 转换为 (B, H, N_target, D_head)
    k_sdpa = k.transpose(1, 2)  # 转换为 (B, H, M_source, D_head)
    v_sdpa = v.transpose(1, 2)  # 转换为 (B, H, M_source, D_head)

    # 构建 attn_mask 以处理 padding
    # 在 scaled_dot_product_attention 中, attn_mask 的 True 值表示对应位置“不应该被注意到”（即被遮蔽）
    # padding_attn_mask 需要能够广播到形状 (B, H, N_target, M_source)
    # 我们将构建一个 (B, 1, N_target, M_source) 的掩码，它会自动广播到 H 个头
    padding_attn_mask = None
    if mask_q is not None or mask_kv is not None:
        # mask_q (B, N_target): True 表示有效, False 表示 padding
        # mask_kv (B, M_source): True 表示有效, False 表示 padding
        # 我们需要的是: True 表示 padding (应被遮蔽)

        _mask_components = [] # 用于收集需要 OR 操作的掩码部分

        if mask_q is not None:
            # (~mask_q) 结果中 True 代表 padding 的位置, 形状 (B, N_target)
            # 需要扩展成 (B, N_target, M_source)，如果 q 的某个 token 是 padding，则它与所有 k 的 token 的注意力都应被遮蔽
            mask_q_for_sdpa = (~mask_q).unsqueeze(2).expand(B, N_target, M_source)
            _mask_components.append(mask_q_for_sdpa)
        
        if mask_kv is not None:
            # (~mask_kv) 结果中 True 代表 padding 的位置, 形状 (B, M_source)
            # 需要扩展成 (B, N_target, M_source)，如果 k 的某个 token 是 padding，则所有 q 的 token 与它的注意力都应被遮蔽
            mask_kv_for_sdpa = (~mask_kv).unsqueeze(1).expand(B, N_target, M_source)
            _mask_components.append(mask_kv_for_sdpa)

        if _mask_components:
            # 使用逻辑或 (OR) 合并所有padding掩码
            # 例如，如果 q_i 是padding 或 k_j 是padding，则 (q_i, k_j) 应被遮蔽
            combined_mask = _mask_components[0]
            for i in range(1, len(_mask_components)):
                combined_mask = combined_mask | _mask_components[i]
            
            padding_attn_mask = combined_mask.unsqueeze(1) # 扩展为 (B, 1, N_target, M_source)

    # 处理因果遮罩 (causal)
    # scaled_dot_product_attention 的 is_causal 参数主要用于自注意力 (N_target == M_source)
    sdpa_is_causal = False
    if causal:
        if N_target != M_source:
            # 如果是目标序列长度 N_target = 1 (例如，解码过程中的第一步或当前步)
            # is_causal=True 对于 scaled_dot_product_attention 来说是安全的，
            # 因为对于单个查询token，没有“未来”的token需要从当前查询token内部遮蔽。
            if N_target == 1:
                sdpa_is_causal = True 
            else:
                # 当 N_target > 1 且 N_target != M_source 时，标准的因果遮罩定义不适用，
                # scaled_dot_product_attention 的 is_causal=True 也会报错。
                raise ValueError(
                    f"当 causal=True 且目标序列长度 N_target ({N_target}) > 1 时, "
                    f"N_target ({N_target}) 必须等于源序列长度 M_source ({M_source}) "
                    "才能在 scaled_dot_product_attention 中使用 is_causal=True。"
                )
        else: # N_target == M_source (典型的自注意力因果遮罩场景)
            sdpa_is_causal = True
            
    # 调用 PyTorch 原生的 scaled_dot_product_attention 实现
    # 注意: PyTorch 2.1+ 版本可以很好地处理 attn_mask (来自 padding) 和 is_causal=True 同时存在的情况，
    # 会将两者进行合并（OR 逻辑），这是我们期望的行为。
    output = F.scaled_dot_product_attention(
        query=q_sdpa,
        key=k_sdpa,
        value=v_sdpa,
        attn_mask=padding_attn_mask,  # padding 掩码，或为 None
        dropout_p=dropout,           # Dropout 概率 (只在训练模式且 dropout > 0 时生效)
        is_causal=sdpa_is_causal     # 是否应用因果（上三角）遮罩
    )

    # 将输出张量的维度转换回原始函数期望的 (B, N_target, H, D_head) 格式
    output = output.transpose(1, 2)

    return output

def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False, backend='pytorch-sdpa'):
    if backend == 'flash-attn':
        return attention_flashattn(q, k, v, mask_q=mask_q, mask_kv=mask_kv,
                                   dropout=dropout, causal=causal)
    elif backend == 'pytorch-sdpa':
        return attention_pytorch_sdpa(q, k, v, mask_q=mask_q, mask_kv=mask_kv,
                                      dropout=dropout, causal=causal)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, output_dim=None, dropout=0, causal=False, mixed_precision='bf16'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.mixed_precision = mixed_precision
    
    def forward(self, x, mask=None):
        # x: [B, N, C]
        # mask: [B, N]
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.chunk(3, dim=0) # [3, B, N, H, D] -> 3 * [1, B, N, H, D]
        x = attention(q[0], k[0], v[0], mask_q=mask, mask_kv=mask, dropout=self.dropout, causal=self.causal, backend='pytorch-sdpa' if self.mixed_precision=='fp32' else 'flash-attn') # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, context_dim=None, output_dim=None, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, context, mask_q=None, mask_kv=None):
        # x: [B, N, C]
        # context: [B, M, C']
        # mask_q: [B, N]
        # mask_kv: [B, M]
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        x = attention(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=self.dropout, causal=False) # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x