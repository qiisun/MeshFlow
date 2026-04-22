import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.utils import RMSNorm


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input

    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False
    print("[WARNING] FlashAttention not available — falling back to PyTorch SDPA. "
          "Install flash-attn for faster training.")


def _unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _attention_flash(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False, window_size=(-1, -1)):
    batch_size, n_tokens, num_heads, head_dim = q.shape
    m_tokens = k.shape[1]

    if causal:
        assert n_tokens == 1 or n_tokens == m_tokens, 'Causal mask only supports self-attention'

    if mask_q is None and mask_kv is None:
        return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size)

    if mask_q is None:
        mask_q = torch.ones(batch_size, n_tokens, dtype=torch.bool, device=q.device)
    if mask_kv is None:
        mask_kv = torch.ones(batch_size, m_tokens, dtype=torch.bool, device=q.device)

    q, indices_q, cu_seqlens_q, max_len_q = _unpad_input(q, mask_q)
    k, indices_kv, cu_seqlens_kv, max_len_kv = _unpad_input(k, mask_kv)
    v = index_first_axis(v.reshape(-1, num_heads, head_dim), indices_kv)

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_len_q,
        max_seqlen_k=max_len_kv,
        dropout_p=dropout,
        causal=causal,
        window_size=window_size,
    )
    return pad_input(out, indices_q, batch_size, n_tokens)


def _attention_sdpa(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False):
    batch_size, n_target, num_heads, head_dim = q.shape
    _, m_source, _, _ = k.shape

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    padding_attn_mask = None
    if mask_q is not None or mask_kv is not None:
        mask_components = []
        if mask_q is not None:
            mask_components.append((~mask_q).unsqueeze(2).expand(batch_size, n_target, m_source))
        if mask_kv is not None:
            mask_components.append((~mask_kv).unsqueeze(1).expand(batch_size, n_target, m_source))

        if mask_components:
            combined_mask = mask_components[0]
            for item in mask_components[1:]:
                combined_mask = combined_mask | item
            padding_attn_mask = combined_mask.unsqueeze(1)

    sdpa_is_causal = False
    if causal:
        if n_target == m_source or n_target == 1:
            sdpa_is_causal = True
        else:
            raise ValueError(
                f"causal=True requires n_target == m_source or n_target == 1, got n_target={n_target}, m_source={m_source}"
            )

    output = F.scaled_dot_product_attention(
        query=q_sdpa,
        key=k_sdpa,
        value=v_sdpa,
        attn_mask=padding_attn_mask,
        dropout_p=dropout,
        is_causal=sdpa_is_causal,
    )
    return output.transpose(1, 2)


def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False, backend='pytorch-sdpa'):
    if backend == 'flash-attn':
        if not FLASH_ATTN_AVAILABLE:
            return _attention_sdpa(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=dropout, causal=causal)
        return _attention_flash(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=dropout, causal=causal)

    if backend == 'pytorch-sdpa':
        return _attention_sdpa(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=dropout, causal=causal)

    raise ValueError(f"Unsupported backend: {backend}")


class PrecisionSafeLayerNorm(nn.Module):
    """Run RMSNorm in fp32 for stability, then cast back to input dtype."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return self.norm(x.float()).to(dtype=x.dtype)
        return self.norm(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        input_dim=None,
        output_dim=None,
        dropout=0.0,
        causal=False,
        mixed_precision='bf16',
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.mixed_precision = mixed_precision

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.q_norm = PrecisionSafeLayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = PrecisionSafeLayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, mask=None):
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q, k = self.q_norm(q[0]), self.k_norm(k[0])

        backend = 'pytorch-sdpa' if self.mixed_precision == 'fp32' else 'flash-attn'
        if q.dtype not in (torch.float16, torch.bfloat16):
            backend = 'pytorch-sdpa'
        x = attention(
            q,
            k,
            v[0],
            mask_q=mask,
            mask_kv=mask,
            dropout=self.dropout,
            causal=self.causal,
            backend=backend,
        )
        return self.out_proj(x.reshape(batch_size, n_tokens, -1))
