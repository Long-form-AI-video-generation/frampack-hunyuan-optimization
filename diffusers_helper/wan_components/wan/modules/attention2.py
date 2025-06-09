# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]


def pytorch_attention_fallback(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1),
    deterministic=False, dtype=torch.bfloat16,
):
    """
    PyTorch fallback implementation for flash attention
    
    Args:
        q: [B, Lq, Nq, C1] Query tensor
        k: [B, Lk, Nk, C1] Key tensor  
        v: [B, Lk, Nk, C2] Value tensor
        q_lens: [B] Sequence lengths for queries (optional)
        k_lens: [B] Sequence lengths for keys (optional)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        q_scale: Additional scaling for queries
        causal: Whether to apply causal masking
        window_size: Local attention window (not implemented in fallback)
        deterministic: Whether to use deterministic operations
        dtype: Target dtype for computation
    
    Returns:
        Tensor: Attention output with same shape as input q
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    
    # Get original shapes and types
    B, Lq, Nq, Cq = q.shape
    _, Lk, Nk, Cv = k.shape[0], k.shape[1], k.shape[2], v.shape[3]
    out_dtype = q.dtype
    
    # Convert to appropriate dtype for computation
    compute_dtype = dtype if q.dtype not in half_dtypes else q.dtype
    q = q.to(compute_dtype)
    k = k.to(compute_dtype) 
    v = v.to(compute_dtype)
    
    # Apply query scaling if provided
    if q_scale is not None:
        q = q * q_scale
        
    # Handle variable length sequences by creating attention masks
    attn_mask = None
    if q_lens is not None or k_lens is not None:
        # Create attention mask for variable length sequences
        if q_lens is None:
            q_lens = torch.full((B,), Lq, dtype=torch.long, device=q.device)
        if k_lens is None:
            k_lens = torch.full((B,), Lk, dtype=torch.long, device=k.device)
            
        # Create mask: [B, Lq, Lk]
        attn_mask = torch.zeros(B, Lq, Lk, dtype=torch.bool, device=q.device)
        for i in range(B):
            attn_mask[i, :q_lens[i], :k_lens[i]] = True
        
        # Convert to additive mask (0 for valid, -inf for masked)
        attn_mask = attn_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        # attn_mask = attn_mask.to(compute_dtype)
        attn_mask = attn_mask.to(q.device).to(compute_dtype)
    
    # Handle multi-head attention
    # If Nq != Nk, we need to handle grouped attention
    if Nq != Nk:
        assert Nq % Nk == 0, f"Number of query heads ({Nq}) must be divisible by key heads ({Nk})"
        # Repeat k and v to match query heads
        repeat_factor = Nq // Nk
        k = k.repeat_interleave(repeat_factor, dim=2)  # [B, Lk, Nq, C1]
        v = v.repeat_interleave(repeat_factor, dim=2)  # [B, Lk, Nq, C2]
    
    # Reshape for PyTorch attention: [B, N, L, C]
    q = q.transpose(1, 2)  # [B, Nq, Lq, C1]
    k = k.transpose(1, 2)  # [B, Nq, Lk, C1] 
    v = v.transpose(1, 2)  # [B, Nq, Lk, C2]
    
    # Expand attention mask for multi-head: [B, Nq, Lq, Lk]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1).expand(B, Nq, Lq, Lk)
    
    # Apply scaled dot product attention
    try:
        # Use PyTorch's optimized attention if available
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if not deterministic else 0.0,
            scale=softmax_scale,
            is_causal=causal and attn_mask is None  # Don't use causal if we have custom mask
        )
    except Exception as e:
        warnings.warn(f"scaled_dot_product_attention failed ({e}), using manual implementation")
        # Manual implementation as fallback
        
        # Compute attention scores
        scale = softmax_scale or (Cq ** -0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, Nq, Lq, Lk]
        
        # Apply masks
        if attn_mask is not None:
            scores = scores + attn_mask
            
        if causal:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(Lq, Lk, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0 and not deterministic:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, Nq, Lq, C2]
    
    # Reshape back to original format: [B, Lq, Nq, C2]
    output = output.transpose(1, 2)
    
    # Convert back to original dtype
    output = output.to(out_dtype)
    
    return output


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Flash attention with automatic fallback to PyTorch implementation
    
    Args:
        q: [B, Lq, Nq, C1] Query tensor
        k: [B, Lk, Nk, C1] Key tensor
        v: [B, Lk, Nk, C2] Value tensor
        q_lens: [B] Optional sequence lengths for queries
        k_lens: [B] Optional sequence lengths for keys
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for attention scores
        q_scale: Additional scaling for queries
        causal: Whether to apply causal masking
        window_size: Local attention window (left, right)
        deterministic: Whether to use deterministic operations
        dtype: Target dtype for computation
        version: Flash attention version to use (2 or 3)
    
    Returns:
        Tensor: Attention output with same shape as input q
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    
    # Check if we're on CUDA (Flash Attention requirement)
    if q.device.type != 'cuda':
        warnings.warn("Flash Attention requires CUDA, using PyTorch fallback")
        return pytorch_attention_fallback(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale, 
            q_scale, causal, window_size, deterministic, dtype
        )
    
    # Check head dimension constraint
    if q.size(-1) > 256:
        warnings.warn(f"Head dimension {q.size(-1)} > 256, using PyTorch fallback")
        return pytorch_attention_fallback(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
            q_scale, causal, window_size, deterministic, dtype
        )

    # Try Flash Attention implementations
    if FLASH_ATTN_3_AVAILABLE and (version is None or version == 3):
        try:
            return _flash_attention_3(
                q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                q_scale, causal, window_size, deterministic, dtype
            )
        except Exception as e:
            warnings.warn(f"Flash Attention 3 failed ({e}), trying Flash Attention 2")
    
    if FLASH_ATTN_2_AVAILABLE:
        try:
            return _flash_attention_2(
                q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                q_scale, causal, window_size, deterministic, dtype
            )
        except Exception as e:
            warnings.warn(f"Flash Attention 2 failed ({e}), using PyTorch fallback")
    
    # Final fallback to PyTorch
    warnings.warn("Flash Attention not available, using PyTorch scaled_dot_product_attention")
    return pytorch_attention_fallback(
        q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
        q_scale, causal, window_size, deterministic, dtype
    )


def _flash_attention_3(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, 
                      q_scale, causal, window_size, deterministic, dtype):
    """Flash Attention 3 implementation"""
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)

    # Preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # Apply Flash Attention 3
    x = flash_attn_interface.flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32),
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic
    )[0].unflatten(0, (b, lq))

    return x.type(out_dtype)


def _flash_attention_2(q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                      q_scale, causal, window_size, deterministic, dtype):
    """Flash Attention 2 implementation"""
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)

    # Preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # Apply Flash Attention 2
    x = flash_attn.flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic
    ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1),
    deterministic=False, dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Main attention function with automatic backend selection
    """
    return flash_attention(
        q=q, k=k, v=v,
        q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
        causal=causal, window_size=window_size,
        deterministic=deterministic, dtype=dtype,
        version=fa_version,
    )