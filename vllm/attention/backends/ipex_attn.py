# SPDX-License-Identifier: Apache-2.0
""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import os
from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)

from vllm.logger import init_logger
logger = init_logger('vllm.attention.backends.ipex_attn')

_PARTITION_SIZE = 512
_IPEX_BACKEND_SUPPORTED_KV_CACHE_FORMAT=["fp8", "auto"]


class IpexAttnBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "IPEX"

    @staticmethod
    def get_impl_cls() -> Type["IpexAttnBackendImpl"]:
        return IpexAttnBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["IpexAttnMetadata"]:
        return IpexAttnMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        torch.xpu.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        torch.xpu.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class IpexAttnMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for IpexAttnBackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor

    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    seq_lens: Optional[List[int]]
    seqlen_q: Optional[torch.Tensor]
    max_seqlen: Optional[int]
    query_start_loc: Optional[torch.Tensor]
    context_lens: Optional[torch.Tensor]

    _cached_prefill_metadata: Optional["IpexAttnMetadata"] = None
    _cached_decode_metadata: Optional["IpexAttnMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None
    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None


    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))


    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))


    def get_attn_bias(
        self,
        attn_type: str,
    ) -> Optional[List[torch.Tensor]]:
        '''
        Extract appropriate attention bias from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate attention bias value given the attention type
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            return self.attn_bias
        elif attn_type == AttentionType.ENCODER:
            return self.encoder_attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            return self.cross_attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")


    def set_attn_bias(
        self,
        attn_bias: List[torch.Tensor],
        attn_type: str,
    ) -> None:
        '''
        Update appropriate attention bias field of attention metadata,
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_bias: The desired attention bias value
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            self.attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER:
            self.encoder_attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            self.cross_attn_bias = attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None
        self.encoder_attn_bias: Optional[List[torch.Tensor]] = None
        self.cross_attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def prefill_metadata(self) -> Optional["IpexAttnMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens is not None
        assert self.block_tables is not None

        self._cached_prefill_metadata = IpexAttnMetadata(
            is_prompt=self.is_prompt,
            seqlen_q=self.seqlen_q,
            max_seqlen=self.max_seqlen,
            num_prefills=self.num_prefills,
            multi_modal_placeholder_index_maps=None,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            # max_query_len=self.max_query_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1] if (torch.is_tensor(self.query_start_loc)) else None,
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1] if (torch.is_tensor(self.seq_start_loc)) else None,
            context_lens=self.context_lens[:self.num_prefills] if (torch.is_tensor(self.context_lens)) else None,
            block_tables=self.block_tables[:self.num_prefills],
            enable_kv_scales_calculation=False,
            # Begin encoder & cross attn fields below...
            max_prefill_seq_len=self.max_prefill_seq_len,
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["IpexAttnMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = IpexAttnMetadata(
            is_prompt=self.is_prompt,
            seqlen_q=self.seqlen_q,
            max_seqlen=self.max_seqlen,
            num_prefills=0,
            multi_modal_placeholder_index_maps=None,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=self.seq_lens[self.num_prefills:],
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            # max_query_len=None,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            # seq_start_loc=None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:],
            context_lens=self.context_lens[self.num_prefills:] if (torch.is_tensor(self.context_lens)) else None,
            block_tables=self.block_tables[self.num_prefills:],
            enable_kv_scales_calculation=False,
            # Begin encoder & cross attn fields below...
            max_prefill_seq_len=self.max_prefill_seq_len,
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables
        )
        return self._cached_decode_metadata
    

    def get_seq_lens(
        self,
        attn_type: str,
    ):
        '''
        Extract appropriate sequence lengths from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate sequence lengths tensor for query
        * Appropriate sequence lengths tensor for key & value
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.seq_lens
        elif attn_type == AttentionType.ENCODER:
            seq_lens_q = self.encoder_seq_lens
            seq_lens_kv = self.encoder_seq_lens
        elif attn_type == AttentionType.ENCODER_DECODER:
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.encoder_seq_lens
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        return seq_lens_q, seq_lens_kv
    

    def get_seq_len_block_table_args(
        self,
        attn_type: str,
    ) -> tuple:
        '''
        The particular choice of sequence-length- and block-table-related
        attributes which should be extracted from attn_metadata is dependent
        on the type of attention operation.

        Decoder attn -> select entirely decoder self-attention-related fields
        Encoder/decoder cross-attn -> select encoder sequence lengths &
                                    cross-attn block-tables fields
        Encoder attn -> select encoder sequence lengths fields & no block tables

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * is_prompt: True if prefill, False otherwise
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:

        * Appropriate sequence-lengths tensor
        * Appropriate max sequence-length scalar
        * Appropriate block tables (or None)
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            # Decoder self-attention
            # Choose max_seq_len based on whether we are in prompt_run
            return (self.seq_lens_tensor, self.max_decode_seq_len,
                    self.block_tables)
        elif attn_type == AttentionType.ENCODER_DECODER:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    self.cross_block_tables)
        elif attn_type == AttentionType.ENCODER:
            # No block tables associated with encoder attention
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    None)
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")


    def advance_step(self, num_seqs, num_queries):
        assert num_seqs == num_queries

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        # assert self.max_query_len == 1
        # assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        # assert self.query_start_loc is not None
        # assert self.query_start_loc.shape == (num_queries + 1, )
        # assert self.seq_start_loc is not None
        # assert self.seq_start_loc.shape == (num_seqs + 1, )

        # assert self.context_lens_tensor is not None
        # assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)


from torch.nn.functional import scaled_dot_product_attention

def _make_attention_mask(
    att_bias: List[torch.Tensor],
    seq_lens: List[int],
    prompt_token_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert att_bias[0].dim() == 3
    assert len(att_bias) == len(seq_lens)
    head_size, _, _ = att_bias[0].size()
    mask = torch.empty(head_size,
                       prompt_token_num,
                       prompt_token_num,
                       dtype=dtype)
    mask.fill_(-torch.inf)
    start = 0
    for prompt_len, sub_mask in zip(seq_lens, att_bias):
        end = start + prompt_len
        mask[:, start:end, start:end] = sub_mask
        start += prompt_len
    return mask


def use_sdp_causal(head_dim, query_states, logits_soft_cap, attn_type):
    disabled = os.environ.get('IPEX_LLM_DISABLE_SDP_CAUSAL', None)
    if disabled is not None:
        disabled = int(disabled)
        if disabled == 1:
            return False
    return (
        (logits_soft_cap != 0                        # for gemma model
        or head_dim in [-1, 64, 80, 96, 128, 256])        # for now
        and query_states.device.type == "xpu"        # GPU
        and query_states.dtype in [torch.float, torch.half]     # fp32/fp16
        and attn_type is AttentionType.DECODER
    )

def use_gqa_kernel(num_heads, num_kv_heads, head_size, logits_soft_cap):
    kv_cache_format = os.environ.get('USE_VLLM_KVCACHE')
    if kv_cache_format is None and num_heads != num_kv_heads and head_size in [128, 96, 80, 64] and logits_soft_cap == 0:
        return True
    else:
        return False

class IpexAttnBackendImpl(AttentionImpl[IpexAttnMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "IPEX backend does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.use_irope = use_irope

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        # if attn_type != AttentionType.DECODER and attn_type != AttentionType.ENCODER_ONLY:
        #     raise NotImplementedError("Encoder/decoder cross-attention "
        #                               "is not implemented for "
        #                               "IpexAttnBackendImpl")
        if kv_cache_dtype not in _IPEX_BACKEND_SUPPORTED_KV_CACHE_FORMAT:
            raise NotImplementedError(f"IPEX backend does not support "
                                       "KV cache format {kv_cache_dtype}")
        # Also check for gqa models...
        self.using_gqa_kernel = use_gqa_kernel(self.num_heads, self.num_kv_heads, self.head_size, self.logits_soft_cap)
        if not self.using_gqa_kernel and kv_cache_dtype == "fp8":
            raise NotImplementedError(f"IPEX backend currently only supports "
                                      "fp8 kv cache in group-query attention")

        self.ipex_varlen_attn = False
        flag = os.getenv("IPEX_LLM_PREFILL_VARLEN_BACKEND", None)
        if flag is not None:
            self.ipex_varlen_attn = True
            logger.info_once(f"Using varlen_attention for prefilling.")

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)

        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache


    def split_kv_cache_ipexllm(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # For GQA kernel, key_cache and value_cache shape should be [num_blocks, num_kv_heads, head_size, block_size]
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
        return key_cache, value_cache


    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IpexAttnMetadata,  # type: ignore
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEX varlen_attention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0 and attn_type != AttentionType.ENCODER:
            if self.using_gqa_kernel:
                key_cache, value_cache = self.split_kv_cache_ipexllm(
                    kv_cache, self.num_kv_heads, self.head_size)
            else:
                key_cache, value_cache = self.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size)
            if (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping
                if self.using_gqa_kernel:
                    ipex_ops.reshape_and_cache_ipexllm(
                        key,
                        value,
                        key_cache,
                        value_cache,
                        updated_slot_mapping.flatten(),
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )
                else:
                    ipex_ops.reshape_and_cache(
                        key,
                        value,
                        key_cache,
                        value_cache,
                        updated_slot_mapping.flatten(),
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )

        if attn_type != AttentionType.ENCODER:
            # Decoder self-attention supports chunked prefill.
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
        else:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0

        if attn_type == AttentionType.DECODER:
            # Only enforce this shape-constraint for decoder
            # self-attention
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        decode_query = query[num_prefill_tokens:]

        is_causal = not self.need_mask
        if attn_type == AttentionType.ENCODER_ONLY:
            is_causal = False

        if prefill_meta := attn_metadata.prefill_metadata:
            assert prefill_meta.seq_lens is not None
            if (kv_cache is None or prefill_meta.block_tables.numel() == 0):


                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)

                attn_masks = attn_metadata.get_attn_bias(attn_type)
                if attn_masks is None:
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(
                            self.alibi_slopes, query.dtype,
                            attn_metadata.seq_lens)  # type: ignore
                    elif self.sliding_window is not None:
                        assert attn_metadata.seq_lens is not None
                        attn_masks = _make_sliding_window_bias(
                            attn_metadata.seq_lens, self.sliding_window,
                            query.dtype)  # type: ignore
                    else:
                        seq_lens, _ = attn_metadata.get_seq_lens(attn_type)
                        attn_masks = [None] * len(seq_lens)
                    attn_metadata.set_attn_bias(attn_masks, attn_type)

                if self.ipex_varlen_attn:
                    output = torch.empty(
                        (num_tokens, self.num_heads, self.head_size),
                        dtype=query.dtype,
                        device=query.device)

                    tmp = [0]
                    tmp.extend(prefill_meta.seq_lens)
                    seqlen = torch.tensor(tmp)
                    seqlen_q = torch.cumsum(seqlen, dim=0).to(device=query.device)
                    ipex_ops.varlen_attention(query,
                                              key,
                                              value,
                                              output,
                                              seqlen_q,
                                              seqlen_q,
                                              prefill_meta.max_seqlen,
                                              prefill_meta.max_seqlen,
                                              pdropout=0.0,
                                              softmax_scale=self.scale,
                                              zero_tensors=False,
                                              is_causal=is_causal,
                                              return_softmax=False,
                                              gen_=None,
                                              logits_soft_cap=self.logits_soft_cap)
                else:
                    # output = torch.empty(
                    #             (num_tokens, self.num_heads, self.head_size),
                    #             dtype=query.dtype, device=query.device)
                    query = query.movedim(0, query.dim() - 2)
                    key = key.movedim(0, key.dim() - 2)
                    value = value.movedim(0, value.dim() - 2)
                    import math
                    scale = 1 / math.sqrt(self.head_size) if self.scale is None else self.scale
                    causal_attn = (attn_type == AttentionType.DECODER)
                    seq_lens_q, seq_lens_kv = attn_metadata.get_seq_lens(attn_type)
                    start_q, start_kv = 0, 0

                    for seq_len_q, seq_len_kv, mask in zip(seq_lens_q, seq_lens_kv,
                                                        attn_masks):
                        end_q = start_q + seq_len_q
                        end_kv = start_kv + seq_len_kv
                        if self.alibi_slopes is None and use_sdp_causal(self.head_size, query, self.logits_soft_cap, attn_type):
                            import xe_addons
                            if mask is not None:
                                mask = mask.unsqueeze(0)
                            if self.logits_soft_cap == 0 or self.head_size != 256:
                                sub_out = xe_addons.sdp_causal(
                                    query[None, :, start_q:end_q, :].contiguous(),
                                    key[None, :, start_kv:end_kv, :].contiguous(),
                                    value[None, :, start_kv:end_kv, :].contiguous(),
                                    mask,
                                    scale).squeeze(0).movedim(
                                        query.dim() - 2, 0)
                            else:
                                sub_out = xe_addons.gemma2_sdp_causal(
                                    query[None, :, start_q:end_q, :].contiguous(),
                                    key[None, :, start_kv:end_kv, :].contiguous(),
                                    value[None, :, start_kv:end_kv, :].contiguous(),
                                    mask,
                                    self.logits_soft_cap,
                                    self.scale).squeeze(0).movedim(
                                        query.dim() - 2, 0)
                        else:
                            sub_out = torch.nn.functional.scaled_dot_product_attention(
                                query[None, :, start_q:end_q, :],
                                key[None, :, start_kv:end_kv, :],
                                value[None, :, start_kv:end_kv, :],
                                attn_mask=mask,
                                dropout_p=0.0,
                                is_causal=causal_attn and mask is None,
                                scale=self.scale).squeeze(0).movedim(
                                    query.dim() - 2, 0)
                        output[start_q:end_q, :, :] = sub_out
                        start_q, start_kv = end_q, end_kv

            else:
                # prefix-enabled attention
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)
                import vllm._C.ops
                assert self.head_size == 128 or self.head_size == 64
                value = os.environ.get('USE_CONTEXT_V1')
                if self.using_gqa_kernel:
                    # if using_gqa_kernel, then only the v1 kernel can be used
                    out = vllm._C.ops.context_attention_forward_v1(query, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.context_lens, prefill_meta.max_seqlen, torch.amax(prefill_meta.context_lens).item())
                elif value is None:
                    # Otherwise, by default use v2 attention forward kernel...
                    query_len = prefill_meta.query_start_loc[1:] - prefill_meta.query_start_loc[:-1]
                    out = vllm._C.ops.context_attention_forward_v2(query, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.context_lens, prefill_meta.max_seqlen, torch.amax(prefill_meta.context_lens).item(), torch.amax(query_len).item())
                else:
                    out = vllm._C.ops.context_attention_forward_v1(query, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.context_lens, prefill_meta.max_seqlen, torch.amax(prefill_meta.context_lens).item())
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            max_seq_len = decode_meta.max_decode_seq_len
            out = torch.empty_like(decode_query)
            num_seqs, num_heads, head_size = decode_query.shape
            max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                                  _PARTITION_SIZE)
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)
            # NOTE(woosuk): We use a simple heuristic to decide whether to use
            # PagedAttention V1 or V2. If the number of partitions is 1, we use
            # V1 to avoid the overhead of reduction. Also, if the number of
            # sequences or heads is large, we use V1 since there is enough work
            # to parallelize.
            # TODO(woosuk): Tune this heuristic.
            # For context len > 8192, use V2 kernel to avoid shared memory
            # shortage.

            bsz = len(decode_meta.seq_lens)
            import vllm._C.ops

            if self.using_gqa_kernel:
                block_size = value_cache.shape[2]
                ipex_ops.paged_attention_gqa(
                    out,
                    decode_query,
                    key_cache,
                    value_cache,
                    bsz,
                    self.num_heads,
                    self.num_kv_heads,
                    self.scale,
                    decode_meta.block_tables,
                    decode_meta.seq_lens_tensor,
                    block_size,
                    head_size,
                    max_seq_len,
                    self.kv_cache_dtype
                )
            else:
                block_size = value_cache.shape[3]
                use_v1 = (max_seq_len <= 8192 and
                        (max_num_partitions == 1 or num_seqs * num_heads > 512))
                if use_v1:
                    # Run PagedAttention V1.
                    ipex_ops.paged_attention_v1(
                        out,
                        decode_query,
                        key_cache,
                        value_cache,
                        self.num_kv_heads,
                        self.scale,
                        block_tables_arg,
                        seq_lens_arg,
                        block_size,
                        max_seq_len_arg,
                        self.alibi_slopes,
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                        self.logits_soft_cap,
                    )
                else:
                    # Run PagedAttention V2.
                    assert _PARTITION_SIZE % block_size == 0
                    tmp_output = torch.empty(
                        size=(num_seqs, num_heads, max_num_partitions, head_size),
                        dtype=output.dtype,
                        device=output.device,
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, num_heads, max_num_partitions),
                        dtype=torch.float32,
                        device=output.device,
                    )
                    max_logits = torch.empty_like(exp_sums)
                    ipex_ops.paged_attention_v2(
                        out,
                        exp_sums,
                        max_logits,
                        tmp_output,
                        decode_query,
                        key_cache,
                        value_cache,
                        self.num_kv_heads,
                        self.scale,
                        block_tables_arg,
                        seq_lens_arg,
                        block_size,
                        max_seq_len_arg,
                        self.alibi_slopes,
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                        self.logits_soft_cap,
                    )
            output[num_prefill_tokens:] = out

            # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None])
        inf_mask = torch.empty(
            (1, seq_len, seq_len),
            dtype=bias.dtype,
            device=alibi_slopes.device).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases = []
    for seq_len in seq_lens:
        tensor = torch.full(
            (1, seq_len, seq_len),
            dtype=dtype,
            fill_value=1,
        )
        shift = 0
        mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
        if window_size is not None:
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
        mask = torch.log(mask)
        attn_biases.append(mask.to(dtype))

    return attn_biases

def get_num_prefill_decode_query_kv_tokens(
    attn_metadata,
    attn_type: str,
) -> Tuple[int, int, int]:
    """
    Calculate the number of prefill and decode tokens for query, key/value
    based on the attention metadata and the specified attention type.

    Args:
        attn_metadata (FlashAttentionMetadata): Attention Metadata object.
        attn_type (AttentionType): The type of attention being used.
    Returns:
        Tuple[int, int, int]: A tuple containing three integers:
            - The number of prefill query tokens.
            - The number of prefill key/value tokens.
            - The number of decode query tokens.

    Raises:
        AssertionError: If the number of encoder tokens in `attn_metadata`
        is `None` when required for the calculations.
    """
    num_prefill_query_tokens = 0
    num_decode_query_tokens = 0
    num_prefill_kv_tokens = 0
    if attn_type == AttentionType.ENCODER:
        # Encoder attention is only invoked during prefill phase.
        # The same input servers a both query and key.
        assert attn_metadata.num_encoder_tokens is not None
        num_prefill_query_tokens = attn_metadata.num_encoder_tokens
        num_prefill_kv_tokens = attn_metadata.num_encoder_tokens
        num_decode_query_tokens = 0
    elif attn_type == AttentionType.ENCODER_DECODER:
        assert attn_metadata.num_encoder_tokens is not None
        num_prefill_query_tokens = attn_metadata.num_prefill_tokens
        # The key is the encoder/cross-attention.
        num_prefill_kv_tokens = attn_metadata.num_encoder_tokens
        num_decode_query_tokens = attn_metadata.num_decode_tokens
    else:  # attn_type == AttentionType.DECODER or
        # attn_type == AttentionType.ENCODER_ONLY
        num_prefill_query_tokens = attn_metadata.num_prefill_tokens
        num_prefill_kv_tokens = attn_metadata.num_prefill_tokens
        num_decode_query_tokens = attn_metadata.num_decode_tokens

    return (num_prefill_query_tokens, num_prefill_kv_tokens,
            num_decode_query_tokens)

def _get_query_key_seq_metadata(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    """
    Returns sequence metadata for key and query based on the specified
    attention type and whether input is a prompt.

    This function computes the starting locations and maximum sequence lengths
    for key and query sequences for different attention types.

    Args:
        attn_metadata: The attention metadata object
        is_prompt (bool): A flag indicating if the input is a prompt
        attn_type (AttentionType): The type of attention being used.

    Returns:
        tuple: A tuple containing four integers:
            - Starting location for the query sequence.
            - Maximum sequence length for the query sequence.
            - Starting location for the key sequence.
            - Maximum sequence length for the key sequence.

    Raises:
        AttributeError: If an invalid attention type is provided.
    """
    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.seq_start_loc, max_seq_len)

    elif attn_type == AttentionType.ENCODER_DECODER:
        # This is cross attention between the where the key
        # is the precomputed encoder attention and query
        # is the input sequence.
        # Choose query max length based on whether it is prompt
        # or not.
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER:
        # For encoder attention both the query and the key are same i.e the
        # encoder sequence.
        return (attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert is_prompt, "Should not have decode for encoder only model."
        return (attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len,
                attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")

