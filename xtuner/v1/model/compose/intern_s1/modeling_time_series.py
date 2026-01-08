from functools import partial
from torch import nn
import torch
from typing import Union, Optional
from typing_extensions import override
import numpy as np

from transformers.modeling_outputs import BaseModelOutput
from transformers import WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

import math

try:
    from timm.layers import DropPath

    has_timm = True
except:
    has_timm = False
from tqdm import tqdm
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .intern_s1_config import InternS1TimeSeriesConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.ops.attn_imp import attn_impl_mapping
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from xtuner.v1.module import RMSNorm
from xtuner.v1.ops.others import Dropout
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import get_logger
from dataclasses import dataclass

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


NORM2FN = {"layer_norm": nn.LayerNorm, "rms_norm": RMSNorm}




class InternS1TimeSeriesConcatSubsampling(nn.Module):
    def __init__(self, in_channels: int, concat_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size
    
    def forward(self, ts_signals: torch.Tensor, ts_lens: torch.Tensor):
        if ts_signals.shape[1] % 2 != 0:
            ts_signals = ts_signals[:, :-1, :]
        even_frames = ts_signals[:, ::2, :]  
        odd_frames = ts_signals[:, 1::2, :]
        ts_signals = torch.cat((even_frames, odd_frames), dim=2)
        ts_lens = ts_lens // 2
        return ts_signals, ts_lens


class InternS1TimeSeriesFixPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model,dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(torch.float32) # (max_len, 1, d_model)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x.clone()


class InternS1TimeSeriesMultiChannelAdaptiveSubsampling(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8,num_encoder_layers = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pos_encoder = InternS1TimeSeriesFixPositionalEncoding(d_model=hidden_dim)
        self.subsampling = InternS1TimeSeriesConcatSubsampling(128, 2)


    def forward(self, inputs, input_lens, sr):
        features, feature_lens = self.forward_patch(inputs, input_lens, sr)
        outputs = features
        output_lens = feature_lens
        return outputs, output_lens
    
    def forward_patch(self, inputs, input_lens, sr):
        sr = sr.float()
        strides = torch.floor(160/((1+torch.exp(-sr/100))**6))
        patch_sizes = strides * 2
        patched_outputs = []
        output_lens = []

        for i in range(len(inputs)):
            seq = inputs[i] # [seq_len, num_channel]
            ps = patch_sizes[i].item()
            st = strides[i].item()
            le = input_lens[i]
            
            output_len = torch.ceil((le - ps) / st) + 1
            pad_len = ((output_len - 1) * st + ps - le).long().item()
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq = nn.functional.pad(seq, (0, 0, 0, pad_len), "constant", 0)
            assert output_len > 0,(seq.shape, ps,st,le,output_len)
            output_lens.append(output_len)
            indices = (torch.arange(0, output_len * st, st).unsqueeze(1) + torch.arange(ps)).long()
            patched = seq[indices]

            output = self.forward_encoder(patched)    #[num_patch, D]
            patched_outputs.append(output)

        outputs = nn.utils.rnn.pad_sequence(patched_outputs, batch_first=True)
        output_lens = torch.tensor(output_lens).squeeze().to(outputs.device).long()
        if output_lens.ndim == 0:
            output_lens = output_lens.unsqueeze(0)

        outputs, output_lens = self.subsampling(outputs.clone(), output_lens.clone())
        return outputs, output_lens
    
    def forward_encoder(self, x):
        num_patch, patch_len, C = x.shape
        # conv1
        x = x.reshape(num_patch*C, 1, patch_len)    # 每个 channel 当作独立样本送入 conv1
        x = nn.functional.relu((self.conv(x))) # [B*C, D1, L]
        x = x.permute(2,0,1)    # [L, B*C, D1]

        x = self.pos_encoder(x) # [L, B*C, D1]
        x = self.transformer_encoder(x.to(torch.bfloat16))
        x = x.mean(0)

        x = x.reshape(num_patch,C,-1)

        return x.mean(1)

class InternS1TimeSeriesEncoder(BaseModel):
    def __init__(self, config: InternS1TimeSeriesConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # self.post_init()

        self.mask_type = None
        self.chunk_length = None

        self.adapt_in = nn.Linear(config.ts_adapt_in_dim, 80)
        self.adapt_out = nn.Linear(self.embed_dim, config.ts_adapt_out_dim)
        
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value
        
    def define_masktype(self, masktype, chunk_length=None):
        self.mask_type = masktype
        self.chunk_length = chunk_length

    def _make_causal_mask(self,
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask
    
    def prepare_chunk_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        
        block_size = round(self.chunk_length / 4 * 2)
        matrix_size = input_shape[1]
        
        matrix = torch.ones(matrix_size, matrix_size)
        
        num_full_blocks = round(matrix_size // block_size)
        remainder = matrix_size % block_size  
        for i in range(num_full_blocks):
            row_start = i * block_size
            col_start = i * block_size
            matrix[row_start:row_start + block_size, col_start:col_start + block_size] = torch.zeros(block_size, block_size)
        
        if remainder > 0:
            last_row_start = num_full_blocks * block_size
            last_col_start = num_full_blocks * block_size
            matrix[last_row_start:last_row_start + remainder, last_col_start:last_col_start + remainder] = torch.zeros(remainder, remainder)
        
        matrix = matrix * -65504
        matrix = matrix.unsqueeze(0).unsqueeze(0).repeat(input_shape[0], 1, 1, 1)
        attention_mask = matrix.to(inputs_embeds.device)
        return attention_mask
    
    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # torch.Size([1, 100, 768])
        embed_pos = self.embed_positions.weight         # torch.Size([1500, 768])
        
        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len-embed_pos.shape[0]]

            embed_pos = nn.functional.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        attention_mask = None
        if self.mask_type == 'chunk':
            attention_mask = self.prepare_chunk_attention_mask(attention_mask, input_shape, inputs_embeds)
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (self.layer_norm(hidden_states),)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    @torch.no_grad()
    def init_weights(self):
        initialized_params: set[str] = set()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                init_params(module.weight, partial(torch.nn.init.normal_, mean=0.0, std=self.config.initializer_range))
                initialized_params.add(f"{name}.weight")
                if module.bias is not None:
                    init_params(module.bias, torch.nn.init.zeros_)
                    initialized_params.add(f"{name}.bias")

            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                init_params(module.weight, torch.nn.init.ones_)
                init_params(module.bias, torch.nn.init.zeros_)
                initialized_params.add(f"{name}.weight")
                initialized_params.add(f"{name}.bias")

        if (missing := {name for name, _ in self.named_parameters()} - initialized_params):
            raise RuntimeError(f"{missing} is not initialized")
        return initialized_params

@dataclass
class InternS1TimeSeriesModelOutput(BaseModelOutput):
    ts_pad_mask: Optional[torch.FloatTensor] = None

class InternS1TimeSeriesModel(BaseModel):
    config: InternS1TimeSeriesConfig

    def __init__(self, config: InternS1TimeSeriesConfig):
        super().__init__()
        self.config = config
        self.encoder_embed = InternS1TimeSeriesMultiChannelAdaptiveSubsampling()
        self.encoder = InternS1TimeSeriesEncoder(config)

        self._hf_prefix = "model.ts_tower."
        self._init_load_spec()

    def get_input_embeddings(self):
        return self.encoder_embed
    
    def make_pad_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
        lengths:
            A 1-D tensor containing sentence lengths.
        max_len:
            The length of masks.
        Returns:
        Return a 2-D bool tensor, where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.

        >>> lengths = torch.tensor([1, 3, 2, 5])
        >>> make_pad_mask(lengths)
        tensor([[False,  True,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False, False, False]])
        """
        assert lengths.ndim == 1, lengths.ndim
        max_len = lengths.max()
        n = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
        return expaned_lengths >= lengths.unsqueeze(-1)

    @torch.no_grad()
    def init_weights(self) -> None:
        initialized_params: set[str] = set()

        for name, module in self.encoder_embed.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                init_params(module.weight, partial(torch.nn.init.normal_, mean=0.0, std=self.config.initializer_range))
                initialized_params.add(f"embeddings.{name}.weight")
                if module.bias is not None:
                    init_params(module.bias, torch.nn.init.zeros_)
                    initialized_params.add(f"embeddings.{name}.bias")

            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                init_params(module.weight, torch.nn.init.ones_)
                initialized_params.add(f"embeddings.{name}.weight")
                if module.bias is not None:
                    init_params(module.bias, torch.nn.init.zeros_)  # type: ignore
                    initialized_params.add(f"embeddings.{name}.bias")

        expected_param_name = {self._clean_param_name(name) for name, _ in self.named_parameters()}
        if (missing := expected_param_name - initialized_params):
            raise RuntimeError(f"{missing} is not initialized")


    def forward(
            self,
            time_series_signals: Optional[torch.FloatTensor] = None,
            ts_lens: Optional[torch.Tensor] = None,
            sr: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # time_series_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[tuple, BaseModelOutput]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if time_series_signals is None and time_series_embeds is None:
        #     raise ValueError('You have to specify time_series_signals or time_series_embeds')

        # # 可以直接传入这个embed之后的值，但是需要维度都对得上
        # if time_series_embeds is not None and len(time_series_embeds.shape) == 3 and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim:
        #     time_series_embeds = time_series_embeds
        # else:
        if (isinstance(time_series_signals,list) and len(time_series_signals[0].shape) == 2) \
            or (isinstance(time_series_signals, torch.Tensor) and  len(time_series_signals.shape) == 3):
            time_series_embeds, ts_lens = self.encoder_embed(time_series_signals, ts_lens, sr)
        else:
            raise ValueError(f'wrong time_series_signals size: {time_series_signals[0].shape}')

        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        encoder_outputs = self.encoder(
            input_features=time_series_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ts_lens after encoder
        ts_lens = (ts_lens+1) // 2
        assert torch.all(ts_lens > 0), f"The length of time_series_embeds is so small. ts_lens: {ts_lens}"
        
        src_key_padding_mask = self.make_pad_mask(ts_lens)

        last_hidden_state = encoder_outputs.last_hidden_state
        if not return_dict:
            return (last_hidden_state, ) + encoder_outputs[1:]

        return InternS1TimeSeriesModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,    # encoder_state，for every layer
            attentions=encoder_outputs.attentions,
            ts_pad_mask=src_key_padding_mask
        )

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        if not checkpoint_preserve_rng_state and self.config.dropout > 0.0:
            checkpoint_preserve_rng_state = True
            logger.warning(f"When using dropout[{self.config.dropout}], checkpoint_preserve_rng_state is set to True to avoid issues.")
        if not checkpoint_preserve_rng_state and self.config.attention_dropout > 0.0:
            checkpoint_preserve_rng_state = True
            logger.warning(f"When using dropout[{self.config.attention_dropout}], checkpoint_preserve_rng_state is set to True to avoid issues.")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )

        # NOTE: 在 cpu_offload 模式下，mesh 应该是 cuda 的，在 meta fully_shard 后在调用 .to_empty(device=cpu)
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        recompute_ratio = fsdp_config.recompute_ratio
        num_recompute_layers = int(len(self.encoder.layer) * recompute_ratio)

        generator = torch.Generator()
        generator.manual_seed(dist.get_rank())
        shuffled_layers_idxs = torch.randperm(len(self.encoder.layer), generator=generator)

        for layer_idx in tqdm(shuffled_layers_idxs, desc="[Time Series Fully Shard]"):
            layer = self.encoder.layer[layer_idx]

            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer,
                                           preserve_rng_state=checkpoint_preserve_rng_state,
                                           checkpoint_impl=CheckpointImpl.REENTRANT)

            self.encoder.layer[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy()
                if fsdp_config.cpu_offload
                else None,
            )

        for layer_cur, layer_next in zip(self.encoder.layer[:-1], self.encoder.layer[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self
