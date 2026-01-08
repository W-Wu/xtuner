from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional,List

from mmengine import is_installed
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from xtuner.v1.float8 import Float8Config
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.model.moe.moe import MoEConfig, TransformerConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.utils import get_device, get_logger


if TYPE_CHECKING:
    from .modeling_intern_s1 import InternS1ForConditionalGeneration,InternS1TSForConditionalGeneration

logger = get_logger()


class InternS1VisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    num_channels: int = 3
    patch_size: tuple[int, int] = (14, 14)
    image_size: tuple[int, int] = (448, 448)
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    use_qk_norm: bool = False
    num_hidden_layers: int = 24
    hidden_act: str = "gelu"
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    attention_bias: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 0.1
    layer_scale_init_value: float = 0.1
    hidden_dropout_prob: float = 0.0
    projection_dropout: float = 0.0
    use_absolute_position_embeddings: bool = True
    use_mask_token: bool = False
    use_mean_pooling: bool = True
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import InternS1VisionModel

        return InternS1VisionModel(self)


class InternS1TimeSeriesConfig(BaseModel):
    model_config = ConfigDict(
        title="Base config for InternS1 TimeSeries encoder",
        extra="forbid",
    )
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    apply_spec_augment: bool = False
    attention_dropout: float = 0.0
    begin_suppress_tokens: List[int] = [220, 50257]
    bos_token_id: int = 50257
    classifier_proj_size: int = 256
    d_model: int = 768
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    decoder_layerdrop: float = 0.0
    decoder_layers: int = 6
    decoder_start_token_id: int = 50258
    dropout: float = 0.0
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 3072
    encoder_layerdrop: float = 0.0
    encoder_layers: int = 17
    eos_token_id: int = 50257
    forced_decoder_ids: List[List[int]] = [[1,50259],[2,50359],[3,50363]]
    init_std: float = 0.02
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_time_prob: float = 0.05
    max_length: int = 448
    max_source_positions: int = 1500
    max_target_positions: int = 448
    median_filter_width: int = 7
    num_hidden_layers: int = 6
    num_mel_bins: int = 80
    pad_token_id: int = 50257
    scale_embedding: bool = False
    suppress_tokens: List[int] = [[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362]]
    ts_adapt_in_dim: int = 256
    ts_adapt_out_dim: int = 1024
    ts_cnn_channels: List[int] = [1, 32, 64, 128, 128]
    ts_cnn_kernel_sizes: List[int] = [3, 5, 5, 5]
    ts_cnn_paddings: List[int] = [1, 2, 2, 2]
    ts_cnn_strides: List[int] = [2, 4, 4, 5]
    ts_concat_subsampling_concat_size: int = 2
    ts_concat_subsampling_in_channels: int = 128
    ts_hidden_dim: int = 1024
    use_cache: bool = True
    use_flash_attn: bool = False
    use_weighted_layer_sum: bool = False
    vocab_size: int = 51865
    layer_scale_init_value: float = 0.1
    # attn_impl: Literal["flash_attention", "eager_attention"] = "eager_attention"
    _attn_implementation = "eager"
    output_attentions: bool = True
    output_hidden_states: bool = True
    use_return_dict: bool = True

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_time_series import InternS1TimeSeriesModel

        return InternS1TimeSeriesModel(self)

class InternS1ProjectorConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    downsample_ratio: float = 0.5
    hidden_act: str = "gelu"
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import InternS1MultiModalProjector

        return InternS1MultiModalProjector(self)


class InternS1TimeSeriesProjectorConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    time_series_hidden_size: int = 1024
    text_hidden_size: int = 4096
    downsample_ratio: float = 1.0
    hidden_act: str = "gelu"
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import InternS1TimeSeriesProjector

        return InternS1TimeSeriesProjector(self)
    


class InternS1BaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    vision_config: InternS1VisionConfig
    projector_config: InternS1ProjectorConfig
    text_config: TransformerConfig

    vision_feature_layer: int = -1
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = True
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    image_token_id: int = 152957
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False
    hf_save_worker: int = 16
    dcp_ignore_frozen_params: bool = True

    def build(self) -> "InternS1ForConditionalGeneration":
        from .modeling_intern_s1 import InternS1ForConditionalGeneration

        return InternS1ForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError

class InternS1TSBaseConfig(InternS1BaseConfig):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    vision_config: InternS1VisionConfig
    projector_config: InternS1ProjectorConfig
    text_config: TransformerConfig
    ts_config: InternS1TimeSeriesConfig
    ts_projector_config: InternS1TimeSeriesProjectorConfig

    vision_feature_layer: int = -1
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = True
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    image_token_id: int = 152957
    ts_token_id: int = 152973
    freeze_vision: bool = True
    freeze_projector: bool = True
    freeze_language: bool = True
    freeze_ts_encoder: bool = False
    freeze_ts_projector: bool = False
    hf_save_worker: int = 16
    dcp_ignore_frozen_params: bool = True

    def build(self) -> "InternS1TSForConditionalGeneration":
        from .modeling_intern_s1 import InternS1TSForConditionalGeneration

        return InternS1TSForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError

class InternS1Config(InternS1BaseConfig):
    vision_config: InternS1VisionConfig = InternS1VisionConfig(
        hidden_size=3200,
        intermediate_size=12800,
        num_hidden_layers=45,
        use_qk_norm=True,
        num_attention_heads=25,
        attention_bias=False,
        norm_type="rms_norm",
    )
    projector_config: InternS1ProjectorConfig = InternS1ProjectorConfig(vision_hidden_size=3200, text_hidden_size=4096)
    text_config: MoEConfig = Qwen3MoE235BA22Config(vocab_size=153216)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None


class InternS1MiniConfig(InternS1BaseConfig):
    vision_config: InternS1VisionConfig = InternS1VisionConfig()
    projector_config: InternS1ProjectorConfig = InternS1ProjectorConfig()
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig(vocab_size=153216)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None


class InternS1MiniTSConfig(InternS1TSBaseConfig):
    vision_config: InternS1VisionConfig = InternS1VisionConfig()
    projector_config: InternS1ProjectorConfig = InternS1ProjectorConfig()
    ts_config: InternS1TimeSeriesConfig = InternS1TimeSeriesConfig()
    ts_projector_config: InternS1TimeSeriesProjectorConfig = InternS1TimeSeriesProjectorConfig()
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig(vocab_size=153216)

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None