"""TransformerTempFlow-compatible encoder/decoder backbone."""

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module):
    """Encode context without a source mask and decode future causally.

    ``covariates`` is already aligned to the target time.  It contains lagged
    values and calendar features; target-dimension embeddings are appended here.
    The forward result contains decoder states only, matching TransformerTempFlow
    where the likelihood is evaluated on future targets only.
    """

    loss_on_future_only = True
    loss_uses_observed_mask = False

    def __init__(self, input_size: int, cond_size: int, d_model: int,
                 n_heads: int = 8, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, dropout: float = 0.1,
                 context_length: int | None = None, max_len: int = 5000,
                 time_feature_size: int = 0):
        super().__init__()
        try:
            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by n_heads={n_heads}"
                )
            self.input_size = input_size
            self.cond_size = cond_size
            self.d_model = d_model
            self.hidden_size = d_model
            self.context_length = context_length
            self.time_feature_size = int(time_feature_size)
            self.target_embedding = nn.Embedding(input_size, 1)
            network_input = cond_size + input_size
            self.input_norm = nn.LayerNorm(network_input)
            self.encoder_input = nn.Linear(network_input, d_model)
            self.decoder_input = nn.Linear(network_input, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=n_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
            )
            self.register_buffer(
                "tgt_mask",
                torch.triu(torch.full((max_len, max_len), float("-inf")), 1),
                persistent=False,
            )
        except Exception as e:
            raise Exception(f"TransformerBackbone - __init__ - Init error: {e}") from e

    def _inputs(self, covariates: torch.Tensor) -> torch.Tensor:
        try:
            if covariates.shape[-1] != self.cond_size:
                raise ValueError(
                    f"expected {self.cond_size} aligned covariates, got "
                    f"{covariates.shape[-1]}"
                )
            batch = covariates.shape[0]
            emb = self.target_embedding.weight[:, 0].view(1, 1, -1)
            emb = emb.expand(batch, covariates.shape[1], -1)
            split = covariates.shape[-1] - self.time_feature_size
            lag_part = covariates[:, :, :split]
            time_part = covariates[:, :, split:]
            values = torch.cat([lag_part, emb, time_part], dim=-1)
            values = torch.nan_to_num(values, nan=0.0, posinf=1e3, neginf=-1e3)
            return values.clamp(-1e3, 1e3)
        except Exception as e:
            raise Exception(f"TransformerBackbone - _inputs - Execution error: {e}") from e

    def _mask(self, size: int, device: torch.device, dtype: torch.dtype):
        return self.tgt_mask[:size, :size].to(device=device, dtype=dtype)

    def encode(self, covariates: torch.Tensor, context_length: int | None = None):
        try:
            ctx = context_length if context_length is not None else self.context_length
            if ctx is None:
                raise ValueError("context_length is required for TransformerBackbone")
            src = self._inputs(covariates[:, :int(ctx)])
            encoded = self.encoder_input(self.input_norm(src)).transpose(0, 1)
            return self.transformer.encoder(encoded)
        except Exception as e:
            raise Exception(f"TransformerBackbone - encode - Execution error: {e}") from e

    def decode(self, covariates: torch.Tensor, memory: torch.Tensor):
        try:
            tgt = self._inputs(covariates)
            length = tgt.shape[1]
            return self.transformer.decoder(
                self.decoder_input(self.input_norm(tgt)).transpose(0, 1), memory,
                tgt_mask=self._mask(length, tgt.device, tgt.dtype),
            ).transpose(0, 1)
        except Exception as e:
            raise Exception(f"TransformerBackbone - decode - Execution error: {e}") from e

    def forward(self, x: torch.Tensor, covariates: torch.Tensor,
                context_length: int | None = None) -> torch.Tensor:
        try:
            del x
            ctx = context_length if context_length is not None else self.context_length
            if ctx is None or ctx >= covariates.shape[1]:
                raise ValueError("Transformer requires context and future rows")
            memory = self.encode(covariates, ctx)
            return self.decode(covariates[:, int(ctx):], memory)
        except Exception as e:
            raise Exception(f"TransformerBackbone - forward - Execution error: {e}") from e
