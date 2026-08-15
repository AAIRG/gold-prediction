"""Autoregressive backbones and the combined model.

    backbone_lstm.py          -> aligned scaled lags + embeddings + time features
    backbone_transformer.py   -> unmasked encoder + causal future decoder
    conditioned_flow_model.py -> backbone + flow head, eq. 4 and the §4.1 loss

Milestones: M4 (LSTM-Real-NVP), M8 (Transformer-MAF).
"""
