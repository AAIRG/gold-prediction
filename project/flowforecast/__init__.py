"""
flowforecast
============

From-scratch PyTorch implementation of "Multivariate Probabilistic Time Series
Forecasting via Conditioned Normalizing Flows" (Rasul, Sheikh, Schuster,
Bergmann & Vollgraf, ICLR 2021, arXiv:2002.06103v3).

The model factorizes the joint predictive distribution autoregressively
(paper eq. 4):

    p_X(x_{t0:T} | x_{1:t0-1}, c_{1:T}; theta) = prod_{t=t0}^{T} p_X(x_t | h_t; theta)

where h_t comes from an LSTM (eq. 3) or a causally masked Transformer, and
p_X(x_t | h_t) is a conditional Real NVP or MAF. The flow acts on the
D-dimensional cross-section x_t; all temporal structure lives in the backbone.

Variants reported by the paper: LSTM-Real-NVP, LSTM-MAF, Transformer-MAF.

Sub-packages:
    flows      -> conditional Real NVP / MAF blocks (§2.1)
    models     -> LSTM / Transformer backbone + flow head (§4)
    data       -> windowing, covariates, mean-scaling, toy pipes (§4.1-4.2, §5.1)
    training   -> §4.1 objective with Adam
    evaluation -> sampling (§4.3), CRPS / CRPS_sum (§5.2), plots
    utils      -> seeding, device, config loading

Build order and acceptance criteria are in ../roadmap.md (milestones M0-M8).
No gluonts / pytorch-ts dependency: the reference repo is used only for
cross-checking behaviour.
"""

__version__ = "0.1.0"
