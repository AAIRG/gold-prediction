"""Conditional normalizing flow blocks (paper §2.1).

    coupling.py  -> ConditionalCouplingLayer (Real NVP affine coupling, eq. 2)
    batchnorm.py -> BatchNormFlow (BatchNorm as a bijection, §2.1 last para)
    flow.py      -> ConditionalFlow: stacks K blocks, log_prob(x, h) / sample(h)
    made.py      -> MADE autoregressive network (used by MAF)
    maf.py       -> ConditionalMAFLayer (Masked Autoregressive Flow)

Milestones: M1 (unconditional), M2 (conditioning), M7 (MAF).
"""
