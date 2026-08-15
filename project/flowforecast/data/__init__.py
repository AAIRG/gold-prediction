"""Data layer.

    datasets.py   -> loaders + DATASET_SPECS from Appendix A, Table 2
    windowing.py  -> random window of size T (§4.1)
    covariates.py -> time features + lag features (§4.2, Appendix D.1)
    scaling.py    -> mean-scaling (§4) and dequantization (§2.1, §5.2)
    toy_pipes.py  -> simulated system of pipes (§5.1, Figures 3 and 4)

Milestones: M0 (load Exchange), M3 (full data layer), M6 (toy pipes).
"""
