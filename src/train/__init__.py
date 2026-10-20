"""Our own minimal DPO trainer (plain PyTorch + PEFT).

We do not use TRL's DPOTrainer because its internals change between versions and
the variants here (R-DPO length regulariser, AuxDPO per-response offsets, a
restricted-capacity policy, the online two-timescale variant) need the loss and
the parameter set under our control.  The loss is ~40 lines; see dpo_loss.py.
"""
