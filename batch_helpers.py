"""batch_helpers.py — batch-aware variants of the per-experiment utilities.

Two things each experiment script needs to swap when going batched:

1. The initial-condition sampler. The original `sample_top(device)`
   returns a single (4,) tensor; the batched equivalent is
   `sample_top_batched(B, device)` returning (B, 4) with each row
   independently sampled from the same uniform distribution.

2. The SA (single-actuated) dynamics wrapper. The v5/v6 experiments
   patch `mpc.true_RK4_disc` to freeze the elbow. The original wrapper
   only handles unbatched (4,) state; the batched-safe version below
   uses `...` indexing so it works for both unbatched and batched.

eval2k_batched is provided as a convenience for fast multi-seed evaluation
— but eval is normally fine unbatched (it's not training-hot).
"""

import math
from typing import Optional

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# Initial-condition samplers (batched)
# ──────────────────────────────────────────────────────────────────────────
def sample_top_batched(
    B: int,
    device: torch.device,
    *,
    pert_q1:  float = 0.30,
    pert_q1d: float = 0.30,
    pert_q2:  float = 0.30,
    pert_q2d: float = 0.30,
    goal_q1:  float = math.pi,
) -> torch.Tensor:
    """Batched replacement for the per-experiment `sample_top(device)` helper.

    Returns (B, 4) with each row drawn uniformly from
        [goal_q1 - pert_q1, goal_q1 + pert_q1] × [-pert_q1d, +pert_q1d]
        × [-pert_q2,         +pert_q2]          × [-pert_q2d, +pert_q2d]

    Default perts match exp_hardware_v1/v2 (broad recovery). Override for
    tighter sampling (SA experiments use 0.05 on q2/q2d).
    """
    pert = torch.tensor(
        [pert_q1, pert_q1d, pert_q2, pert_q2d],
        device=device, dtype=torch.float64,
    )
    centre = torch.tensor(
        [goal_q1, 0.0, 0.0, 0.0],
        device=device, dtype=torch.float64,
    )
    # uniform(-1, +1) × pert + centre, per element
    u = torch.rand((B, 4), device=device, dtype=torch.float64) * 2.0 - 1.0
    return centre + u * pert


def sample_x0_mixed_batched(
    B: int,
    device: torch.device,
    *,
    rest_frac: float = 0.5,
    spin_frac: float = 0.3,
    top_frac:  float = 0.2,
    pert_q1:  float = 0.30,
    pert_q1d: float = 0.30,
    pert_q2:  float = 0.30,
    pert_q2d: float = 0.30,
    spin_max: float = 5.0,
    goal_q1:  float = math.pi,
) -> torch.Tensor:
    """Mixed batched IC sampling — matches the spirit of v1's sample_x0_mixed.

    Each trajectory is independently assigned a category:
      * rest_frac   → start at zero (swing-up)
      * spin_frac   → start with random q1_dot ∈ [-spin_max, +spin_max]
                       (high-velocity recovery)
      * top_frac    → start near goal (stabilisation perturbation)

    Fractions must sum to 1 (within 1e-6).
    """
    total = rest_frac + spin_frac + top_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"rest_frac+spin_frac+top_frac must sum to 1; got {total}"
        )

    # Allocate categories by floor + remainder rule, then shuffle.
    n_rest = int(round(B * rest_frac))
    n_spin = int(round(B * spin_frac))
    n_top  = B - n_rest - n_spin

    out = torch.zeros((B, 4), device=device, dtype=torch.float64)
    if n_spin > 0:
        out[n_rest:n_rest + n_spin, 1] = (
            torch.rand(n_spin, device=device, dtype=torch.float64) * 2 - 1
        ) * spin_max
    if n_top > 0:
        out[n_rest + n_spin:] = sample_top_batched(
            n_top, device,
            pert_q1=pert_q1, pert_q1d=pert_q1d,
            pert_q2=pert_q2, pert_q2d=pert_q2d,
            goal_q1=goal_q1,
        )

    # Shuffle so categories aren't contiguous (matters for any code that
    # accidentally takes batched-mean-over-prefix).
    perm = torch.randperm(B, device=device)
    return out[perm]


# ──────────────────────────────────────────────────────────────────────────
# Single-actuated dynamics wrapper (batch-aware)
# ──────────────────────────────────────────────────────────────────────────
def wrap_sa_dynamics_batched(mpc):
    """Patch mpc.true_RK4_disc to freeze the elbow (q2 = 0 always).

    Drop-in replacement for the per-experiment `wrap_sa_dynamics`. Works
    for both unbatched state (4,) and batched state (..., 4). The
    elbow-freeze is implemented by zeroing q2, q2d and u2 before the
    RK4 step and reasserting them on output.
    """
    orig_rk4 = mpc.true_RK4_disc

    def sa_rk4(x: torch.Tensor, u: torch.Tensor, dt, n_sub: int = 10) -> torch.Tensor:
        # Zero the elbow components regardless of leading shape.
        # x[..., :2] = [q1, q1_dot] kept;  x[..., 2:] = [q2, q2_dot] zeroed.
        zero_x_tail = torch.zeros_like(x[..., 2:])
        zero_u_tail = torch.zeros_like(u[..., 1:])
        x_in = torch.cat([x[..., :2], zero_x_tail], dim=-1)
        u_in = torch.cat([u[..., :1], zero_u_tail], dim=-1)

        x_out = orig_rk4(x_in, u_in, dt, n_sub)

        # Zero the elbow components of the output too (rigid-link model).
        return torch.cat(
            [x_out[..., :2], torch.zeros_like(x_out[..., 2:])],
            dim=-1,
        )

    mpc.true_RK4_disc = sa_rk4
    return mpc


# ──────────────────────────────────────────────────────────────────────────
# Probe helper — uses batched lin_net forward for B=1 (matches existing API)
# ──────────────────────────────────────────────────────────────────────────
def probe_network_batched(model, mpc, device):
    """Probe a (now-batched-capable) lin_net at three canonical states.

    Identical output schema to the legacy probe_network() helper used in
    all experiment scripts. Works whether the network is batched or
    unbatched internally — the lin_net.forward() dual-mode path handles
    both single (5, 4) and batched (B, 5, 4) inputs transparently.
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for name, q1 in [("bot", 0.0), ("mid", math.pi / 2), ("top", math.pi)]:
            s = torch.tensor([q1, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
            hist = s.unsqueeze(0).expand(5, -1).contiguous()  # (5, 4)
            gQ, _, fe, _, _, _ = model(hist, mpc.q_base_diag, mpc.r_base_diag)
            results[name] = {
                "Q_q1":    float(gQ[:, 0].mean()),
                "fe_norm": float(fe.norm()),
            }
    model.train()
    return results
