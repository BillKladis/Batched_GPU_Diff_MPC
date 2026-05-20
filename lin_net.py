"""
lin_net.py — Stage D LinearizationNetwork with kickstart bias on f_head.

WHAT IS NEW IN THIS REVISION:
- The final-layer bias of f_head is initialised to a half-sinusoid pattern
  in the τ1 components, zero in the τ2 components. This puts the network
  output at initialisation roughly at:
        f_extra[k, 0] ≈ kickstart_amp × sin(π (k + 0.5) / N) for k=0..N-1
        f_extra[k, 1] ≈ 0
  so the very first rollout produces non-zero alternating-magnitude τ1
  rather than near-zero output.
- The bias is LEARNABLE — the network will modify or discard it during
  training. We are only providing a starting point that breaks the
  symmetry of the constant-direction failure mode.

The kickstart amplitude is configurable via f_kickstart_amp (default 1.0,
which after tanh×bound gives ~32% of full f_extra range — strong enough
to create real motion, weak enough to keep the QP solution in the interior
rather than saturating immediately).

Architecture otherwise unchanged from the previous Stage D version:
  state_history → state_encoder → trunk → {q_head, r_head, f_head}

BATCH SUPPORT (this revision):
- forward() now accepts either an unbatched state history x_sequence of
  shape (5, state_dim) or a batched x_sequence of shape (B, 5, state_dim).
  Output shapes mirror the input — unbatched in → unbatched out, batched
  in → batched out (with an extra leading B dim on every returned tensor).
  Existing unbatched callers (Simulate.py, hardware_deploy.py, …) work
  unchanged.
- Applies to all three classes in this file: LinearizationNetwork,
  LinearizationNetworkSC, and (transparently, via dispatch) the
  SeparatedLinearizationNetwork wrapper.
"""

import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LinearizationNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        gate_range_q: float = 0.95,
        gate_range_r: float = 0.20,
        f_extra_bound: float = 3.0,
        f_kickstart_amp: float = 1.0,
        # Qf-head: optional fourth head that outputs 4 gates that scale
        # the diagonal of the MPC's terminal cost Qf at inference.
        # Default disabled (gate_range_qf=0) for back-compat with all
        # existing checkpoints which have no qf_head weights.
        # When enabled, gates_Qf = 1.0 + gate_range_qf * tanh(raw); a
        # raw of 0 → gate of 1.0 (network can still default to "no
        # change"). Same pattern as gates_Q.
        gate_range_qf: float = 0.0,
        # Goal-centred sin/cos input representation.
        # Replaces [q1,dq1,q2,dq2] → [sin(q1-π),cos(q1-π),dq1/8,sin(q2),cos(q2),dq2/8].
        # cos(q1-π)≈+1 at the goal (inverted), ≈-1 at hanging — makes the
        # near-goal region explicit to the trunk without any learned mapping.
        # Only valid for the double-pendulum (state_dim=4, 2 angle dims).
        use_sincos: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.use_sincos = use_sincos
        self.horizon = horizon

        for name, value in (
            ("gate_range_q", gate_range_q),
            ("gate_range_r", gate_range_r),
        ):
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if not (0.0 <= gate_range_qf < 1.0):
            raise ValueError(f"gate_range_qf must be in [0, 1), got {gate_range_qf}")
        if f_extra_bound <= 0:
            raise ValueError(f"f_extra_bound must be positive, got {f_extra_bound}")
        if f_kickstart_amp < 0:
            raise ValueError(f"f_kickstart_amp must be ≥ 0, got {f_kickstart_amp}")

        self.gate_range_q = gate_range_q
        self.gate_range_r = gate_range_r
        self.gate_range_qf = gate_range_qf
        self.f_extra_bound = f_extra_bound
        self.f_kickstart_amp = f_kickstart_amp

        # ── Input branch ────────────────────────────────────────────────────
        # With use_sincos the 4-D state expands to 6-D per frame:
        #   [q1,dq1,q2,dq2] → [sin(q1-π),cos(q1-π),dq1/8,sin(q2),cos(q2),dq2/8]
        # The sin/cos values are already in [-1,1]; velocities are pre-divided by 8.
        obs_dim = 6 if use_sincos else state_dim
        state_input_dim = 5 * obs_dim
        enc_dim = hidden_dim
        if use_sincos:
            # sin/cos outputs are in [-1,1]; no further scaling needed.
            # Store a ones buffer so code paths that reference state_scale still work.
            state_scale = torch.ones(state_input_dim, dtype=torch.float64)
        else:
            state_scale = torch.tensor(
                [math.pi, 8.0, math.pi, 8.0], dtype=torch.float64
            ).repeat(5)
        self.register_buffer("state_scale", state_scale)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, enc_dim),
            nn.Tanh(),
        )

        # ── Trunk ───────────────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # ── Heads ───────────────────────────────────────────────────────────
        q_out_dim = (horizon - 1) * state_dim
        r_out_dim = horizon * control_dim
        f_out_dim = horizon * control_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, q_out_dim),
        )
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, r_out_dim),
        )
        self.f_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, f_out_dim),
        )

        # Qf-head: outputs 4 numbers (one per state dim) → terminal cost gates.
        # Created always so loading old checkpoints with strict=False works.
        qf_out_dim = state_dim
        self.qf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, qf_out_dim),
        )

        self._initialize_weights()

        self.metadata = {
            "state_dim": state_dim,
            "control_dim": control_dim,
            "hidden_dim": hidden_dim,
            "horizon": horizon,
            "gate_range_q": gate_range_q,
            "gate_range_r": gate_range_r,
            "gate_range_qf": gate_range_qf,
            "f_extra_bound": f_extra_bound,
            "f_kickstart_amp": f_kickstart_amp,
            "use_sincos": use_sincos,
            "architecture": "stage_d_three_head_qrf_kickstart",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ──────────────────────────────────────────────────────────────────────
    def _initialize_weights(self):
        def _init_block(seq: nn.Sequential, final_std: float = 0.01):
            layers = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            for layer in layers[:-1]:
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)
            final = layers[-1]
            nn.init.normal_(final.weight, mean=0.0, std=final_std)
            nn.init.zeros_(final.bias)

        _init_block(self.state_encoder, final_std=0.1)
        _init_block(self.trunk, final_std=0.1)
        _init_block(self.q_head, final_std=0.01)
        _init_block(self.r_head, final_std=0.01)
        # f_head: small final std on weights so output ≈ bias at initialisation,
        # then the bias kickstart below dictates the starting f_extra.
        _init_block(self.f_head, final_std=0.001)
        # qf_head: tiny final std so raw output ≈ 0 → gates_Qf ≈ 1.0 at init.
        # This makes the freshly-added head a no-op until training learns to
        # modulate it, preserving compatibility with checkpoints loaded with
        # strict=False (qf_head will be randomly initialised with output ≈ 1).
        _init_block(self.qf_head, final_std=0.001)

        # ── Kickstart bias on f_head's final layer ──────────────────────
        # Half-sinusoid across the horizon in τ1; zero in τ2.
        # After tanh × bound, the output magnitude at step k is:
        #   bound × tanh(amp × sin(π(k+0.5)/N))
        # which for amp=1.0, bound=3.0 gives a smoothly-alternating
        # peak ≈ ±0.95 across the horizon.
        if self.f_kickstart_amp > 0.0:
            f_final = list(self.f_head.modules())[-1]   # last Linear
            n_u = self.control_dim
            init_bias = torch.zeros(self.horizon * n_u, dtype=f_final.bias.dtype)
            for k in range(self.horizon):
                phase = math.pi * (k + 0.5) / self.horizon  # in (0, π)
                init_bias[k * n_u + 0] = self.f_kickstart_amp * math.sin(phase)
                # τ2 stays at zero
            with torch.no_grad():
                f_final.bias.copy_(init_bias)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_sequence: torch.Tensor,
        q_base_diag: Optional[torch.Tensor] = None,
        r_base_diag: Optional[torch.Tensor] = None,
    ):
        # Detect batching by rank:
        #   (5, state_dim)         → unbatched (legacy path)
        #   (B, 5, state_dim)      → batched
        # Internally we always process as batched, then squeeze on the way out.
        is_batched = (x_sequence.dim() == 3)
        if not is_batched:
            x_sequence = x_sequence.unsqueeze(0)        # (1, 5, state_dim)
        B = x_sequence.shape[0]

        if self.use_sincos:
            # x_sequence: (B, 5, 4) = [q1, dq1, q2, dq2]
            q1  = x_sequence[..., 0:1]                  # (B, 5, 1)
            dq1 = x_sequence[..., 1:2]
            q2  = x_sequence[..., 2:3]
            dq2 = x_sequence[..., 3:4]
            x_obs = torch.cat([
                torch.sin(q1 - math.pi),                # 0 at goal and hanging; ±1 at ±π/2
                torch.cos(q1 - math.pi),                # +1 at goal (q1=π); -1 at hanging (q1=0)
                dq1 / 8.0,
                torch.sin(q2),
                torch.cos(q2),
                dq2 / 8.0,
            ], dim=-1)                                  # (B, 5, 6)
            x_flat = x_obs.reshape(B, -1)               # (B, 30)
            x_normed = x_flat                           # state_scale is ones
        else:
            x_flat = x_sequence.reshape(B, -1)          # (B, 5*state_dim)
            # state_scale: (5*state_dim,) broadcasts cleanly across batch.
            x_normed = x_flat / self.state_scale

        state_emb = self.state_encoder(x_normed)        # (B, hidden)
        features  = self.trunk(state_emb)               # (B, hidden)

        raw_Q   = self.q_head(features).reshape(B, self.horizon - 1, self.state_dim)
        gates_Q = 1.0 + self.gate_range_q * torch.tanh(raw_Q)

        raw_R   = self.r_head(features).reshape(B, self.horizon, self.control_dim)
        gates_R = 1.0 + self.gate_range_r * torch.tanh(raw_R)

        raw_F   = self.f_head(features).reshape(B, self.horizon, self.control_dim)
        f_extra = self.f_extra_bound * torch.tanh(raw_F)

        # Qf gates: emitted only when gate_range_qf > 0 (legacy default 0
        # → None → MPC uses base Qf unchanged, preserving back-compat).
        if self.gate_range_qf > 0.0:
            raw_QF = self.qf_head(features).reshape(B, self.state_dim)
            gates_Qf = 1.0 + self.gate_range_qf * torch.tanh(raw_QF)
        else:
            gates_Qf = None

        # q_base_diag, r_base_diag are 1-D tensors of shape (state_dim,) and
        # (control_dim,) respectively. They broadcast across the (B, H, …) gates.
        q_diags = (q_base_diag * gates_Q) if q_base_diag is not None else None
        r_diags = (r_base_diag * gates_R) if r_base_diag is not None else None

        if not is_batched:
            # Squeeze the batch dim back out to match the legacy contract.
            gates_Q = gates_Q.squeeze(0)
            gates_R = gates_R.squeeze(0)
            f_extra = f_extra.squeeze(0)
            if gates_Qf is not None:
                gates_Qf = gates_Qf.squeeze(0)
            if q_diags is not None:
                q_diags = q_diags.squeeze(0)
            if r_diags is not None:
                r_diags = r_diags.squeeze(0)

        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf

    # ──────────────────────────────────────────────────────────────────────
    def save(self, filepath: str, metadata: Optional[Dict] = None):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if metadata:
            self.metadata.update(metadata)
        torch.save(
            {"model_state_dict": self.state_dict(), "metadata": self.metadata},
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        model = cls(
            state_dim       = metadata.get("state_dim", 4),
            control_dim     = metadata.get("control_dim", 2),
            hidden_dim      = metadata.get("hidden_dim", 128),
            horizon         = metadata.get("horizon", 10),
            gate_range_q    = metadata.get("gate_range_q", 0.95),
            gate_range_r    = metadata.get("gate_range_r", 0.20),
            gate_range_qf   = metadata.get("gate_range_qf", 0.0),     # 0 for back-compat
            f_extra_bound   = metadata.get("f_extra_bound", 3.0),
            f_kickstart_amp = metadata.get("f_kickstart_amp", 0.0),   # 0 for back-compat
            use_sincos      = metadata.get("use_sincos", False),      # False for back-compat
        )
        # strict=False so old checkpoints without qf_head weights load cleanly;
        # the qf_head will keep its random init (output ~ 0 → gates_Qf ~ 1).
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device).double()
        model.metadata = metadata
        return model


# ──────────────────────────────────────────────────────────────────────────
# Sin/cos variant: encodes joint angles as (sin, cos) pairs
# ──────────────────────────────────────────────────────────────────────────
class LinearizationNetworkSC(nn.Module):
    """Same as LinearizationNetwork but encodes joint angles as (sin, cos).

    Input per frame: [sin(q1), cos(q1), q1d/8, sin(q2), cos(q2), q2d/8] = 6 dims
    Total input: 5 × 6 = 30 dims (vs 20 for the original)

    Benefits:
    - q1=+π and q1=-π map to the SAME input (sin=0, cos=-1)
    - sign(sin(q1)) directly encodes swing direction
    - The network can learn symmetric swing-up with less training data
    - No extra parameters in trunk/heads — only first encoder layer differs

    Batch-mode: forward() accepts (5, state_dim) or (B, 5, state_dim).
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        gate_range_q: float = 0.95,
        gate_range_r: float = 0.20,
        f_extra_bound: float = 3.0,
        f_kickstart_amp: float = 1.0,
        vel_scale: float = 8.0,   # normalisation for joint velocities
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.gate_range_q = gate_range_q
        self.gate_range_r = gate_range_r
        self.f_extra_bound = f_extra_bound
        self.f_kickstart_amp = f_kickstart_amp
        self.vel_scale = vel_scale

        # For a 4-dim state [q1, q1d, q2, q2d]:
        # 2 angles × 2 sincos + 2 velocities = 6 dims per frame
        n_angle_dims = state_dim // 2  # number of joint angles (= 2 for double pendulum)
        sincos_input_dim = 5 * (2 * n_angle_dims + (state_dim - n_angle_dims))
        # = 5 × (2×2 + 2) = 5 × 6 = 30

        enc_dim = hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(sincos_input_dim, enc_dim), nn.Tanh(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        q_out_dim = (horizon - 1) * state_dim
        r_out_dim = horizon * control_dim
        f_out_dim = horizon * control_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, q_out_dim),
        )
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, r_out_dim),
        )
        self.f_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, f_out_dim),
        )

        self._initialize_weights()

        self.metadata = {
            "state_dim": state_dim,
            "control_dim": control_dim,
            "hidden_dim": hidden_dim,
            "horizon": horizon,
            "gate_range_q": gate_range_q,
            "gate_range_r": gate_range_r,
            "f_extra_bound": f_extra_bound,
            "f_kickstart_amp": f_kickstart_amp,
            "vel_scale": vel_scale,
            "architecture": "stage_d_sincos_encoder",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _initialize_weights(self):
        def _init_block(seq: nn.Sequential, final_std: float = 0.01):
            layers = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            for layer in layers[:-1]:
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)
            final = layers[-1]
            nn.init.normal_(final.weight, mean=0.0, std=final_std)
            nn.init.zeros_(final.bias)

        _init_block(self.state_encoder, final_std=0.1)
        _init_block(self.trunk, final_std=0.1)
        _init_block(self.q_head, final_std=0.01)
        _init_block(self.r_head, final_std=0.01)
        _init_block(self.f_head, final_std=0.001)

        if self.f_kickstart_amp > 0.0:
            f_final = list(self.f_head.modules())[-1]
            n_u = self.control_dim
            init_bias = torch.zeros(self.horizon * n_u, dtype=f_final.bias.dtype)
            for k in range(self.horizon):
                phase = math.pi * (k + 0.5) / self.horizon
                init_bias[k * n_u + 0] = self.f_kickstart_amp * math.sin(phase)
            with torch.no_grad():
                f_final.bias.copy_(init_bias)

    def _encode(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """Convert state history to sin/cos encoded input.

        x_sequence: (..., 5, state_dim) with state_dim=4.
        Returns:    (..., 30).

        Vectorised — no Python loop, so this is essentially free even
        for large batches.
        """
        q1  = x_sequence[..., 0]
        q1d = x_sequence[..., 1]
        q2  = x_sequence[..., 2]
        q2d = x_sequence[..., 3]
        frames = torch.stack([
            torch.sin(q1), torch.cos(q1), q1d / self.vel_scale,
            torch.sin(q2), torch.cos(q2), q2d / self.vel_scale,
        ], dim=-1)                                          # (..., 5, 6)
        return frames.reshape(*frames.shape[:-2], -1)       # (..., 30)

    def forward(
        self,
        x_sequence: torch.Tensor,
        q_base_diag: Optional[torch.Tensor] = None,
        r_base_diag: Optional[torch.Tensor] = None,
    ):
        is_batched = (x_sequence.dim() == 3)
        if not is_batched:
            x_sequence = x_sequence.unsqueeze(0)            # (1, 5, state_dim)
        B = x_sequence.shape[0]

        x_enc     = self._encode(x_sequence)                # (B, 30)
        state_emb = self.state_encoder(x_enc)
        features  = self.trunk(state_emb)

        raw_Q   = self.q_head(features).reshape(B, self.horizon - 1, self.state_dim)
        gates_Q = 1.0 + self.gate_range_q * torch.tanh(raw_Q)

        raw_R   = self.r_head(features).reshape(B, self.horizon, self.control_dim)
        gates_R = 1.0 + self.gate_range_r * torch.tanh(raw_R)

        raw_F   = self.f_head(features).reshape(B, self.horizon, self.control_dim)
        f_extra = self.f_extra_bound * torch.tanh(raw_F)

        q_diags = (q_base_diag * gates_Q) if q_base_diag is not None else None
        r_diags = (r_base_diag * gates_R) if r_base_diag is not None else None

        if not is_batched:
            gates_Q = gates_Q.squeeze(0)
            gates_R = gates_R.squeeze(0)
            f_extra = f_extra.squeeze(0)
            if q_diags is not None:
                q_diags = q_diags.squeeze(0)
            if r_diags is not None:
                r_diags = r_diags.squeeze(0)

        # SC variant doesn't have qf_head; return None to match the 6-tuple
        # signature that Simulate.py expects since the qf-head feature was added.
        return gates_Q, gates_R, f_extra, q_diags, r_diags, None

    def save(self, filepath: str, metadata: Optional[Dict] = None):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if metadata:
            self.metadata.update(metadata)
        torch.save(
            {"model_state_dict": self.state_dict(), "metadata": self.metadata},
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        model = cls(
            state_dim       = metadata.get("state_dim", 4),
            control_dim     = metadata.get("control_dim", 2),
            hidden_dim      = metadata.get("hidden_dim", 128),
            horizon         = metadata.get("horizon", 10),
            gate_range_q    = metadata.get("gate_range_q", 0.95),
            gate_range_r    = metadata.get("gate_range_r", 0.20),
            f_extra_bound   = metadata.get("f_extra_bound", 3.0),
            f_kickstart_amp = metadata.get("f_kickstart_amp", 0.0),
            vel_scale       = metadata.get("vel_scale", 8.0),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).double()
        model.metadata = metadata
        return model


# ──────────────────────────────────────────────────────────────────────────
# Separated network: f_net and q_net have independent trunks.
# Eliminates trunk bleed-through: top-episode Q training never affects f_head.
# ──────────────────────────────────────────────────────────────────────────
class SeparatedLinearizationNetwork(nn.Module):
    """Two independent sub-networks sharing only the forward() interface.

    f_net (state_encoder_f → trunk_f → f_head + r_head):
        Trained by energy tracking gradient in bottom episode.
        Gets zero gradient from top episode (detach_f_extra_for_qp=True).

    q_net (state_encoder_q → trunk_q → q_head):
        Trained only by w_q_profile gradients (PUMP in bottom, STABLE in top).
        Gets zero gradient from QP in bottom (detach_gates_Q_for_qp=True).

    Result: no trunk bleed-through. Q@mid stays near PUMP regardless of
    how high Q@top grows, because q_net's trunk_q learns purely from the
    q_profile signal and the two feature spaces never interfere.

    Batch support is inherited transparently from LinearizationNetwork — this
    wrapper just dispatches and returns the picked tuple components.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        gate_range_q: float = 0.99,
        gate_range_r: float = 0.20,
        f_extra_bound: float = 2.0,
        f_kickstart_amp: float = 1.0,
        gate_range_qf: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.gate_range_q = gate_range_q
        self.gate_range_r = gate_range_r
        self.f_extra_bound = f_extra_bound       # accessed by Simulate.py

        # Both sub-nets use goal-centred sin/cos input representation.
        self.f_net = LinearizationNetwork(
            state_dim=state_dim, control_dim=control_dim, horizon=horizon,
            hidden_dim=hidden_dim, gate_range_q=gate_range_q,
            gate_range_r=gate_range_r, f_extra_bound=f_extra_bound,
            f_kickstart_amp=f_kickstart_amp, gate_range_qf=gate_range_qf,
            use_sincos=True,
        )
        self.q_net = LinearizationNetwork(
            state_dim=state_dim, control_dim=control_dim, horizon=horizon,
            hidden_dim=hidden_dim, gate_range_q=gate_range_q,
            gate_range_r=gate_range_r, f_extra_bound=f_extra_bound,
            f_kickstart_amp=0.0,                 # q_net produces Q gates, not f_extra
            gate_range_qf=gate_range_qf,
            use_sincos=True,
        )

        # Expose q_head at top level for apply_q1_bias() compatibility.
        self.q_head = self.q_net.q_head

        self.metadata = {
            "state_dim": state_dim,
            "control_dim": control_dim,
            "hidden_dim": hidden_dim,
            "horizon": horizon,
            "gate_range_q": gate_range_q,
            "gate_range_r": gate_range_r,
            "gate_range_qf": gate_range_qf,
            "f_extra_bound": f_extra_bound,
            "f_kickstart_amp": f_kickstart_amp,
            "use_sincos": True,
            "architecture": "separated_fnet_qnet",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def forward(
        self,
        x_sequence: torch.Tensor,
        q_base_diag: Optional[torch.Tensor] = None,
        r_base_diag: Optional[torch.Tensor] = None,
    ):
        # gates_Q from q_net (only q_profile gradients ever reach q_net.trunk_q)
        gates_Q, _, _, q_diags, _, gates_Qf = self.q_net(x_sequence, q_base_diag, r_base_diag)

        # f_extra and gates_R from f_net (only energy-tracking gradients reach f_net.trunk_f)
        _, gates_R, f_extra, _, r_diags, _ = self.f_net(x_sequence, q_base_diag, r_base_diag)

        return gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf

    def save(self, filepath: str, metadata: Optional[Dict] = None):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if metadata:
            self.metadata.update(metadata)
        torch.save(
            {"model_state_dict": self.state_dict(), "metadata": self.metadata},
            filepath,
        )


# ──────────────────────────────────────────────────────────────────────────
# Recorder & ModelManager (unchanged from previous Stage D version)
# ──────────────────────────────────────────────────────────────────────────
class NetworkOutputRecorder:
    def __init__(self):
        self.epochs: List[Dict] = []
        self._current_epoch_steps: List[Dict] = []

    def start_epoch(self):
        self._current_epoch_steps = []

    def record_step(
        self,
        gates_Q: torch.Tensor,
        gates_R: torch.Tensor,
        f_extra: torch.Tensor,
        q_diags: Optional[torch.Tensor],
        r_diags: Optional[torch.Tensor],
        u_mpc: torch.Tensor,
        state_err: torch.Tensor,
    ):
        with torch.no_grad():
            self._current_epoch_steps.append({
                "gates_Q":   gates_Q.detach().cpu().tolist(),
                "gates_R":   gates_R.detach().cpu().tolist(),
                "f_extra":   f_extra.detach().cpu().tolist(),
                "q_diags":   q_diags.detach().cpu().tolist() if q_diags is not None else None,
                "r_diags":   r_diags.detach().cpu().tolist() if r_diags is not None else None,
                "u_mpc":     u_mpc.detach().cpu().tolist(),
                "state_err": state_err.item() if isinstance(state_err, torch.Tensor) else state_err,
            })

    def end_epoch(self, epoch_loss: float):
        self.epochs.append({
            "steps": self._current_epoch_steps,
            "epoch_loss": epoch_loss,
        })
        self._current_epoch_steps = []

    def epoch_summary(self, epoch_idx: int) -> Dict:
        epoch = self.epochs[epoch_idx]
        steps = epoch["steps"]
        if not steps:
            return {"epoch_loss": epoch["epoch_loss"], "num_steps": 0}
        errs    = [s["state_err"] for s in steps]
        u_norms = [float(torch.tensor(s["u_mpc"]).norm()) for s in steps]
        q_devs  = [float((torch.tensor(s["gates_Q"]) - 1.0).abs().mean()) for s in steps]
        r_devs  = [float((torch.tensor(s["gates_R"]) - 1.0).abs().mean()) for s in steps]
        f_norms = [float(torch.tensor(s["f_extra"]).norm()) for s in steps]
        f_tau1_first = [float(torch.tensor(s["f_extra"])[0, 0]) for s in steps]
        return {
            "epoch_loss":         epoch["epoch_loss"],
            "num_steps":          len(steps),
            "mean_state_err":     float(sum(errs) / len(errs)),
            "mean_u_norm":        float(sum(u_norms) / len(u_norms)),
            "mean_Q_gate_dev":    float(sum(q_devs) / len(q_devs)),
            "mean_R_gate_dev":    float(sum(r_devs) / len(r_devs)),
            "mean_f_extra_norm":  float(sum(f_norms) / len(f_norms)),
            "mean_f_tau1_first": float(sum(f_tau1_first) / len(f_tau1_first)),
        }

    def save_pt(self, filepath: str):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({"recorder": self.epochs}, filepath)


class ModelManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "saved_models"
        )
        os.makedirs(self.base_dir, exist_ok=True)

    def save_training_session(
        self, model, loss_history, training_params,
        session_name=None, recorder=None,
    ):
        session_name = session_name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(self.base_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        model.save(
            os.path.join(session_dir, f"{session_name}.pth"),
            {"training_loss_history": loss_history, "training_params": training_params},
        )
        if recorder:
            recorder.save_pt(os.path.join(session_dir, f"{session_name}_network_outputs.pt"))
        return session_dir


# ──────────────────────────────────────────────────────────────────────────
# Self-test: batched output must equal stacking per-element unbatched calls.
# Run with: python lin_net.py
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    B = 8
    history_len = 5
    state_dim, control_dim, horizon = 4, 2, 10

    # Diverse states (so per-element calls differ).
    x_seq_batch = torch.randn(B, history_len, state_dim, dtype=torch.float64) * 0.5

    q_base = torch.tensor([100.0, 10.0, 100.0, 10.0], dtype=torch.float64)
    r_base = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def check(model_factory, name):
        torch.manual_seed(42)
        model = model_factory().double().eval()
        with torch.no_grad():
            gQ_b, gR_b, fE_b, qd_b, rd_b, gQf_b = model(x_seq_batch, q_base, r_base)
        # Per-element check.
        outs = [model(x_seq_batch[i], q_base, r_base) for i in range(B)]
        gQ_i  = torch.stack([o[0] for o in outs], dim=0)
        gR_i  = torch.stack([o[1] for o in outs], dim=0)
        fE_i  = torch.stack([o[2] for o in outs], dim=0)
        qd_i  = torch.stack([o[3] for o in outs], dim=0) if outs[0][3] is not None else None
        rd_i  = torch.stack([o[4] for o in outs], dim=0) if outs[0][4] is not None else None
        gQf_i = torch.stack([o[5] for o in outs], dim=0) if outs[0][5] is not None else None

        for tag, b, i in [("gates_Q", gQ_b, gQ_i), ("gates_R", gR_b, gR_i),
                          ("f_extra", fE_b, fE_i),
                          ("q_diags", qd_b, qd_i), ("r_diags", rd_b, rd_i),
                          ("gates_Qf", gQf_b, gQf_i)]:
            if b is None and i is None:
                continue
            err = (b - i).abs().max().item()
            print(f"  {name:30s} {tag:10s} max err {err:.3e}")
            assert err < 1e-12, f"{name}/{tag} mismatch: {err}"

    print("LinearizationNetwork (use_sincos=False):")
    check(lambda: LinearizationNetwork(state_dim, control_dim, horizon,
                                       use_sincos=False, gate_range_qf=0.0),
          "LinearizationNetwork(off)")

    print("LinearizationNetwork (use_sincos=True, qf on):")
    check(lambda: LinearizationNetwork(state_dim, control_dim, horizon,
                                       use_sincos=True, gate_range_qf=0.10),
          "LinearizationNetwork(on)")

    print("LinearizationNetworkSC:")
    check(lambda: LinearizationNetworkSC(state_dim, control_dim, horizon),
          "LinearizationNetworkSC")

    print("SeparatedLinearizationNetwork:")
    check(lambda: SeparatedLinearizationNetwork(state_dim, control_dim, horizon,
                                                gate_range_qf=0.10),
          "SeparatedLinearizationNetwork")

    print("OK — batched forward matches stacked per-element to machine precision.")
