"""Simulate_batched.py — batched closed-loop training and rollout for differentiable MPC.

A direct port of Simulate.train_linearization_network() that runs B
trajectories in parallel through one batched MPC.control() call per step
instead of one call per trajectory. Identical loss design, identical
hyperparameters, identical optimizer interface — the only change visible
at the call site is that `x0` (and optionally `x_goal`) are now (B, 4)
tensors instead of (4,) tensors.

The original Simulate.train_linearization_network() and Simulate.rollout()
are NOT modified — they keep working bit-for-bit. This file lives alongside
Simulate.py so you can mix-and-match (e.g. train batched, eval unbatched).

Public entry points:

    train_linearization_network_batched(...)
        One meta-epoch of training over B trajectories simultaneously.
        Returns (loss_history, recorder). Recorder records trajectory[0]
        as the single-trajectory view; the actual training uses all B.

    sample_x0_batch(B, device, mode, ...)
        Helper to generate batched initial conditions matching the
        sampling distributions the existing experiment scripts use.

Loss design (unchanged from Simulate.train_linearization_network):

    track_loss(t) =  (E(next_state) - E(demo[t+1]))² / E_range²      # "energy"
                  |  (cos(q1)-cos(q1_t))² + (sin(q1)-sin(q1_t))²
                     + 0.1 * (q1d - q1d_t)² / 64                      # "cos_q1"
    total_loss   =  W_TRACK * mean over (T, B) of track_loss
                  +  sum of phase_pen_terms (each averaged over B)

Per-step gradient walks back through one true_RK4_disc step (batched),
through the QP solve (cvxpylayers batched implicit-diff), through the
cost-matrix construction (batched), into the network heads. state_detached
between steps prevents BPTT explosion through the Lyapunov-unstable
inverted-pendulum dynamics.
"""

import copy
import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import lin_net as network_module
import mpc_controller


# ──────────────────────────────────────────────────────────────────────────
# Helper: sample batched initial conditions
# ──────────────────────────────────────────────────────────────────────────
def sample_x0_batch(
    B: int,
    device: torch.device,
    mode: str = "rest",
    *,
    base: Optional[torch.Tensor] = None,
    pert_q1:  float = 0.0,
    pert_q1d: float = 0.0,
    pert_q2:  float = 0.0,
    pert_q2d: float = 0.0,
) -> torch.Tensor:
    """Generate B initial conditions in a tensor of shape (B, 4).

    mode="rest"        → all B zeros (good for swing-up curriculum)
    mode="pert_around" → base + uniform pert on each component
                         Requires `base` of shape (4,). For top-of-pendulum
                         recovery: base=[π,0,0,0] with small perts.

    Use mode="pert_around" to replace sample_top() in batched experiments:
        x0 = sample_x0_batch(B, device, "pert_around",
                              base=x_goal, pert_q1=0.3, pert_q1d=0.3,
                              pert_q2=0.05, pert_q2d=0.05)
    """
    if mode == "rest":
        return torch.zeros((B, 4), dtype=torch.float64, device=device)
    if mode == "pert_around":
        if base is None:
            raise ValueError("mode='pert_around' requires base=(4,) tensor")
        base = base.to(device=device, dtype=torch.float64)
        # Uniform in [-pert, +pert] for each component.
        perts = torch.tensor(
            [pert_q1, pert_q1d, pert_q2, pert_q2d],
            device=device, dtype=torch.float64,
        )
        noise = (torch.rand((B, 4), device=device, dtype=torch.float64) * 2 - 1) * perts
        return base.unsqueeze(0).expand(B, -1) + noise
    raise ValueError(f"Unknown mode: {mode!r}")


# ──────────────────────────────────────────────────────────────────────────
# Gradient diagnostics (used by grad_debug=True)
# ──────────────────────────────────────────────────────────────────────────
def _gradient_stats(lin_net: nn.Module) -> dict:
    """Per-module gradient L2 norms. Cheap when called once per epoch."""
    tracked = ["state_encoder", "trunk", "q_head", "r_head", "f_head"]
    module_sq = {k: 0.0 for k in tracked}
    total_sq = 0.0
    missing = []
    for name, param in lin_net.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        g2 = float(param.grad.detach().pow(2).sum().item())
        total_sq += g2
        for prefix in tracked:
            if name.startswith(prefix):
                module_sq[prefix] += g2
                break
    return {
        "total_norm":    math.sqrt(max(total_sq, 0.0)),
        "module_norms":  {k: math.sqrt(max(v, 0.0)) for k, v in module_sq.items()},
        "missing_count": len(missing),
        "missing_names": missing,
    }


# ──────────────────────────────────────────────────────────────────────────
# Helper to broadcast x_goal to (B, 4) if given as (4,)
# ──────────────────────────────────────────────────────────────────────────
def _broadcast_goal(x_goal: torch.Tensor, B: int) -> torch.Tensor:
    if x_goal.dim() == 1:
        return x_goal.unsqueeze(0).expand(B, -1).contiguous()
    if x_goal.dim() == 2 and x_goal.shape[0] == B:
        return x_goal
    raise ValueError(
        f"x_goal shape {tuple(x_goal.shape)} incompatible with batch size B={B}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Main training loop (batched)
# ──────────────────────────────────────────────────────────────────────────
def train_linearization_network_batched(
    lin_net: nn.Module,
    mpc: mpc_controller.MPC_controller,
    x0: torch.Tensor,                        # (B, 4) — batched ICs (required)
    x_goal: torch.Tensor,                    # (4,) broadcast OR (B, 4)
    demo: torch.Tensor,                      # (T, 4) — shared across batch
    num_steps: int,
    num_epochs: int = 30,
    lr: float = 1e-4,
    debug_monitor = None,
    recorder: Optional[network_module.NetworkOutputRecorder] = None,
    grad_debug: bool = False,
    grad_debug_every: int = 1,
    track_mode: str = "energy",              # "energy" or "cos_q1"

    # ── Q-gate profile target ─────────────────────────────────────────────
    w_q_profile: float = 0.0,
    q_profile_pump:   Optional[List[float]] = None,
    q_profile_stable: Optional[List[float]] = None,
    q_profile_state_phase: bool = False,
    q_profile_near_pi_power: float = 1.0,

    # ── Selective gradient detachment ─────────────────────────────────────
    detach_gates_Q_for_qp:  bool = False,
    detach_f_extra_for_qp:  bool = False,

    # ── f_extra regularisers ──────────────────────────────────────────────
    w_f_end_reg: float = 0.0,
    f_end_reg_steps: int = 20,
    w_f_pos_only: float = 0.0,

    # ── Hard ZeroFNet gate ────────────────────────────────────────────────
    f_gate_thresh: float = 0.0,

    # ── Stable-phase direct position tracking ─────────────────────────────
    w_stable_phase: float = 0.0,
    stable_phase_steps: int = 30,

    # ── Observation noise injection (data augmentation) ──────────────────
    train_noise_sigma: Optional[List[float]] = None,

    # ── Optimisation ──────────────────────────────────────────────────────
    early_stop_patience: int = 15,
    external_optimizer: Optional[torch.optim.Optimizer] = None,
    restore_best: bool = True,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:
    """Batched version of Simulate.train_linearization_network.

    See module docstring for design notes. The only signature differences
    versus the unbatched version are that x0 must be (B, 4) and x_goal
    may be (4,) (broadcast) or (B, 4).

    Loss magnitudes are batch-normalised (each summed loss term is
    divided by both T and B) so hyperparameters tuned for B=1 transfer
    to higher B without LR rescaling.
    """
    # ── Loss weights ──────────────────────────────────────────────────────
    W_TRACK = 5.0
    STEP_LOSS_CLAMP = 200.0
    SKIP_UPDATE_GRAD_NORM = 5e7
    CLIP_OTHER = 2.0

    n_u = mpc.MPC_dynamics.u_min.shape[0]
    demo_T = demo.shape[0]

    if x0.dim() != 2 or x0.shape[-1] != 4:
        raise ValueError(
            f"x0 must be (B, 4); got shape {tuple(x0.shape)}. "
            f"Use Simulate.train_linearization_network for unbatched."
        )
    B = x0.shape[0]
    x_goal_b = _broadcast_goal(x_goal, B)        # (B, 4)

    # Q-profile target tensors
    if q_profile_pump is None:
        q_profile_pump = [0.01, 1.0, 1.0, 1.0]
    if q_profile_stable is None:
        q_profile_stable = [1.0, 1.0, 1.0, 1.0]
    q_profile_pump_t = torch.tensor(q_profile_pump,
                                    device=mpc.device, dtype=torch.float64)
    q_profile_stable_t = torch.tensor(q_profile_stable,
                                      device=mpc.device, dtype=torch.float64)

    # Precompute demo's energy curve once (shared across batch).
    with torch.no_grad():
        E_demo = torch.stack(
            [mpc.compute_energy_single(demo[i]) for i in range(demo_T)]
        )                                          # (T,)
        E_range = (E_demo.max() - E_demo.min()).clamp(min=1.0)

    if external_optimizer is not None:
        optimizer = external_optimizer
    else:
        optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=max(num_epochs, 1),
    )

    EARLY_STOP_PATIENCE = early_stop_patience
    epochs_since_improvement = 0
    loss_history: List[float] = []
    best_goal_dist = float("inf")
    best_state_dict = None

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    # Build noise tensor for observation noise injection (broadcasts to (B, 4)).
    if train_noise_sigma is not None and any(s > 0 for s in train_noise_sigma):
        train_noise_tensor = torch.tensor(
            train_noise_sigma, device=mpc.device, dtype=torch.float64,
        )
    else:
        train_noise_tensor = None

    def add_train_noise(state):
        # state: (B, 4); noise broadcasts over B.
        if train_noise_tensor is None:
            return state.clone()
        return state + torch.randn_like(state) * train_noise_tensor

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, "qp_fallback_count", 0))

        current_state_detached = x0.detach().clone()                # (B, 4)
        state_history = [
            add_train_noise(current_state_detached).detach()
            for _ in range(5)
        ]                                                            # list of (B, 4)

        # u_seq_guess: per-trajectory horizon control guess.
        u_seq_guess = torch.zeros(
            (B, mpc.N, n_u), device=mpc.device, dtype=torch.float64,
        )

        track_step_terms: List[torch.Tensor] = []   # each element (B,)
        phase_pen_terms:  List[torch.Tensor] = []   # each element scalar (already meaned)

        for step in range(num_steps):
            # state_history is list of 5 tensors of (B, 4). Stack along dim=-2
            # → (B, 5, 4), matching the batched lin_net.forward expected shape.
            state_history_seq = torch.stack(state_history, dim=-2)

            gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = lin_net(
                state_history_seq,
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )
            # Shapes:
            #   gates_Q  : (B, N-1, 4)
            #   gates_R  : (B, N,   2)
            #   f_extra  : (B, N,   2)
            #   gates_Qf : (B, 4) or None

            # ── Hard ZeroFNet gate (per-trajectory) ───────────────────────
            # near_pi computed per trajectory; gate broadcasts over horizon.
            if f_gate_thresh > 0.0:
                _q1_t   = current_state_detached[..., 0]                 # (B,)
                _near_pi = (1.0 + torch.cos(_q1_t - x_goal_b[..., 0])) / 2.0
                _zf_gate = (
                    (_near_pi - f_gate_thresh)
                    / max(1e-8, 1.0 - f_gate_thresh)
                ).clamp(0.0, 1.0)                                         # (B,)
                # Reshape gate to (B, 1, 1) so it broadcasts across (N, 2).
                f_extra = f_extra * (1.0 - _zf_gate.detach()).view(B, 1, 1)

            # ── Q-gate profile penalty ────────────────────────────────────
            if w_q_profile > 0.0:
                if q_profile_state_phase:
                    near_goal = (
                        1.0 + torch.cos(current_state_detached[..., 0]
                                        - x_goal_b[..., 0])
                    ) / 2.0
                    near_goal = torch.clamp(near_goal, 0.0, 1.0)         # (B,)
                    if q_profile_near_pi_power != 1.0:
                        near_goal = near_goal ** q_profile_near_pi_power
                    # Blend per-trajectory: (B,1) * (1,4) → (B,4)
                    target = (
                        (1.0 - near_goal).unsqueeze(-1) * q_profile_pump_t
                        + near_goal.unsqueeze(-1)       * q_profile_stable_t
                    )                                                     # (B, 4)
                    # Broadcast against gates_Q (B, N-1, 4): target (B, 1, 4)
                    profile_dev = ((gates_Q - target.unsqueeze(-2)) ** 2).mean()
                else:
                    target = (q_profile_pump_t if step < num_steps // 2
                              else q_profile_stable_t)                    # (4,)
                    profile_dev = ((gates_Q - target) ** 2).mean()
                phase_pen_terms.append(w_q_profile * profile_dev)

            # ── f_extra end-phase L2 penalty ──────────────────────────────
            if w_f_end_reg > 0.0 and step >= num_steps - f_end_reg_steps:
                f_reg = w_f_end_reg * (f_extra ** 2).mean()
                phase_pen_terms.append(f_reg)

            # ── f_extra position-conditional penalty ──────────────────────
            if w_f_pos_only > 0.0:
                q1_d = current_state_detached[..., 0]                     # (B,)
                near_goal_pos = (1.0 + torch.cos(q1_d - x_goal_b[..., 0])) / 2.0
                near_goal_pos = torch.clamp(near_goal_pos, 0.0, 1.0)
                # Mean (f_extra²) over (N, 2); weight by per-trajectory scalar.
                fe_sq = (f_extra ** 2).mean(dim=(-2, -1))                # (B,)
                f_pos_pen = w_f_pos_only * (near_goal_pos * fe_sq).mean()
                phase_pen_terms.append(f_pos_pen)

            # ── Build linearisation buffers and call the QP ───────────────
            # x_lin_seq: (B, N, 4) — broadcast current state along horizon.
            x_lin_seq = current_state_detached.unsqueeze(-2)              # (B, 1, 4)
            x_lin_seq = x_lin_seq.expand(B, mpc.N, 4).clone()             # (B, N, 4)

            u_lin_seq = torch.clamp(
                u_seq_guess.clone(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),                   # (1, 2) broadcasts
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )                                                              # (B, N, 2)

            # Selective detachment for the QP call.
            f_extra_qp = f_extra.detach() if detach_f_extra_for_qp else f_extra
            # extra_linear_control expects (B, N*nu) flat across the horizon.
            extra_ctrl = f_extra_qp.flatten(start_dim=-2)                  # (B, N*nu)
            gates_Q_qp = gates_Q.detach() if detach_gates_Q_for_qp else gates_Q

            u_mpc, U_opt_full = mpc.control(
                current_state_detached, x_lin_seq, u_lin_seq, x_goal_b,
                diag_corrections_Q=gates_Q_qp,
                diag_corrections_R=gates_R,
                extra_linear_control=extra_ctrl,
                diag_corrections_Qf=gates_Qf,
            )
            # u_mpc: (B, n_u); U_opt_full: (B, N*n_u)

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)
            # (B, 4)

            # ── Tracking term ─────────────────────────────────────────────
            target_idx = min(step + 1, demo_T - 1)
            target = demo[target_idx]                                     # (4,)

            if track_mode == "energy":
                E_now = mpc.compute_energy_single(next_state)             # (B,)
                track_step = ((E_now - E_demo[target_idx]) / E_range) ** 2  # (B,)
            elif track_mode == "cos_q1":
                q1, q1d = next_state[..., 0], next_state[..., 1]          # (B,), (B,)
                q1_t, q1d_t = target[0], target[1]                        # scalar, scalar
                angle_err = (
                    (torch.cos(q1) - torch.cos(q1_t)) ** 2
                    + (torch.sin(q1) - torch.sin(q1_t)) ** 2
                )                                                          # (B,)
                vel_err = (q1d - q1d_t) ** 2 / 64.0                       # (B,)
                track_step = angle_err + 0.1 * vel_err                    # (B,)
            else:
                raise ValueError(
                    f"track_mode must be 'energy' or 'cos_q1', got {track_mode!r}"
                )

            track_step = torch.clamp(track_step, max=STEP_LOSS_CLAMP)
            track_step_terms.append(track_step)

            # ── Stable-phase direct position-to-goal loss ────────────────
            if w_stable_phase > 0.0 and step >= num_steps - stable_phase_steps:
                q1s,  q1ds = next_state[..., 0], next_state[..., 1]       # (B,)
                q2s,  q2ds = next_state[..., 2], next_state[..., 3]
                q1_err_s = torch.atan2(
                    torch.sin(q1s - x_goal_b[..., 0]),
                    torch.cos(q1s - x_goal_b[..., 0]),
                )
                stable_loss = w_stable_phase * (
                    q1_err_s ** 2
                    + (q1ds / 8.0) ** 2
                    + (q2s / math.pi) ** 2
                    + (q2ds / 8.0) ** 2
                )                                                          # (B,)
                phase_pen_terms.append(stable_loss.mean())

            # Record trajectory 0 (representative single-trajectory view).
            # All B trajectories still drive the training.
            with torch.no_grad():
                recorder.record_step(
                    gates_Q=gates_Q[0], gates_R=gates_R[0],
                    f_extra=f_extra[0],
                    q_diags=(q_diags[0] if q_diags is not None else None),
                    r_diags=(r_diags[0] if r_diags is not None else None),
                    u_mpc=u_mpc[0],
                    state_err=((next_state[0].detach() - x_goal_b[0]) ** 2).sum(),
                )

            current_state_detached = next_state.detach()

            # Roll u_seq_guess forward by one step (per trajectory).
            # U_opt_full: (B, N*n_u) → reshape to (B, N, n_u) → drop first, repeat last.
            U_opt_reshaped = U_opt_full.detach().view(B, mpc.N, n_u)
            u_seq_guess = torch.cat(
                [U_opt_reshaped[:, 1:], U_opt_reshaped[:, -1:]], dim=1,
            ).clone()

            state_history.pop(0)
            state_history.append(
                add_train_noise(current_state_detached).detach().clone()
            )

        # ── Combine losses ───────────────────────────────────────────────
        # track_step_terms: list of (B,) tensors, length num_steps.
        # Stack → (T, B), then mean over both T and B.
        track_loss = torch.stack(track_step_terms).mean()
        phase_pen_loss = (
            torch.stack(phase_pen_terms).mean()
            if phase_pen_terms else
            torch.tensor(0.0, device=mpc.device, dtype=torch.float64)
        )
        total_loss = W_TRACK * track_loss + phase_pen_loss
        loss_history.append(total_loss.item())
        recorder.end_epoch(total_loss.item())

        total_loss.backward()
        grad_stats = None
        if grad_debug and ((epoch + 1) % max(1, grad_debug_every) == 0 or epoch == 0):
            grad_stats = _gradient_stats(lin_net)

        with torch.no_grad():
            # Per-trajectory L2 distance to goal, averaged over batch.
            goal_dist_per = torch.norm(current_state_detached - x_goal_b, dim=-1)  # (B,)
            goal_dist = goal_dist_per.mean().item()

        if goal_dist < best_goal_dist:
            best_goal_dist = goal_dist
            best_state_dict = copy.deepcopy(lin_net.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if not torch.isfinite(total_loss):
            optimizer.zero_grad()
        else:
            is_bad = any(
                (not torch.isfinite(p.grad).all())
                for _, p in lin_net.named_parameters()
                if p.grad is not None
            )
            if is_bad:
                optimizer.zero_grad()
            elif (grad_stats is not None
                  and grad_stats["total_norm"] > SKIP_UPDATE_GRAD_NORM):
                optimizer.zero_grad()
            else:
                params = [p for p in lin_net.parameters() if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=CLIP_OTHER)
                optimizer.step()

        scheduler.step(epoch + 1)

        if debug_monitor:
            with torch.no_grad():
                summary = recorder.epoch_summary(epoch)
            qp_fallbacks_epoch = (
                int(getattr(mpc, "qp_fallback_count", 0)) - qp_fallback_start
            )
            debug_monitor.log_epoch(epoch, num_epochs, total_loss.item(), {
                "epoch_time":         time.time() - epoch_start_time,
                "learning_rate":      optimizer.param_groups[0]["lr"],
                "loss_track":         track_loss.item(),
                "qp_fallbacks":       qp_fallbacks_epoch,
                "pure_end_error":     goal_dist,
                "batch_size":         B,
                "mean_Q_gate_dev":    summary.get("mean_Q_gate_dev",    float("nan")),
                "mean_f_extra_norm":  summary.get("mean_f_extra_norm",  float("nan")),
                "mean_f_tau1_first": summary.get("mean_f_tau1_first", float("nan")),
            })

        if grad_stats is not None:
            mn = grad_stats["module_norms"]
            print(
                "  GradFlow | "
                f"tot={grad_stats['total_norm']:.3e}  "
                f"trunk={mn['trunk']:.3e}  "
                f"q={mn['q_head']:.3e}  "
                f"r={mn['r_head']:.3e}  "
                f"f={mn['f_head']:.3e}  "
                f"missing={grad_stats['missing_count']}"
            )

        if (
            epochs_since_improvement >= EARLY_STOP_PATIENCE
            and best_goal_dist < 1.0
        ):
            print(
                f"  EarlyStop after epoch {epoch+1}: "
                f"best_goal_dist={best_goal_dist:.4f} hasn't improved "
                f"for {epochs_since_improvement} epochs."
            )
            break

    if restore_best and best_state_dict is not None:
        lin_net.load_state_dict(best_state_dict)

    return loss_history, recorder
