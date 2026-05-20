"""Simulate.py — closed-loop training and rollout for differentiable MPC.

Two public entry points used by the active hardware experiments
(exp_hardware_v{1, 2_continue, 2_noiserobust, 3, 4, 5, 6}):

  train_linearization_network(...)
      One meta-epoch of training: rolls out for `num_steps` under closed-loop
      MPC, computes a tracking loss against `demo`, and steps the optimizer.
      Returns (loss_history, recorder).

  rollout(lin_net, mpc, x0, x_goal, num_steps)
      Plain closed-loop simulation with no learning, used by eval2k.

Loss design (the only loss form the active scripts use):

    track_loss(t) = (E(next_state) - E(demo[t+1]))² / E_range²    # track_mode="energy"
                  | (cos(q1)-cos(q1_t))² + (sin(q1)-sin(q1_t))²
                       + 0.1 * (q1d - q1d_t)² / 64                # track_mode="cos_q1"

    total_loss = W_TRACK * mean(track_loss)
               + sum(phase_pen_terms)        # f_end_reg, q_profile, f_pos_only,
                                             # stable_phase — see callsites

The per-step gradient walks back through one true_RK4_disc step, through the
QP solve (cvxpylayers implicit-diff), through the cost-matrix construction,
into the network heads.  state_detached between steps prevents BPTT explosion
through the Lyapunov-unstable inverted-pendulum dynamics.
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
# Gradient diagnostics (used by grad_debug=True)
# ──────────────────────────────────────────────────────────────────────────
def _gradient_stats(lin_net: nn.Module) -> dict:
    """Per-module gradient L2 norms.  Cheap when called once per epoch."""
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
# Main training loop
# ──────────────────────────────────────────────────────────────────────────
def train_linearization_network(
    lin_net:    nn.Module,
    mpc:        mpc_controller.MPC_controller,
    x0:         torch.Tensor,
    x_goal:     torch.Tensor,
    demo:       torch.Tensor,
    num_steps:  int,
    num_epochs: int   = 30,
    lr:         float = 1e-4,
    debug_monitor       = None,
    recorder:           Optional[network_module.NetworkOutputRecorder] = None,
    grad_debug:         bool  = False,
    grad_debug_every:   int   = 1,
    track_mode: str     = "energy",   # "energy" or "cos_q1"
    # ── Q-gate profile target ─────────────────────────────────────────────
    # Penalise gates_Q for deviating from a per-dim target.  When
    # state_phase=True, the target blends pump↔stable from cos(q1−q1_goal),
    # giving a smooth state-conditional supervision signal independent of
    # whatever the rollout produces (clean signal for q_net).
    w_q_profile:        float = 0.0,
    q_profile_pump:     Optional[List[float]] = None,    # default [0.01,1,1,1]
    q_profile_stable:   Optional[List[float]] = None,    # default [1,1,1,1]
    q_profile_state_phase: bool = False,
    q_profile_near_pi_power: float = 1.0,
    # ── Selective gradient detachment ─────────────────────────────────────
    # detach_gates_Q_for_qp: pass detached gates_Q to the QP so the
    # tracking-loss gradient does NOT flow back to q_head.  q_head is then
    # trained only by the q_profile penalty (clean state-conditional signal).
    detach_gates_Q_for_qp: bool = False,
    # detach_f_extra_for_qp: same idea on the other side — used in top
    # episodes to keep f_head's training pure to swing-up.
    detach_f_extra_for_qp: bool = False,
    # ── f_extra regularisers ──────────────────────────────────────────────
    # End-phase L2 penalty: in the last `f_end_reg_steps` steps, penalise
    # ‖f_extra‖² to keep the controller from issuing large pumping torques
    # near the goal (an "overlearned pumping" failure mode).
    w_f_end_reg:          float = 0.0,
    f_end_reg_steps:      int   = 20,
    # Position-only stable penalty: penalise ‖f_extra‖² weighted by
    # (1+cos(q1-q1_goal))/2 — fires whenever near upright, regardless of
    # velocity.  Brakes oscillation through the goal.
    w_f_pos_only:         float = 0.0,
    # ── Hard ZeroFNet gate ────────────────────────────────────────────────
    # When near_pi > f_gate_thresh, multiply f_extra by a soft ramp from 1
    # (away from goal) to 0 (at goal).  Detached gate → no gradient to
    # f_head near the top.  f_gate_thresh=0 disables.
    f_gate_thresh:        float = 0.0,
    # ── Stable-phase direct position tracking ─────────────────────────────
    # In the last `stable_phase_steps` steps, add a wrapped-q1 + normalised-
    # velocity loss that drives the state directly to the goal.  Stronger
    # than energy-only tracking for the final hold (energy can be correct
    # while still swinging through the goal).
    w_stable_phase:       float = 0.0,
    stable_phase_steps:   int   = 30,
    # ── Observation noise injection (data augmentation) ──────────────────
    # Add Gaussian noise to the state observations the network sees.
    # Pass 4 σ values (one per state dim).  None → no noise.
    train_noise_sigma:    Optional[List[float]] = None,
    # ── Optimisation ──────────────────────────────────────────────────────
    early_stop_patience:  int  = 15,
    # External optimizer is REQUIRED for curriculum-style use (num_epochs=1
    # called many times).  Without it, AdamW's momentum is reset each call
    # and no cumulative learning happens.
    external_optimizer:   Optional[torch.optim.Optimizer] = None,
    # restore_best=False is REQUIRED for curriculum-style use, otherwise
    # best_state_dict captured BEFORE the optimiser step rolls each call back.
    restore_best:         bool = True,
) -> Tuple[List[float], network_module.NetworkOutputRecorder]:

    # ── Loss weights ──────────────────────────────────────────────────────
    W_TRACK         = 5.0
    STEP_LOSS_CLAMP = 200.0

    SKIP_UPDATE_GRAD_NORM = 5e7
    CLIP_OTHER = 2.0

    n_u = mpc.MPC_dynamics.u_min.shape[0]
    demo_T = demo.shape[0]

    # Q-profile target tensors
    if q_profile_pump is None:
        q_profile_pump = [0.01, 1.0, 1.0, 1.0]
    if q_profile_stable is None:
        q_profile_stable = [1.0, 1.0, 1.0, 1.0]
    q_profile_pump_t   = torch.tensor(q_profile_pump,   device=mpc.device, dtype=torch.float64)
    q_profile_stable_t = torch.tensor(q_profile_stable, device=mpc.device, dtype=torch.float64)

    # Precompute demo's energy curve once (for track_mode == "energy").
    with torch.no_grad():
        E_demo = torch.stack([mpc.compute_energy_single(demo[i]) for i in range(demo_T)])
    # Energy is in (Joules)² which differs numerically from (rad)² in state
    # tracking — rescale by 1/E_range² so gradient magnitude is comparable.
    E_range = (E_demo.max() - E_demo.min()).clamp(min=1.0)

    if external_optimizer is not None:
        optimizer = external_optimizer
    else:
        optimizer = torch.optim.AdamW(lin_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=max(num_epochs, 1),
    )

    # Early-stop on best_goal_dist plateau.
    EARLY_STOP_PATIENCE = early_stop_patience
    epochs_since_improvement = 0

    loss_history    = []
    best_goal_dist  = float("inf")
    best_state_dict = None

    if recorder is None:
        recorder = network_module.NetworkOutputRecorder()

    # Build noise tensor for observation noise injection.
    if train_noise_sigma is not None and any(s > 0 for s in train_noise_sigma):
        train_noise_tensor = torch.tensor(
            train_noise_sigma, device=mpc.device, dtype=torch.float64,
        )
    else:
        train_noise_tensor = None

    def add_train_noise(state):
        if train_noise_tensor is None:
            return state.clone()
        return state + torch.randn_like(state) * train_noise_tensor

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        lin_net.train()
        optimizer.zero_grad()
        recorder.start_epoch()
        qp_fallback_start = int(getattr(mpc, "qp_fallback_count", 0))

        current_state_detached = x0.detach().clone()
        state_history = [add_train_noise(current_state_detached).detach()
                         for _ in range(5)]
        u_seq_guess = torch.zeros((mpc.N, n_u), device=mpc.device, dtype=torch.float64)

        track_step_terms = []
        phase_pen_terms  = []

        for step in range(num_steps):
            state_history_seq = torch.stack(state_history, dim=0)

            gates_Q, gates_R, f_extra, q_diags, r_diags, gates_Qf = lin_net(
                state_history_seq,
                q_base_diag=mpc.q_base_diag,
                r_base_diag=mpc.r_base_diag,
            )

            # Hard ZeroFNet gate: zero f_extra when near_pi > f_gate_thresh.
            # No gradient to f_head (detached gate); Q/R heads still adapt.
            if f_gate_thresh > 0.0:
                _q1_t = current_state_detached[0]
                _near_pi = (1.0 + torch.cos(_q1_t - x_goal[0])) / 2.0
                _zf_gate = (
                    (_near_pi - f_gate_thresh) / max(1e-8, 1.0 - f_gate_thresh)
                ).clamp(0.0, 1.0)
                f_extra = f_extra * (1.0 - _zf_gate.detach())

            # Q-gate profile penalty.
            if w_q_profile > 0.0:
                if q_profile_state_phase:
                    near_goal = (1.0 + torch.cos(current_state_detached[0] - x_goal[0])) / 2.0
                    near_goal = torch.clamp(near_goal, 0.0, 1.0)
                    if q_profile_near_pi_power != 1.0:
                        near_goal = near_goal ** q_profile_near_pi_power
                    target = (1.0 - near_goal) * q_profile_pump_t \
                            + near_goal * q_profile_stable_t
                else:
                    target = q_profile_pump_t if step < num_steps // 2 \
                                              else q_profile_stable_t
                profile_dev = ((gates_Q - target.unsqueeze(0)) ** 2).mean()
                phase_pen_terms.append(w_q_profile * profile_dev)

            # f_extra end-phase L2 penalty.
            if w_f_end_reg > 0.0 and step >= num_steps - f_end_reg_steps:
                f_reg = w_f_end_reg * (f_extra ** 2).mean()
                phase_pen_terms.append(f_reg)

            # f_extra position-conditional penalty (fires near upright,
            # any velocity).
            if w_f_pos_only > 0.0:
                q1_d  = current_state_detached[0]
                near_goal_pos = (1.0 + torch.cos(q1_d - x_goal[0])) / 2.0
                near_goal_pos = torch.clamp(near_goal_pos, 0.0, 1.0)
                f_pos_pen = w_f_pos_only * near_goal_pos * (f_extra ** 2).mean()
                phase_pen_terms.append(f_pos_pen)

            # Build linearisation buffers and call the QP.
            x_lin_seq = current_state_detached.unsqueeze(0).expand(mpc.N, -1).clone()
            u_lin_seq = torch.clamp(
                u_seq_guess.clone(),
                min=mpc.MPC_dynamics.u_min.unsqueeze(0),
                max=mpc.MPC_dynamics.u_max.unsqueeze(0),
            )

            # Selective detachment for the QP call.
            f_extra_qp = f_extra.detach() if detach_f_extra_for_qp else f_extra
            extra_ctrl = f_extra_qp.reshape(-1)
            gates_Q_qp = gates_Q.detach() if detach_gates_Q_for_qp else gates_Q

            u_mpc, U_opt_full = mpc.control(
                current_state_detached, x_lin_seq, u_lin_seq, x_goal,
                diag_corrections_Q=gates_Q_qp,
                diag_corrections_R=gates_R,
                extra_linear_control=extra_ctrl,
                diag_corrections_Qf=gates_Qf,
            )

            next_state = mpc.true_RK4_disc(current_state_detached, u_mpc, mpc.dt)

            # Tracking term.
            target_idx = min(step + 1, demo_T - 1)
            target = demo[target_idx]
            if track_mode == "energy":
                E_now = mpc.compute_energy_single(next_state)
                track_step = ((E_now - E_demo[target_idx]) / E_range) ** 2
            elif track_mode == "cos_q1":
                # Wrapped q1-angle tracking — bounded [0, 4.2], unique
                # minimum at the demo state (breaks the spinning degeneracy
                # of pure energy tracking).
                q1, q1d = next_state[0], next_state[1]
                q1_t, q1d_t = target[0], target[1]
                angle_err = (torch.cos(q1) - torch.cos(q1_t))**2 \
                          + (torch.sin(q1) - torch.sin(q1_t))**2
                vel_err = (q1d - q1d_t)**2 / 64.0
                track_step = angle_err + 0.1 * vel_err
            else:
                raise ValueError(
                    f"track_mode must be 'energy' or 'cos_q1', got {track_mode!r}"
                )
            track_step = torch.clamp(track_step, max=STEP_LOSS_CLAMP)
            track_step_terms.append(track_step)

            # Stable-phase direct position-to-goal loss for the final
            # `stable_phase_steps` steps.
            if w_stable_phase > 0.0 and step >= num_steps - stable_phase_steps:
                q1s, q1ds = next_state[0], next_state[1]
                q2s, q2ds = next_state[2], next_state[3]
                q1_err_s = torch.atan2(
                    torch.sin(q1s - x_goal[0]),
                    torch.cos(q1s - x_goal[0]),
                )
                stable_loss = w_stable_phase * (
                    q1_err_s ** 2
                    + (q1ds / 8.0) ** 2
                    + (q2s / math.pi) ** 2
                    + (q2ds / 8.0) ** 2
                )
                phase_pen_terms.append(stable_loss)

            recorder.record_step(
                gates_Q=gates_Q, gates_R=gates_R, f_extra=f_extra,
                q_diags=q_diags, r_diags=r_diags,
                u_mpc=u_mpc,
                state_err=((next_state.detach() - x_goal) ** 2).sum(),
            )

            current_state_detached = next_state.detach()
            U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
            u_seq_guess = torch.cat(
                [U_opt_reshaped[1:], U_opt_reshaped[-1:]], dim=0,
            ).clone()
            state_history.pop(0)
            state_history.append(add_train_noise(current_state_detached).detach().clone())

        # ── Combine ──────────────────────────────────────────────────────
        track_loss = torch.stack(track_step_terms).sum() / num_steps
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
            goal_dist = torch.norm(current_state_detached - x_goal).item()
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
            elif grad_stats is not None and grad_stats["total_norm"] > SKIP_UPDATE_GRAD_NORM:
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
                "mean_Q_gate_dev":    summary.get("mean_Q_gate_dev",    float("nan")),
                "mean_f_extra_norm":  summary.get("mean_f_extra_norm",  float("nan")),
                "mean_f_tau1_first":  summary.get("mean_f_tau1_first",  float("nan")),
            })

        if grad_stats is not None:
            mn = grad_stats["module_norms"]
            print(
                "      GradFlow | "
                f"tot={grad_stats['total_norm']:.3e} "
                f"trunk={mn['trunk']:.3e} "
                f"q={mn['q_head']:.3e} "
                f"r={mn['r_head']:.3e} "
                f"f={mn['f_head']:.3e} "
                f"missing={grad_stats['missing_count']}"
            )

        if (
            epochs_since_improvement >= EARLY_STOP_PATIENCE
            and best_goal_dist < 1.0
        ):
            print(
                f"      EarlyStop after epoch {epoch+1}: "
                f"best_goal_dist={best_goal_dist:.4f} hasn't improved "
                f"for {epochs_since_improvement} epochs."
            )
            break

    if restore_best and best_state_dict is not None:
        lin_net.load_state_dict(best_state_dict)

    return loss_history, recorder


# ──────────────────────────────────────────────────────────────────────────
# Closed-loop rollout (no learning)
# ──────────────────────────────────────────────────────────────────────────
def rollout(
    lin_net,
    mpc:       mpc_controller.MPC_controller,
    x0:        torch.Tensor,
    x_goal:    torch.Tensor,
    num_steps: int,
    f_gate_thresh: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the trained network + MPC for `num_steps` steps, no gradients.

    Returns (x_hist[num_steps+1, 4], u_hist[num_steps, 2]).  Used by eval2k
    in every experiment script and by hardware sim/eval.
    """
    n_x = x0.shape[0]
    n_u = mpc.MPC_dynamics.u_min.shape[0]

    x_hist = torch.zeros(num_steps + 1, n_x, dtype=torch.float64, device=mpc.device)
    u_hist = torch.zeros(num_steps,     n_u, dtype=torch.float64, device=mpc.device)

    x = x0.clone().to(mpc.device)
    x_hist[0] = x
    u_seq_guess = torch.zeros((mpc.N, n_u), dtype=torch.float64, device=mpc.device)

    # Warm-start the QP with a gravity-direction torque if x0 isn't at rest.
    init_q1 = float(x[0].item())
    if abs(init_q1) > 0.01:
        gravity_torque = 2.0 * 9.81 * 0.5 * abs(math.sin(init_q1))
        wrapped_err = math.atan2(
            math.sin(float(x_goal[0].item()) - init_q1),
            math.cos(float(x_goal[0].item()) - init_q1),
        )
        goal_sign = 1.0 if wrapped_err > 0 else -1.0
        seed_tau1 = goal_sign * min(
            float(mpc.MPC_dynamics.u_max[0].item()),
            gravity_torque * 2.0,
        )
        u_seq_guess[:, 0] = seed_tau1

    state_history = [x.clone() for _ in range(5)]

    if lin_net is not None:
        lin_net.eval()

    for step in range(num_steps):
        with torch.no_grad():
            if lin_net is not None:
                gates_Q, gates_R, f_extra, _, _, gates_Qf = lin_net(
                    torch.stack(state_history, dim=0),
                    q_base_diag=mpc.q_base_diag,
                    r_base_diag=mpc.r_base_diag,
                )
            else:
                gates_Q, gates_R, f_extra, gates_Qf = None, None, None, None

        x_lin_seq = x.unsqueeze(0).expand(mpc.N, -1).clone()
        u_lin_seq = torch.clamp(
            u_seq_guess.clone(),
            min=mpc.MPC_dynamics.u_min.unsqueeze(0),
            max=mpc.MPC_dynamics.u_max.unsqueeze(0),
        )

        # Optional ZeroFNet gate at inference.
        if f_gate_thresh > 0.0 and f_extra is not None:
            _near_pi = (1.0 + torch.cos(x[0] - x_goal[0])) / 2.0
            _zf = (
                (_near_pi - f_gate_thresh) / max(1e-8, 1.0 - f_gate_thresh)
            ).clamp(0.0, 1.0)
            f_extra = f_extra * (1.0 - _zf)

        extra_ctrl = f_extra.reshape(-1) if f_extra is not None else None

        u_opt, U_opt_full = mpc.control(
            x, x_lin_seq, u_lin_seq, x_goal,
            diag_corrections_Q=gates_Q,
            diag_corrections_R=gates_R,
            extra_linear_control=extra_ctrl,
            diag_corrections_Qf=gates_Qf,
        )

        x = mpc.true_RK4_disc(x, u_opt, mpc.dt)

        u_hist[step]     = u_opt.detach()
        x_hist[step + 1] = x.detach()

        U_opt_reshaped = U_opt_full.detach().view(mpc.N, n_u)
        u_seq_guess[:-1] = U_opt_reshaped[1:].clone()
        u_seq_guess[-1]  = U_opt_reshaped[-1].clone()

        state_history.pop(0)
        state_history.append(x.detach().clone())

    return x_hist, u_hist
