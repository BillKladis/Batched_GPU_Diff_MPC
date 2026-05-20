"""exp_hardware_v6_sa010_batched.py — batched port of exp_hardware_v6_sa010.py.

Functionally identical to exp_hardware_v6_sa010.py with TWO differences:

1.  Each curriculum-stage call to train_linearization_network is replaced
    by a call to train_linearization_network_batched, with B trajectories
    rolled out in parallel through one batched MPC.control() per step.
2.  Initial-condition samplers are batched. For the swing-up (bottom and
    q-profile-bottom) phases, a small IC perturbation is added so that
    the B trajectories are genuinely diverse — otherwise B identical
    rollouts produce a gradient equivalent to B=1, defeating the point.

Everything else (curriculum, hyperparameters, optimizers, save policy,
evaluation) is unchanged. Drop in the four Phase-1-to-4 modules and this
file, then run.

Tune BATCH_SIZE based on memory:
    B=16   conservative; ~3-4× speedup on CPU, ~8-15× on a typical GPU
    B=64   moderate;     ~7× on CPU, ~25-50× on a typical GPU
    B=128  aggressive;   needs ~3 GB VRAM at N=10, more at deeper horizons
"""

import glob
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

import lin_net as network_module
import mpc_controller as mpc_module
import Simulate as train_module                       # legacy (used for rollout/eval2k)
import Simulate_batched as train_batched              # new batched train
import batch_helpers as bh                            # samplers + SA wrapper

# ── Config ─────────────────────────────────────────────────────────────────────
X0     = [0.0,     0.0, 0.0, 0.0]
X_GOAL = [math.pi, 0.0, 0.0, 0.0]
DT      = 0.05
HORIZON = 10
U_LIM   = 0.10            # Nm — full shoulder authority

STATE_DIM    = 4
CONTROL_DIM  = 2
HIDDEN_DIM   = 128
GATE_RANGE_Q = 0.99
GATE_RANGE_R = 0.20

F_EXTRA_BOUND   = 1.5
F_KICKSTART_AMP = 0.01
W_F_END_REG     = 1.0
F_END_REG_STEPS = 10
Q_NEAR_PI_POWER = 4

META_EPOCHS       = 20
N_BOTTOM_PER_TOP  = 3
N_BOTTOM          = 25
N_TOP             = 100
LR                = 5e-4
WEIGHT_DECAY      = 1e-4
W_Q_PROFILE       = 100.0
PUMP   = [1.0, 1.0, 1.0, 1.0]
STABLE = [2.0, 1.0, 2.0, 1.0]

W_STABLE_PHASE     = 3.0
STABLE_PHASE_STEPS = N_TOP

W_F_POS_ONLY_TOP   = 0.3
F_GATE_THRESH_TOP  = 0.8
DETACH_F_EXTRA_TOP = True

W_F_POS_ONLY_FE    = 0.5
N_FE_STEPS         = 5

W_Q_PROFILE_BOT    = 10.0
N_Q_PROFILE_STEPS  = 5

# Top-phase IC perturbations (already random in the legacy script).
TOP_PERT_Q1  = 0.30
TOP_PERT_Q1D = 0.30
TOP_PERT_Q2  = 0.05            # tight: joint 2 held near 0 by SA freeze
TOP_PERT_Q2D = 0.05

# NEW: small swing-up IC perturbation, so B parallel trajectories are
# genuinely diverse rather than B copies of the same deterministic rollout.
# Magnitudes are well below the swing-up scale (q1 reaches π) but enough
# to decorrelate the per-trajectory gradients.
BOT_PERT_Q1  = 0.05
BOT_PERT_Q1D = 0.10
BOT_PERT_Q2  = 0.02
BOT_PERT_Q2D = 0.02

# ── Batching ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64       # tune to memory; see top docstring

EVAL_EVERY      = 10
SAVE_EVERY      = 50
DIAG_SAVE_EVERY = 20
SAVE_DIR = "saved_models"
LOG_FILE = os.path.join(REPO_DIR, "logs", "hw_v6_sa010_batched.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_energy_demo(n, device, q1_start=0.0):
    demo = torch.zeros((n, 4), dtype=torch.float32, device=device)
    span = math.pi - q1_start
    for i in range(n):
        alpha = i / max(n - 1, 1)
        t = 0.5 * (1.0 - math.cos(math.pi * alpha))
        demo[i, 0] = q1_start + span * t
    return demo


def make_hold_demo(n, device):
    demo = torch.zeros((n, 4), dtype=torch.float32, device=device)
    demo[:, 0] = math.pi
    return demo


def sample_bottom_batched(B, device):
    """Tiny perturbation around rest, so the B trajectories diverge.

    Without this, B identical zero-state rollouts give gradients
    indistinguishable from B=1 (defeats batching). The perturbation
    magnitudes are deliberately small (well below pendulum dynamics scale)
    so the swing-up curriculum semantics are preserved.
    """
    return bh.sample_x0_mixed_batched(
        B, device,
        rest_frac=1.0, spin_frac=0.0, top_frac=0.0,
    ) + bh.sample_top_batched(
        # Reuse sample_top_batched as a uniform-perturbation factory by
        # zeroing the centre (goal_q1=0) and using bottom-phase perts.
        B, device,
        pert_q1=BOT_PERT_Q1, pert_q1d=BOT_PERT_Q1D,
        pert_q2=BOT_PERT_Q2, pert_q2d=BOT_PERT_Q2D,
        goal_q1=0.0,
    )


def eval2k(model, mpc, x0, x_goal, steps=2000):
    """Unbatched evaluation — runs one trajectory at a time (no need to
    batch eval; it's not training-hot)."""
    model.eval()
    x_t, _ = train_module.rollout(lin_net=model, mpc=mpc, x0=x0,
                                  x_goal=x_goal, num_steps=steps)
    traj = x_t.cpu().numpy()
    wraps = np.array([
        math.sqrt(
            math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)) ** 2
            + s[1] ** 2 + s[2] ** 2 + s[3] ** 2
        )
        for s in traj
    ])
    arr = next((i for i, w in enumerate(wraps) if w < 0.3), None)
    post = float((wraps[arr:] < 0.10).mean()) if arr is not None else None
    f01  = float((wraps < 0.10).mean())
    model.train()
    return f01, arr, post


def save_checkpoint(model_kwargs, state_dict, meta, label, save_dir, tag=""):
    name = f"hw_v6_sa010b{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{meta}"
    m = network_module.SeparatedLinearizationNetwork(**model_kwargs).double()
    m.load_state_dict(state_dict)
    network_module.ModelManager(base_dir=save_dir).save_training_session(
        model=m, loss_history=[],
        training_params={
            "experiment": "hardware_v6_sa010_batched",
            "meta_epoch": meta,
            "label": label,
            "u_lim": U_LIM,
            "single_actuated": True,
            "rigid_elbow": True,
            "batch_size": BATCH_SIZE,
        },
        session_name=name,
    )
    return name


def main():
    log = open(LOG_FILE, "w", buffering=1)
    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0     = torch.tensor(X0,     dtype=torch.float32, device=device)   # (4,) for eval2k
    x_goal = torch.tensor(X_GOAL, dtype=torch.float32, device=device)   # (4,), broadcasts

    out("=" * 80)
    out(" EXP: HARDWARE v6 BATCHED — single-actuated (shoulder only), u_max=0.10 Nm")
    out(f" device: {device}    BATCH_SIZE: {BATCH_SIZE}")
    out(f" U_LIM={U_LIM} (single-actuated: elbow frozen at q2=0)")
    out("=" * 80)

    mpc = mpc_module.MPC_controller(
        x0=x0, x_goal=x_goal, N=HORIZON, device=device, u_lim=U_LIM,
    )
    mpc.dt = torch.tensor(DT, dtype=torch.float32, device=device)

    # Batched SA wrapper — works for unbatched (eval2k) and batched (training) input.
    bh.wrap_sa_dynamics_batched(mpc)
    out(f" SA dynamics: elbow frozen at q2=0 (rigid link approximation, batch-aware)")

    demo_bottom = make_energy_demo(N_BOTTOM, device)
    demo_top    = make_hold_demo(N_TOP, device)

    model_kwargs = dict(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM,
        horizon=HORIZON, hidden_dim=HIDDEN_DIM,
        gate_range_q=GATE_RANGE_Q, gate_range_r=GATE_RANGE_R,
        f_extra_bound=F_EXTRA_BOUND, f_kickstart_amp=F_KICKSTART_AMP,
    )

    # Load best hw_v1 checkpoint
    ckpt_paths = glob.glob("saved_models/hw_v1*/*.pth")
    if not ckpt_paths:
        out("ERROR: No hw_v1 checkpoint found. Run exp_hardware_v1.py first.")
        return
    ckpt = max(ckpt_paths, key=os.path.getmtime)
    out(f" Loading from: {ckpt}")
    data = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict = data.get("model_state_dict", data)

    model = network_module.SeparatedLinearizationNetwork(**model_kwargs).to(device).double()
    model.load_state_dict(state_dict)

    optimizer_f = torch.optim.AdamW(model.f_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_q = torch.optim.AdamW(model.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    p = bh.probe_network_batched(model, mpc, device)
    out(f"\n Init: Q[q1] bot={p['bot']['Q_q1']:.3f} mid={p['mid']['Q_q1']:.3f} top={p['top']['Q_q1']:.3f}")
    out(f"       fe   bot={p['bot']['fe_norm']:.3f} mid={p['mid']['fe_norm']:.3f} top={p['top']['fe_norm']:.3f}")
    out(f"\n Starting single-actuated batched fine-tuning ...\n")

    hdr = (f" {'Meta':>5} {'L_bot':>8} {'L_top':>8} "
           f"{'Q@bot':>6} {'Q@top':>6} "
           f"{'fe@bot':>7} {'f01':>6} {'arr':>5} {'post':>6}")
    out(hdr)
    out(" " + "-" * len(hdr.rstrip()))

    best_f01 = 0.0
    best_state = None
    t0 = time.time()

    for meta in range(META_EPOCHS):
        L_bot_last = float("nan")

        # ── (1) BOTTOM PHASE — swing-up energy tracking ───────────────────
        for _ in range(N_BOTTOM_PER_TOP):
            x0_bot = sample_bottom_batched(BATCH_SIZE, device)
            loss_b, _ = train_batched.train_linearization_network_batched(
                lin_net=model, mpc=mpc,
                x0=x0_bot, x_goal=x_goal, demo=demo_bottom,
                num_steps=N_BOTTOM, num_epochs=1, lr=LR,
                track_mode="energy",
                detach_gates_Q_for_qp=True,
                w_f_end_reg=W_F_END_REG,
                f_end_reg_steps=F_END_REG_STEPS,
                external_optimizer=optimizer_f,
                restore_best=False,
            )
            L_bot_last = loss_b[0] if loss_b else float("nan")

        # ── (2) FE PHASE — stable hold near top, position-only penalty ────
        x0_fe = bh.sample_top_batched(
            BATCH_SIZE, device,
            pert_q1=TOP_PERT_Q1,   pert_q1d=TOP_PERT_Q1D,
            pert_q2=TOP_PERT_Q2,   pert_q2d=TOP_PERT_Q2D,
        )
        train_batched.train_linearization_network_batched(
            lin_net=model, mpc=mpc,
            x0=x0_fe, x_goal=x_goal, demo=demo_top,
            num_steps=N_FE_STEPS, num_epochs=1, lr=LR,
            track_mode="cos_q1",
            detach_gates_Q_for_qp=True,
            detach_f_extra_for_qp=True,
            w_f_pos_only=W_F_POS_ONLY_FE,
            external_optimizer=optimizer_f,
            restore_best=False,
        )

        # ── (3) Q-PROFILE-BOTTOM PHASE — pump target everywhere ───────────
        x0_qp_bot = sample_bottom_batched(BATCH_SIZE, device)
        train_batched.train_linearization_network_batched(
            lin_net=model, mpc=mpc,
            x0=x0_qp_bot, x_goal=x_goal, demo=demo_bottom,
            num_steps=N_Q_PROFILE_STEPS, num_epochs=1, lr=LR,
            track_mode="energy",
            detach_gates_Q_for_qp=True,
            w_q_profile=W_Q_PROFILE_BOT,
            q_profile_pump=PUMP, q_profile_stable=PUMP,
            q_profile_state_phase=True,
            external_optimizer=optimizer_q,
            restore_best=False,
        )

        # ── (4) TOP PHASE — stabilisation with full q-profile + stable_phase ──
        x0_top = bh.sample_top_batched(
            BATCH_SIZE, device,
            pert_q1=TOP_PERT_Q1,   pert_q1d=TOP_PERT_Q1D,
            pert_q2=TOP_PERT_Q2,   pert_q2d=TOP_PERT_Q2D,
        )
        loss_t, _ = train_batched.train_linearization_network_batched(
            lin_net=model, mpc=mpc,
            x0=x0_top, x_goal=x_goal, demo=demo_top,
            num_steps=N_TOP, num_epochs=1, lr=LR,
            track_mode="cos_q1",
            w_q_profile=W_Q_PROFILE,
            q_profile_pump=PUMP, q_profile_stable=STABLE,
            q_profile_state_phase=True,
            q_profile_near_pi_power=Q_NEAR_PI_POWER,
            w_stable_phase=W_STABLE_PHASE,
            stable_phase_steps=STABLE_PHASE_STEPS,
            w_f_pos_only=W_F_POS_ONLY_TOP,
            f_gate_thresh=F_GATE_THRESH_TOP,
            detach_f_extra_for_qp=DETACH_F_EXTRA_TOP,
            external_optimizer=optimizer_q,
            restore_best=False,
        )
        L_top = loss_t[0] if loss_t else float("nan")

        # ── Probe + periodic eval/save ─────────────────────────────────────
        p = bh.probe_network_batched(model, mpc, device)
        f01_str = arr_str = post_str = "—"
        mark = ""

        if (meta + 1) % EVAL_EVERY == 0:
            f01, arr, post = eval2k(model, mpc, x0, x_goal)
            f01_str  = f"{f01:.1%}"
            arr_str  = str(arr) if arr is not None else "None"
            post_str = f"{post:.1%}" if post is not None else "N/A"
            if f01 > best_f01:
                best_f01 = f01
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                mark = " ★"

        out(f" [{meta+1:>3}] {L_bot_last:>8.3f} {L_top:>8.3f}  "
            f"  {p['bot']['Q_q1']:.3f}  {p['top']['Q_q1']:.3f}    "
            f" {p['bot']['fe_norm']:.3f}    "
            f"{f01_str:>6} {arr_str:>5} {post_str:>6}{mark}")

        if (meta + 1) % SAVE_EVERY == 0 and best_state is not None:
            name = save_checkpoint(model_kwargs, best_state, meta + 1,
                                   f"best_f01={best_f01:.1%}", SAVE_DIR)
            out(f"    → Saved: {name}")

        if (meta + 1) % DIAG_SAVE_EVERY == 0:
            cur_state = {k: v.clone() for k, v in model.state_dict().items()}
            name = save_checkpoint(model_kwargs, cur_state, meta + 1,
                                   f"diag_ep{meta+1}", SAVE_DIR, tag="_diag")
            out(f"    → Diag snapshot: {name}")
            if best_state is not None:
                bname = save_checkpoint(model_kwargs, best_state, meta + 1,
                                        f"best_f01={best_f01:.1%}_diag{meta+1}",
                                        SAVE_DIR, tag="_best")
                out(f"    → Best state: {bname} (f01={best_f01:.1%})")

    elapsed = time.time() - t0
    if best_state is not None:
        name = save_checkpoint(model_kwargs, best_state, META_EPOCHS,
                               f"best_f01={best_f01:.1%}", SAVE_DIR, tag="_FINAL")
        out(f"\n FINAL best f01={best_f01:.1%} saved: {name}")

    out(f" Total time: {elapsed/60:.1f} min")
    log.close()


if __name__ == "__main__":
    main()
