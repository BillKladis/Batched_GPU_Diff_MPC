"""
mpc_controller.py — Linearised receding-horizon MPC with switchable QP backend.

Two QP solvers, sharing the same cost-matrix construction:

    solver_backend="cvx" (default)
        cvxpylayers + SCS. Slower (~50–300 ms per solve) but DIFFERENTIABLE
        via implicit-diff through KKT. Required for training.

    solver_backend="osqp"
        OSQP direct C solver. Fast (~1–5 ms per solve) but NOT differentiable.
        Use at deployment time when no gradients are needed.

QP form (control-space, delta-u parameterisation):
    min  ½ ΔUᵀ H ΔU  +  fᵀ ΔU
    s.t. lb ≤ ΔU ≤ ub

The cvxpylayers DPP-compliant formulation passes H via its Cholesky-like
square root H_sqrt where H_sqrt.T @ H_sqrt = H. OSQP takes H directly
in upper-triangular CSC form.

BATCH SUPPORT (new in this revision):
    All public methods are dual-mode: they accept either unbatched inputs
    (legacy contract, used by Simulate.py and hardware_deploy.py) or
    batched inputs with a leading B dimension. Detection is from the rank
    of `current_state`:
        current_state.dim() == 1 → unbatched, all I/O shapes match legacy
        current_state.dim() == 2 → batched, leading B added to every tensor

    cvxpylayers handles batched parameters natively. OSQP loops over the
    batch in Python (OSQP is deploy-only / not differentiable; single
    trajectory is the realistic case there).
"""

from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch.func import jacrev, vmap

import MPC_dynamics
import true_dynamics


class MPC_controller:

    def __init__(
        self,
        x0: torch.Tensor,
        x_goal: torch.Tensor,
        N: int,
        device: torch.device,
        u_lim: float = 0.15,
        solver_backend: str = "cvx",      # "cvx" (differentiable) or "osqp" (fast)
        qp_eps: float = 1e-3,              # OSQP convergence tolerance
        qp_max_iters: int = 200,           # OSQP iteration cap (also used by SCS)
    ):
        self.device = device
        self.x0     = x0.detach().clone().to(device=device, dtype=torch.float64)
        self.x_goal = x_goal.detach().clone().to(device=device, dtype=torch.float64)
        self.N      = N
        self.dt     = torch.tensor(0.05, device=device, dtype=torch.float64)

        # Baseline cost diagonals — scaled for real hardware (m~0.1kg, l=0.05m, u_max=0.15Nm).
        # B_vel ≈ 55-90 per step (small inertia → large acceleration response).
        # Q_vel must stay ≤ 0.0001 to keep H well-conditioned at all linearisation points.
        # Q_pos = 0.1 gives strong position drive (cost ratio Q*π²/R ≈ 1 >> R*u_max²=0.02).
        # Top conditioning: cond ≈ 2.3e6 (acceptable for SCS with eps=1e-6).
        # State order: [q1, q1_dot, q2, q2_dot].  Hardware order: [q1, q2, q1_dot, q2_dot].
        self.q_base_diag = torch.tensor(
            [0.1, 0.0001, 0.1, 0.0001], device=device, dtype=torch.float64
        )
        self.r_base_diag = torch.tensor(
            [1.0, 1.0], device=device, dtype=torch.float64
        )
        self.Qf = torch.diag(torch.tensor(
            [0.2, 0.0002, 0.2, 0.0002], device=device, dtype=torch.float64
        ))

        self.true_dynamics = true_dynamics.DoublePendulumDynamics(device=device, u_lim=u_lim)
        self.MPC_dynamics  = MPC_dynamics.DoublePendulumDynamics(device=device, u_lim=u_lim)
        self.n_u_total = self.N * self.MPC_dynamics.u_min.shape[0]
        self.qp_fallback_count = 0

        # Stash solver-tuning parameters for both backends.
        self.qp_eps       = float(qp_eps)
        self.qp_max_iters = int(qp_max_iters)

        # Build the chosen backend. cvx is the default (covers training).
        if solver_backend not in ("cvx", "osqp"):
            raise ValueError(
                f"solver_backend must be 'cvx' or 'osqp', got {solver_backend!r}"
            )
        self.solver_backend = solver_backend
        if solver_backend == "cvx":
            self._build_qp_layer()
        else:
            self._build_osqp_workspace()

    # ──────────────────────────────────────────────────────────────────────
    # cvxpylayers QP construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_qp_layer(self):
        n = self.n_u_total
        DU      = cp.Variable(n)
        H_sqrt  = cp.Parameter((n, n))     # H = H_sqrt.T @ H_sqrt
        f_par   = cp.Parameter(n)
        lb_par  = cp.Parameter(n)
        ub_par  = cp.Parameter(n)

        objective = cp.Minimize(
            0.5 * cp.sum_squares(H_sqrt @ DU) + f_par @ DU
        )
        constraints = [DU >= lb_par, DU <= ub_par]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "QP is not DPP-compliant"

        self.qp_layer = CvxpyLayer(
            problem,
            parameters=[H_sqrt, f_par, lb_par, ub_par],
            variables=[DU],
        )

    # ──────────────────────────────────────────────────────────────────────
    # OSQP workspace (fast direct solver, no autograd)
    # ──────────────────────────────────────────────────────────────────────
    def _build_osqp_workspace(self):
        """Pre-build a reusable OSQP workspace.

        OSQP wants P in upper-triangular CSC form. We allocate a fully
        dense upper-triangular pattern at setup so that on every solve the
        pattern (nnz indices) is unchanged — only the .data values move.
        That lets us call `update(Px=...)` instead of `setup(...)` each
        step, which is the difference between ~1 ms and ~10 ms per solve.
        """
        import osqp
        import scipy.sparse as sp

        n = self.n_u_total

        # Fully-dense upper-triangular pattern (n*(n+1)/2 nonzeros).
        # Setting all to 1.0 first guarantees no entries get dropped.
        P_dense_init = np.triu(np.ones((n, n), dtype=np.float64))
        P_csc = sp.csc_matrix(P_dense_init)

        # Constraint matrix for box constraints is just identity.
        A_csc = sp.eye(n, format="csc", dtype=np.float64)

        self.osqp_prob = osqp.OSQP()
        self.osqp_prob.setup(
            P_csc, np.zeros(n), A_csc,
            np.full(n, -1e6), np.full(n, 1e6),
            eps_abs=self.qp_eps,
            eps_rel=self.qp_eps,
            max_iter=self.qp_max_iters,
            verbose=False,
            polish=False,            # polishing adds ~0.5 ms; box QPs rarely need it
            warm_start=True,         # carry over previous solution as warm start
            adaptive_rho=False,      # adaptive_rho costs Python overhead per solve
            check_termination=25,    # check convergence every 25 iters (cheap)
            scaling=10,              # equilibration helps conditioning (~free)
        )

        # Cache the (row, col) index arrays so we can extract H values from a
        # dense numpy array in CSC column-major order. scipy.sparse.find
        # returns (rows, cols, data) — we want them in the same order OSQP's
        # internal Px array expects, which is the order P_csc.data was built.
        rows = []
        cols = []
        for j in range(n):
            for i in range(j + 1):
                rows.append(i)
                cols.append(j)
        self._osqp_P_rows = np.asarray(rows, dtype=np.int64)
        self._osqp_P_cols = np.asarray(cols, dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────
    # Cost matrices
    # ──────────────────────────────────────────────────────────────────────
    def build_cost_matrices(
        self,
        diag_corrections_Q: Optional[torch.Tensor]  = None,
        diag_corrections_R: Optional[torch.Tensor]  = None,
        diag_corrections_Qf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q_bar, R_diag).

        Unbatched: Q_bar (N*4, N*4), R_diag (N*2,)
        Batched:   Q_bar (B, N*4, N*4), R_diag (B, N*2) or (N*2,)

        Batching is detected from whichever correction is provided; if
        none are provided, returns the unbatched defaults.
        """
        # Detect batching from whichever correction tensor has a leading B dim.
        batch_shape: Tuple[int, ...] = ()
        for corr, expected_rank_unbatched in (
            (diag_corrections_Q,  2),    # unbatched (N-1, 4)
            (diag_corrections_R,  2),    # unbatched (N,   2)
            (diag_corrections_Qf, 1),    # unbatched (4,)
        ):
            if corr is not None and corr.dim() == expected_rank_unbatched + 1:
                batch_shape = (corr.shape[0],)
                break

        is_batched = len(batch_shape) > 0

        # Q running diagonal.
        if diag_corrections_Q is not None:
            # diag_corrections_Q broadcasts cleanly against q_base_diag (state_dim,)
            # whether unbatched (N-1, 4) or batched (B, N-1, 4).
            q_k = self.q_base_diag * diag_corrections_Q
            Q_run = q_k.flatten(start_dim=-2)        # (N-1)*4 or (B, (N-1)*4)
        else:
            Q_run = self.q_base_diag.repeat(self.N - 1)   # ((N-1)*4,)
            if is_batched:
                Q_run = Q_run.unsqueeze(0).expand(*batch_shape, -1)   # (B, (N-1)*4)

        # Pad with zeros where Qf will be inserted next.
        if is_batched:
            zero_pad = torch.zeros(
                (*batch_shape, 4), device=self.device, dtype=torch.float64,
            )
        else:
            zero_pad = torch.zeros(4, device=self.device, dtype=torch.float64)
        Q_diag_full = torch.cat([Q_run, zero_pad], dim=-1)    # (N*4,) or (B, N*4)
        Q_bar = torch.diag_embed(Q_diag_full)                 # (N*4,N*4) or (B,N*4,N*4)

        # Insert Qf block.
        if diag_corrections_Qf is not None:
            # self.Qf.diag() is (4,); * (4,) or (B,4) → broadcasts.
            Qf_diag = self.Qf.diag() * diag_corrections_Qf
            Qf_block = torch.diag_embed(Qf_diag)              # (4,4) or (B,4,4)
            Q_bar[..., -4:, -4:] = Qf_block
        else:
            # self.Qf is (4,4); broadcasts to (B, 4, 4) on assignment.
            Q_bar[..., -4:, -4:] = self.Qf

        # R diagonal vector.
        if diag_corrections_R is not None:
            r_k = self.r_base_diag * diag_corrections_R
            R_diag = r_k.flatten(start_dim=-2)        # (N*2,) or (B, N*2)
        else:
            R_diag = self.r_base_diag.repeat(self.N)  # (N*2,)
            # Keep 1D when no batched corrections — downstream broadcasts fine.

        return Q_bar, R_diag

    # ──────────────────────────────────────────────────────────────────────
    # Energy helper
    # ──────────────────────────────────────────────────────────────────────
    def compute_energy_single(self, x: torch.Tensor) -> torch.Tensor:
        """Total mechanical energy (T + V) for the real rigid-body hardware.

        x: (4,) → scalar energy
        x: (..., 4) → (...,) energy
        """
        dyn = self.true_dynamics
        m1, m2 = dyn.m1, dyn.m2
        l1, r1, r2 = dyn.l1, dyn.r1, dyn.r2
        I1, I2 = dyn.I1, dyn.I2
        g = dyn.g

        # Use ... indexing so this works for any leading shape.
        q1     = x[..., 0]
        q1_dot = x[..., 1]
        q2     = x[..., 2]
        q2_dot = x[..., 3]

        # Potential energy (reference: both links fully down)
        V = -m1*g*r1*torch.cos(q1) - m2*g*(l1*torch.cos(q1) + r2*torch.cos(q1 + q2))

        # Kinetic energy: rotational + translational CoM
        # CoM1 velocity magnitude squared: (r1*q1_dot)^2
        v1_sq = (r1 * q1_dot) ** 2
        KE1 = 0.5*m1*v1_sq + 0.5*I1*q1_dot**2

        # CoM2 velocity
        vx2 = l1*torch.cos(q1)*q1_dot   + r2*torch.cos(q1+q2)*(q1_dot+q2_dot)
        vy2 = l1*torch.sin(q1)*q1_dot   + r2*torch.sin(q1+q2)*(q1_dot+q2_dot)
        v2_sq = vx2**2 + vy2**2
        KE2 = 0.5*m2*v2_sq + 0.5*I2*(q1_dot+q2_dot)**2

        return KE1 + KE2 + V

    # Convenience alias for batched callers (no behavior change vs single).
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Alias accepting any leading batch shape; same math as compute_energy_single."""
        return self.compute_energy_single(x)

    # ──────────────────────────────────────────────────────────────────────
    # Discretisation
    # ──────────────────────────────────────────────────────────────────────
    def true_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor,
                      n_sub: int = 10) -> torch.Tensor:
        """Discrete-time true dynamics step. x: (...,4), u: (...,2)."""
        # 10 sub-steps of h=dt/10=0.005s prevent Coriolis overflow (M22^{-1}≈2944).
        h = dt / n_sub
        for _ in range(n_sub):
            t0 = h.new_zeros(())
            tH = 0.5 * h
            t1 = h
            f1 = self.true_dynamics.deriv(t0, x, u)
            f2 = self.true_dynamics.deriv(tH, x + 0.5 * h * f1, u)
            f3 = self.true_dynamics.deriv(tH, x + 0.5 * h * f2, u)
            f4 = self.true_dynamics.deriv(t1, x + h * f3, u)
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x

    def MPC_RK4_disc(self, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Discrete-time MPC-model step (no Coulomb / matches MPC_dynamics)."""
        t0 = dt.new_zeros(())
        tH = 0.5 * dt
        t1 = dt
        f1 = self.MPC_dynamics.deriv(t0, x, u)
        f2 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f1, u)
        f3 = self.MPC_dynamics.deriv(tH, x + 0.5 * dt * f2, u)
        f4 = self.MPC_dynamics.deriv(t1, x + dt * f3, u)
        return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    # ──────────────────────────────────────────────────────────────────────
    # Linearisation
    # ──────────────────────────────────────────────────────────────────────
    def linearize_discrete(
        self,
        x_batch: torch.Tensor,
        u_batch: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Compute Jacobians of MPC_RK4_disc w.r.t. x and u along the horizon.

        Unbatched:   x_batch (N, 4),   u_batch (N, 2)
                     → A_list[N] each (4, 4),   B_list[N] each (4, 2)
        Batched:     x_batch (B, N, 4), u_batch (B, N, 2)
                     → A_list[N] each (B, 4, 4), B_list[N] each (B, 4, 2)

        Implementation: vmap over the horizon axis; one more vmap level
        wrapping it for batched inputs.
        """
        def step_fn(x, u):
            return self.MPC_RK4_disc(x, u, dt)

        if x_batch.dim() == 2:
            jac_x_fn = vmap(jacrev(step_fn, argnums=0))
            jac_u_fn = vmap(jacrev(step_fn, argnums=1))
        elif x_batch.dim() == 3:
            jac_x_fn = vmap(vmap(jacrev(step_fn, argnums=0)))
            jac_u_fn = vmap(vmap(jacrev(step_fn, argnums=1)))
        else:
            raise ValueError(
                f"linearize_discrete: x_batch rank must be 2 or 3, got {x_batch.dim()}"
            )

        A_full = jac_x_fn(x_batch.detach(), u_batch.detach())
        B_full = jac_u_fn(x_batch.detach(), u_batch.detach())

        # Slice the horizon axis (which is always dim=-3 of the Jacobian
        # tensors: unbatched (N,4,4) → dim 0, batched (B,N,4,4) → dim 1).
        N = A_full.shape[-3]
        A_list = [A_full.select(-3, i) for i in range(N)]
        B_list = [B_full.select(-3, i) for i in range(N)]
        return A_list, B_list

    def linearize_horizon(self, x_lin_seq, u_lin_seq):
        return self.linearize_discrete(x_lin_seq, u_lin_seq, self.dt)

    # ──────────────────────────────────────────────────────────────────────
    # Nominal rollout helper (used by Simulate.py for energy shaping)
    # ──────────────────────────────────────────────────────────────────────
    def compute_nominal_rollout(
        self,
        current_state: torch.Tensor,
        u_guess_seq:   torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward-roll current_state under u_guess_seq under the MPC model.

        Unbatched: current_state (4,), u_guess_seq (N, 2)
                   → X_bar_seq (N, 4), B_list[N] each (4, 2)
        Batched:   current_state (B, 4), u_guess_seq (B, N, 2)
                   → X_bar_seq (B, N, 4), B_list[N] each (B, 4, 2)
        """
        x_op_list: list = []
        X_bar_list: list = []
        curr = current_state.detach()
        for i in range(self.N):
            x_op_list.append(curr)
            u_at_i = u_guess_seq[..., i, :].detach()
            curr = self.MPC_RK4_disc(curr, u_at_i, self.dt)
            X_bar_list.append(curr)
        x_op_seq  = torch.stack(x_op_list,  dim=-2)    # (N, 4) or (B, N, 4)
        X_bar_seq = torch.stack(X_bar_list, dim=-2)
        _, B_list = self.linearize_horizon(x_op_seq, u_guess_seq.detach())
        return X_bar_seq, B_list

    # ──────────────────────────────────────────────────────────────────────
    # Prediction matrices
    # ──────────────────────────────────────────────────────────────────────
    def build_prediction_matrices(self, A_seq, B_seq):
        """Lower-triangular block-Toeplitz prediction matrices.

        Unbatched: A_seq[N] each (n_x, n_x), B_seq[N] each (n_x, n_u)
                   → A_big (N*n_x, n_x), B_big (N*n_x, N*n_u)
        Batched:   A_seq[N] each (B, n_x, n_x), B_seq[N] each (B, n_x, n_u)
                   → A_big (B, N*n_x, n_x), B_big (B, N*n_x, N*n_u)
        """
        N = len(A_seq)
        is_batched = (A_seq[0].dim() == 3)
        n_x = A_seq[0].shape[-2]
        n_u = B_seq[0].shape[-1]

        Phi_rows = [A_seq[0]]
        Phi_prev = A_seq[0]
        for i in range(1, N):
            Phi_prev = A_seq[i] @ Phi_prev        # batched matmul if 3-D
            Phi_rows.append(Phi_prev)
        A_big = torch.cat(Phi_rows, dim=-2)        # along block-row axis

        # Zero block whose leading dims match the rest.
        if is_batched:
            B = A_seq[0].shape[0]
            zero_block = torch.zeros((B, n_x, n_u), device=self.device, dtype=torch.float64)
        else:
            zero_block = torch.zeros((n_x, n_u), device=self.device, dtype=torch.float64)

        B_cols = []
        for j in range(N):
            col_blocks = [zero_block] * j
            col = B_seq[j]
            col_blocks.append(col)
            for i in range(j + 1, N):
                col = A_seq[i - 1] @ col
                col_blocks.append(col)
            B_cols.append(torch.cat(col_blocks, dim=-2))    # (N*n_x, n_u) or (B, N*n_x, n_u)
        B_big = torch.cat(B_cols, dim=-1)                   # (N*n_x, N*n_u) or (B, N*n_x, N*n_u)
        return A_big, B_big

    # ──────────────────────────────────────────────────────────────────────
    # QP cost in delta-u form
    # ──────────────────────────────────────────────────────────────────────
    def build_qp_matrices_delta(
        self,
        B_big:    torch.Tensor,
        X_bar:    torch.Tensor,
        U_bar:    torch.Tensor,
        x_goal:   torch.Tensor,
        Q_bar:    torch.Tensor,
        R_diag:   torch.Tensor,
        extra_linear_control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the QP cost matrices H, f.

        Unbatched: B_big (N*4, N*2), X_bar (N*4,), U_bar (N*2,), x_goal (4,)
                   → H (N*2, N*2), f (N*2,)
        Batched:   B_big (B, N*4, N*2), X_bar (B, N*4), U_bar (B, N*2), x_goal (B, 4)
                   → H (B, N*2, N*2), f (B, N*2)

        extra_linear_control: (N*2,) or (B, N*2) — added DIRECTLY to f in
        control space, bypassing B̄ᵀ mapping. Used for τ1-only energy
        shaping (see Simulate.py).
        """
        is_batched = (B_big.dim() == 3)

        # X_ref construction: tile x_goal across the horizon.
        if is_batched:
            B = x_goal.shape[0]
            X_ref = x_goal.unsqueeze(-2).expand(B, self.N, -1).reshape(B, -1)
        else:
            X_ref = x_goal.unsqueeze(0).expand(self.N, -1).reshape(-1)

        E_raw = X_bar - X_ref

        # Angle-wrap q1 and q2 errors into (−π, π].
        q1_idx = torch.arange(0, 4 * self.N, 4, device=self.device)
        q2_idx = torch.arange(2, 4 * self.N, 4, device=self.device)
        angle_idx = torch.cat([q1_idx, q2_idx])

        # Use ... indexing so it works for both (N*4,) and (B, N*4).
        E = E_raw.clone()
        angle_vals = E_raw[..., angle_idx]
        E[..., angle_idx] = torch.atan2(
            torch.sin(angle_vals),
            torch.cos(angle_vals),
        )

        # H = 2 B̄ᵀ Q̄ B̄ + diag(2 R)
        # transpose(-2,-1) is the batched-safe replacement for .T.
        BTQ    = B_big.transpose(-2, -1) @ Q_bar          # (..., N*2, N*4)
        H_quad = 2.0 * (BTQ @ B_big)                       # (..., N*2, N*2)

        # diag(2 * R_diag): use diag_embed so a (B, n) input → (B, n, n).
        # If R_diag is 1-D in batched mode (no R corrections), broadcast it.
        if is_batched and R_diag.dim() == 1:
            R_diag_b = R_diag.unsqueeze(0).expand(B_big.shape[0], -1)
        else:
            R_diag_b = R_diag
        H_R = torch.diag_embed(2.0 * R_diag_b)             # (..., N*2, N*2)
        H = H_quad + H_R
        H = 0.5 * (H + H.transpose(-2, -1))

        eye_n = torch.eye(H.shape[-1], device=self.device, dtype=torch.float64)
        H = H + 1e-4 * eye_n                # broadcasts (n,n) into (B,n,n) cleanly

        # state_linear_term = 2 Q̄ E   (matrix–vector under batched leading dims).
        state_linear_term = 2.0 * torch.einsum('...ij,...j->...i', Q_bar, E)

        # f = B̄ᵀ state_linear_term + 2 R · Ū
        f_state = torch.einsum('...ij,...j->...i', B_big.transpose(-2, -1), state_linear_term)
        # R_diag * U_bar broadcasts: R_diag (N*2,) or (B,N*2);  U_bar same shape.
        f = f_state + 2.0 * (R_diag * U_bar)

        if extra_linear_control is not None:
            f = f + extra_linear_control

        return H, f

    def build_constraints_delta(self, U_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ΔU-bounds. U_bar (N*2,) or (B, N*2) → lb, ub same shape."""
        lb_base = self.MPC_dynamics.u_min.repeat(self.N)    # (N*2,)
        ub_base = self.MPC_dynamics.u_max.repeat(self.N)    # (N*2,)
        # Broadcasting: (N*2,) - (..., N*2) → (..., N*2).
        return lb_base - U_bar, ub_base - U_bar

    # ──────────────────────────────────────────────────────────────────────
    # QP formulation orchestration
    # ──────────────────────────────────────────────────────────────────────
    def QP_formulation(
        self,
        current_state: torch.Tensor,
        u_guess_seq:   torch.Tensor,
        x_goal:        torch.Tensor,
        diag_corrections_Q:  Optional[torch.Tensor] = None,
        diag_corrections_R:  Optional[torch.Tensor] = None,
        extra_linear_control: Optional[torch.Tensor] = None,
        diag_corrections_Qf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build H, f, U_bar for the QP.

        Unbatched: current_state (4,), u_guess_seq (N,2), x_goal (4,)
                   → H (N*2, N*2), f (N*2,), U_bar (N*2,)
        Batched:   current_state (B, 4), u_guess_seq (B,N,2), x_goal (B, 4)
                   → H (B, N*2, N*2), f (B, N*2), U_bar (B, N*2)
        """
        x_op_list  = []
        X_bar_list = []
        curr_x = current_state
        for i in range(self.N):
            x_op_list.append(curr_x)
            # u_guess_seq[..., i, :] selects the i-th time step regardless of
            # whether u_guess_seq is (N,2) or (B,N,2).
            u_at_i = u_guess_seq[..., i, :]
            curr_x = self.MPC_RK4_disc(curr_x, u_at_i, self.dt)
            X_bar_list.append(curr_x)

        # stack along dim=-2 puts horizon as the second-to-last axis.
        # Unbatched: list of (4,) → (N, 4).  Batched: list of (B, 4) → (B, N, 4).
        x_op_seq = torch.stack(x_op_list, dim=-2)
        # X_bar = vertical concat of predicted states along the time→features axis.
        # Unbatched: cat list of (4,) along -1 → (N*4,).
        # Batched:   cat list of (B,4) along -1 → (B, N*4).
        X_bar = torch.cat(X_bar_list, dim=-1)
        # U_bar = flatten u_guess_seq's last two dims.
        U_bar = u_guess_seq.flatten(start_dim=-2)

        A_list, B_list = self.linearize_horizon(x_op_seq, u_guess_seq)
        _, B_big = self.build_prediction_matrices(A_list, B_list)

        Q_bar, R_diag = self.build_cost_matrices(
            diag_corrections_Q  = diag_corrections_Q,
            diag_corrections_R  = diag_corrections_R,
            diag_corrections_Qf = diag_corrections_Qf,
        )

        H, f = self.build_qp_matrices_delta(
            B_big, X_bar, U_bar, x_goal, Q_bar, R_diag,
            extra_linear_control=extra_linear_control,
        )
        return H, f, U_bar

    # ──────────────────────────────────────────────────────────────────────
    # QP solve
    # ──────────────────────────────────────────────────────────────────────
    def solve_mpc_qp(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the QP and return ΔU* (n,) or (B, n). Dispatches by backend."""
        if self.solver_backend == "osqp":
            return self._solve_mpc_qp_osqp(H, f, lb, ub)
        return self._solve_mpc_qp_cvx(H, f, lb, ub)

    def _solve_mpc_qp_cvx(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """cvxpylayers backend — slower but differentiable. Used in training.

        Accepts unbatched (H: (n,n), f/lb/ub: (n,)) or batched (with leading
        B dim on all parameters). cvxpylayers supports batched parameters
        natively and returns a leading B dim on the solution.
        """
        is_batched = (H.dim() == 3)
        n = self.n_u_total

        def _fallback_zero():
            self.qp_fallback_count += 1
            if is_batched:
                return torch.zeros(H.shape[0], n, device=self.device, dtype=torch.float64)
            return torch.zeros(n, device=self.device, dtype=torch.float64)

        if not torch.isfinite(H).all() or not torch.isfinite(f).all():
            return _fallback_zero()

        # Cholesky of H to get H_sqrt such that H_sqrt.T @ H_sqrt = H.
        # batched Cholesky: torch.linalg.cholesky handles (..., n, n) → (..., n, n).
        try:
            eye_n = torch.eye(n, device=self.device, dtype=torch.float64)
            H_reg = H + 1e-6 * eye_n
            L = torch.linalg.cholesky(H_reg)
            H_sqrt = L.transpose(-2, -1)
        except Exception:
            return _fallback_zero()

        try:
            (DU_opt,) = self.qp_layer(
                H_sqrt, f, lb, ub,
                solver_args={
                    "solve_method": "SCS",
                    "eps":       self.qp_eps,
                    "max_iters": self.qp_max_iters,
                },
            )
        except Exception:
            return _fallback_zero()

        if not torch.isfinite(DU_opt).all():
            return _fallback_zero()

        DU_opt = DU_opt.clamp(lb, ub)

        if DU_opt.requires_grad:
            DU_opt.register_hook(
                lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            )
        return DU_opt

    def _solve_mpc_qp_osqp(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """OSQP backend — direct C QP solver, ~10–100× faster than cvx.

        Not differentiable; deploy-only. OSQP doesn't batch natively, so
        batched inputs are processed by looping over the batch dimension.
        The realistic deploy case is a single trajectory anyway.
        """
        is_batched = (H.dim() == 3)
        if not is_batched:
            return self._solve_mpc_qp_osqp_single(H, f, lb, ub)

        # Batched: process per-trajectory; warm-start is shared across calls
        # by virtue of self.osqp_prob being reused. This is the right thing
        # at deploy (where you'd typically not even pass batched anyway).
        results = []
        B = H.shape[0]
        for i in range(B):
            results.append(self._solve_mpc_qp_osqp_single(H[i], f[i], lb[i], ub[i]))
        return torch.stack(results, dim=0)

    def _solve_mpc_qp_osqp_single(
        self,
        H:  torch.Tensor,
        f:  torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a single (unbatched) QP via OSQP. Internal helper."""
        def _fallback_zero():
            self.qp_fallback_count += 1
            return torch.zeros(self.n_u_total, device=self.device, dtype=torch.float64)

        if not torch.isfinite(H).all() or not torch.isfinite(f).all():
            return _fallback_zero()

        # Symmetric ridge for numerical safety (matches cvx path).
        H_np = H.detach().cpu().numpy().astype(np.float64, copy=False)
        n = H_np.shape[0]
        H_np = H_np + 1e-6 * np.eye(n, dtype=np.float64)

        # Extract upper-triangular values in the same column-major CSC order
        # OSQP saw at setup time.
        Px = H_np[self._osqp_P_rows, self._osqp_P_cols]
        f_np  = f.detach().cpu().numpy().astype(np.float64, copy=False)
        lb_np = lb.detach().cpu().numpy().astype(np.float64, copy=False)
        ub_np = ub.detach().cpu().numpy().astype(np.float64, copy=False)

        try:
            self.osqp_prob.update(Px=Px, q=f_np, l=lb_np, u=ub_np)
            result = self.osqp_prob.solve()
        except Exception:
            return _fallback_zero()

        # OSQP status: 'solved' / 'solved_inaccurate' are both usable.
        # Anything else (max_iter_reached, primal_infeasible, ...) → fallback.
        status = getattr(result.info, "status", "")
        if status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            return _fallback_zero()

        DU = np.asarray(result.x, dtype=np.float64)
        if not np.isfinite(DU).all():
            return _fallback_zero()
        DU_t = torch.from_numpy(DU).to(device=self.device, dtype=torch.float64)
        DU_t = DU_t.clamp(lb, ub)
        return DU_t

    # ──────────────────────────────────────────────────────────────────────
    # Top-level control entry point
    # ──────────────────────────────────────────────────────────────────────
    def control(
        self,
        current_state: torch.Tensor,
        x_lin_seq:     torch.Tensor,    # kept for API compatibility
        u_lin_seq:     torch.Tensor,
        x_goal:        torch.Tensor,
        diag_corrections_Q:   Optional[torch.Tensor] = None,
        diag_corrections_R:   Optional[torch.Tensor] = None,
        extra_linear_control: Optional[torch.Tensor] = None,
        diag_corrections_Qf:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one MPC step.

        Unbatched: current_state (4,), x_lin_seq (N,4), u_lin_seq (N,2),
                   x_goal (4,) → u_opt (n_u,), U_opt (N*n_u,)
        Batched:   current_state (B,4), x_lin_seq (B,N,4), u_lin_seq (B,N,2),
                   x_goal (B,4) → u_opt (B,n_u), U_opt (B,N*n_u)
        """
        H, f, U_bar = self.QP_formulation(
            current_state, u_lin_seq, x_goal,
            diag_corrections_Q   = diag_corrections_Q,
            diag_corrections_R   = diag_corrections_R,
            extra_linear_control = extra_linear_control,
            diag_corrections_Qf  = diag_corrections_Qf,
        )
        lb_delta, ub_delta = self.build_constraints_delta(U_bar)
        Delta_U_opt = self.solve_mpc_qp(H, f, lb_delta, ub_delta)
        U_opt = U_bar + Delta_U_opt

        n_u = self.MPC_dynamics.u_min.shape[0]
        # Slice along last dim so it works for (N*n_u,) and (B, N*n_u).
        u_opt = torch.nan_to_num(U_opt[..., :n_u], nan=0.0, posinf=0.0, neginf=0.0)
        return u_opt, U_opt
