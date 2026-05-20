import torch

# Real hardware parameters (MAB Robotics double pendulum)
_M1 = 0.10548177618443695
_M2 = 0.07619744360415454
_L1 = 0.05          # link 1 length
_L2 = 0.05          # link 2 length
_R1 = 0.05          # CoM distance link 1
_R2 = 0.03670036749567022   # CoM distance link 2
_I1 = 0.00046166221821039165
_I2 = 0.00023702395072092597
_G  = 9.81
_U_LIM = 0.15       # Nm

# Joint viscous friction (Coulomb omitted — its tanh smoothing creates
# artificial stiffness in MPC linearisation that paralyses control).
# bv=0.005 → at q_dot=10 rad/s friction = 0.05 Nm (33% u_max), at q_dot=1
# only 0.005 (3%): allows free motion at low speed but damps runaway.
_BV1 = 0.005
_BV2 = 0.005


class DoublePendulumDynamics:
    """
    Rigid-body double pendulum matching the real hardware.

    State x format: [q1, q1_dot, q2, q2_dot]
        q1     = absolute angle of link 1 (world frame, 0=down)
        q1_dot = angular velocity of link 1
        q2     = RELATIVE angle of link 2 w.r.t. link 1
        q2_dot = angular velocity of link 2

    Hardware state format is [q1, q2, q1_dot, q2_dot].
    Interface permutation: x_ours = x_hw[[0, 2, 1, 3]]

    Control: tau = [u1, u2] (joint torques, Nm)

    BATCH SUPPORT (new in this revision):
        All methods accept either an unbatched state x of shape (4,) with
        tau of shape (2,), OR a batched state x of shape (..., 4) with tau
        of shape (..., 2). The leading dims `...` are preserved and may be
        empty (unbatched), a single B (per-trajectory batching), or more
        (e.g. B × T for parallel rollout buffers). Output shapes mirror
        the input — unbatched in → unbatched out, batched in → batched
        out. Existing unbatched callers continue to work unchanged.
    """

    def __init__(self, device=None, dtype=torch.float64, u_lim=_U_LIM):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.dtype = dtype

        self.m1 = torch.tensor(_M1, device=self.device, dtype=self.dtype)
        self.m2 = torch.tensor(_M2, device=self.device, dtype=self.dtype)
        self.l1 = torch.tensor(_L1, device=self.device, dtype=self.dtype)
        self.l2 = torch.tensor(_L2, device=self.device, dtype=self.dtype)
        self.r1 = torch.tensor(_R1, device=self.device, dtype=self.dtype)
        self.r2 = torch.tensor(_R2, device=self.device, dtype=self.dtype)
        self.I1 = torch.tensor(_I1, device=self.device, dtype=self.dtype)
        self.I2 = torch.tensor(_I2, device=self.device, dtype=self.dtype)
        self.g  = torch.tensor(_G,  device=self.device, dtype=self.dtype)

        self.u_min = torch.tensor([-u_lim, -u_lim], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([ u_lim,  u_lim], device=self.device, dtype=self.dtype)
        self.bv    = torch.tensor([_BV1,  _BV2 ],  device=self.device, dtype=self.dtype)

    # ──────────────────────────────────────────────────────────────────
    def compute_M_C_G(self, x: torch.Tensor):
        """Compute M, C, G for state x of shape (..., 4).

        Returns:
            M: (..., 2, 2)
            C: (..., 2, 2)
            G: (..., 2)
        For unbatched x of shape (4,), returns shapes (2,2), (2,2), (2,).
        """
        x = x.to(device=self.device, dtype=self.dtype)
        # Indexing on the last axis preserves leading dims. Works for
        # both (4,) → scalars and (B, 4) → (B,) for each component.
        q1     = x[..., 0]
        q1_dot = x[..., 1]
        q2     = x[..., 2]
        q2_dot = x[..., 3]

        # Inertia matrix entries. H has the shape of q2 (broadcastable
        # with q1 since they share leading dims).
        H = self.m2 * self.l1 * self.r2 * torch.cos(q2)
        M11 = (self.I1 + self.m1*self.r1**2 + self.I2
               + self.m2*(self.l1**2 + self.r2**2) + 2*H)
        M12 = self.I2 + self.m2*self.r2**2 + H
        # M22 is q-independent — broadcast it to M11's shape so torch.stack
        # doesn't barf on mismatched dims for batched inputs.
        M22_scalar = self.I2 + self.m2*self.r2**2
        M22 = M22_scalar + torch.zeros_like(M11)

        # Assemble (..., 2, 2): inner stack builds rows along dim=-1,
        # outer stack stacks the two rows along dim=-2.
        M_row0 = torch.stack([M11, M12], dim=-1)
        M_row1 = torch.stack([M12, M22], dim=-1)
        M = torch.stack([M_row0, M_row1], dim=-2)

        # Coriolis / centrifugal matrix (C@qdot identical to hardware formulation)
        h = self.m2 * self.l1 * self.r2 * torch.sin(q2)
        C11 = -2 * h * q2_dot
        C12 = -h * q2_dot
        C21 = h * q1_dot
        C22 = torch.zeros_like(C11)   # was: torch.zeros((), ...) — now batch-shaped

        C_row0 = torch.stack([C11, C12], dim=-1)
        C_row1 = torch.stack([C21, C22], dim=-1)
        C = torch.stack([C_row0, C_row1], dim=-2)

        # Gravity torque vector
        G1 = -(self.m1*self.r1 + self.m2*self.l1)*self.g*torch.sin(q1) \
             - self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G2 = -self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G = torch.stack([G1, G2], dim=-1)

        return M, C, G

    # ──────────────────────────────────────────────────────────────────
    def deriv(self, t, x, tau=None):
        """Continuous-time derivative dx/dt for state x of shape (..., 4).

        tau: (..., 2) joint torques (or None → zeros with matching shape).
        Returns dx/dt of shape (..., 4).
        """
        x = x.to(device=self.device, dtype=self.dtype)
        if tau is None:
            tau = torch.zeros(x.shape[:-1] + (2,),
                              device=self.device, dtype=self.dtype)
        else:
            tau = tau.to(device=self.device, dtype=self.dtype)

        q1_dot = x[..., 1]
        q2_dot = x[..., 3]
        q_dot = torch.stack([q1_dot, q2_dot], dim=-1)   # (..., 2)

        # Viscous friction: self.bv has shape (2,), q_dot has (..., 2).
        # Broadcasting handles both unbatched and batched cases.
        tau_eff = tau - self.bv * q_dot                  # (..., 2)

        M, C, G = self.compute_M_C_G(x)                  # (...,2,2), (...,2,2), (...,2)

        # C @ q_dot via einsum so it works for any leading dims.
        # Unbatched: (2,2) @ (2,) = (2,). Batched: (B,2,2) @ (B,2) = (B,2).
        C_qdot = torch.einsum('...ij,...j->...i', C, q_dot)
        rhs = (tau_eff - C_qdot + G).unsqueeze(-1)       # (..., 2, 1)
        q_ddot = torch.linalg.solve(M, rhs).squeeze(-1)  # (..., 2)

        return torch.stack(
            [q1_dot, q_ddot[..., 0], q2_dot, q_ddot[..., 1]],
            dim=-1,
        )

    # ──────────────────────────────────────────────────────────────────
    def rk4_step(self, x, dt, tau_func=None):
        """RK4 step. x: (..., 4); tau_func: callable returning (..., 2).

        Works unchanged for both unbatched and batched inputs as long as
        the user-supplied tau_func respects the batch shape of x.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)
        if tau_func is None:
            tau = torch.zeros(x.shape[:-1] + (2,),
                              device=self.device, dtype=self.dtype)
        else:
            tau = tau_func(x).to(device=self.device, dtype=self.dtype)

        k1 = self.deriv(0.0, x,                tau)
        k2 = self.deriv(0.0, x + 0.5 * dt * k1, tau)
        k3 = self.deriv(0.0, x + 0.5 * dt * k2, tau)
        k4 = self.deriv(0.0, x + dt       * k3, tau)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ──────────────────────────────────────────────────────────────────
    def step(self, x0, dt, tau_func=None, n_steps=10):
        """Roll out n_steps of RK4. x0: (..., 4) → returns (n_steps+1, ..., 4).

        Note the time axis is the leading dim (consistent with original).
        """
        x = x0.clone().to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)
        traj = [x.clone()]
        h = dt / n_steps
        for _ in range(n_steps):
            x = self.rk4_step(x, h, tau_func)
            traj.append(x.clone())
        return torch.stack(traj, dim=0)


# ──────────────────────────────────────────────────────────────────────────
# Self-test: batched output must match unbatched output trajectory-by-trajectory.
# Run with: python true_dynamics.py
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")
    dyn = DoublePendulumDynamics(device=device)

    # Build B random states and torques.
    B = 8
    x_batch = torch.randn(B, 4, dtype=torch.float64, device=device) * 0.5
    tau_batch = torch.randn(B, 2, dtype=torch.float64, device=device) * 0.05

    # Batched evaluation.
    dx_batch = dyn.deriv(0.0, x_batch, tau_batch)         # (B, 4)
    rk_batch = dyn.rk4_step(x_batch, 0.05,
                            tau_func=lambda _x: tau_batch)  # (B, 4)

    # Unbatched evaluation, per element.
    dx_indiv = torch.stack([dyn.deriv(0.0, x_batch[i], tau_batch[i])
                            for i in range(B)], dim=0)
    rk_indiv = torch.stack([dyn.rk4_step(x_batch[i], 0.05,
                                         tau_func=lambda _x, i=i: tau_batch[i])
                            for i in range(B)], dim=0)

    err_deriv = (dx_batch - dx_indiv).abs().max().item()
    err_rk4   = (rk_batch - rk_indiv).abs().max().item()

    print(f"max |batched - per-element| (deriv):   {err_deriv:.3e}")
    print(f"max |batched - per-element| (rk4_step):{err_rk4:.3e}")
    assert err_deriv < 1e-12, "deriv batched mismatch"
    assert err_rk4   < 1e-12, "rk4_step batched mismatch"
    print("OK — batched dynamics match unbatched to machine precision.")
