import torch

# Real hardware parameters (MAB Robotics double pendulum)
_M1 = 0.10548177618443695
_M2 = 0.07619744360415454
_L1 = 0.05
_L2 = 0.05
_R1 = 0.05
_R2 = 0.03670036749567022
_I1 = 0.00046166221821039165
_I2 = 0.00023702395072092597
_G  = 9.81
_U_LIM = 0.15

# Joint viscous friction — must match true_dynamics so MPC predictions agree.
_BV1 = 0.005
_BV2 = 0.005


class DoublePendulumDynamics:
    """
    MPC's internal model — same rigid-body double pendulum equations as
    true_dynamics, kept in a separate module so Coulomb / stiction terms
    can be added to true_dynamics without polluting the MPC linearisation
    (see NEXT_STEPS.md). DO NOT add discontinuous friction here.

    State x format: [q1, q1_dot, q2, q2_dot]
    Control: tau = [u1, u2] (joint torques, Nm)

    BATCH SUPPORT (new in this revision):
        All methods accept either an unbatched state x of shape (4,) with
        tau of shape (2,), OR a batched state x of shape (..., 4) with tau
        of shape (..., 2). Leading dims are preserved. Existing unbatched
        callers continue to work unchanged.
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

    def compute_M_C_G(self, x: torch.Tensor):
        """x: (..., 4) → M: (...,2,2), C: (...,2,2), G: (...,2)."""
        x = x.to(device=self.device, dtype=self.dtype)
        q1     = x[..., 0]
        q1_dot = x[..., 1]
        q2     = x[..., 2]
        q2_dot = x[..., 3]

        H = self.m2 * self.l1 * self.r2 * torch.cos(q2)
        M11 = (self.I1 + self.m1*self.r1**2 + self.I2
               + self.m2*(self.l1**2 + self.r2**2) + 2*H)
        M12 = self.I2 + self.m2*self.r2**2 + H
        M22_scalar = self.I2 + self.m2*self.r2**2
        M22 = M22_scalar + torch.zeros_like(M11)

        M_row0 = torch.stack([M11, M12], dim=-1)
        M_row1 = torch.stack([M12, M22], dim=-1)
        M = torch.stack([M_row0, M_row1], dim=-2)

        h = self.m2 * self.l1 * self.r2 * torch.sin(q2)
        C11 = -2 * h * q2_dot
        C12 = -h * q2_dot
        C21 = h * q1_dot
        C22 = torch.zeros_like(C11)

        C_row0 = torch.stack([C11, C12], dim=-1)
        C_row1 = torch.stack([C21, C22], dim=-1)
        C = torch.stack([C_row0, C_row1], dim=-2)

        G1 = -(self.m1*self.r1 + self.m2*self.l1)*self.g*torch.sin(q1) \
             - self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G2 = -self.m2*self.g*self.r2*torch.sin(q1 + q2)
        G = torch.stack([G1, G2], dim=-1)

        return M, C, G

    def deriv(self, t, x, tau=None):
        """x: (..., 4); tau: (..., 2) or None. Returns (..., 4)."""
        x = x.to(device=self.device, dtype=self.dtype)
        if tau is None:
            tau = torch.zeros(x.shape[:-1] + (2,),
                              device=self.device, dtype=self.dtype)
        else:
            tau = tau.to(device=self.device, dtype=self.dtype)

        q1_dot = x[..., 1]
        q2_dot = x[..., 3]
        q_dot = torch.stack([q1_dot, q2_dot], dim=-1)

        tau_eff = tau - self.bv * q_dot

        M, C, G = self.compute_M_C_G(x)

        C_qdot = torch.einsum('...ij,...j->...i', C, q_dot)
        rhs = (tau_eff - C_qdot + G).unsqueeze(-1)
        q_ddot = torch.linalg.solve(M, rhs).squeeze(-1)

        return torch.stack(
            [q1_dot, q_ddot[..., 0], q2_dot, q_ddot[..., 1]],
            dim=-1,
        )

    def rk4_step(self, x, dt, tau_func=None):
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

    def step(self, x0, dt, tau_func=None, n_steps=10):
        x = x0.clone().to(device=self.device, dtype=self.dtype)
        dt = torch.as_tensor(dt, device=self.device, dtype=self.dtype)
        traj = [x.clone()]
        h = dt / n_steps
        for _ in range(n_steps):
            x = self.rk4_step(x, h, tau_func)
            traj.append(x.clone())
        return torch.stack(traj, dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    dyn = DoublePendulumDynamics(device=torch.device("cpu"))
    B = 8
    x_batch = torch.randn(B, 4, dtype=torch.float64) * 0.5
    tau_batch = torch.randn(B, 2, dtype=torch.float64) * 0.05

    dx_batch = dyn.deriv(0.0, x_batch, tau_batch)
    rk_batch = dyn.rk4_step(x_batch, 0.05, tau_func=lambda _x: tau_batch)

    dx_indiv = torch.stack([dyn.deriv(0.0, x_batch[i], tau_batch[i])
                            for i in range(B)], dim=0)
    rk_indiv = torch.stack([dyn.rk4_step(x_batch[i], 0.05,
                                         tau_func=lambda _x, i=i: tau_batch[i])
                            for i in range(B)], dim=0)

    err_deriv = (dx_batch - dx_indiv).abs().max().item()
    err_rk4   = (rk_batch - rk_indiv).abs().max().item()
    print(f"max |batched - per-element| (deriv):   {err_deriv:.3e}")
    print(f"max |batched - per-element| (rk4_step):{err_rk4:.3e}")
    assert err_deriv < 1e-12 and err_rk4 < 1e-12
    print("OK — MPC_dynamics batched matches unbatched to machine precision.")
