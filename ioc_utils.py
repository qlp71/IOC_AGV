import jax
import jax.numpy as jnp
import numpy as np
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # avoid OOM on GPU with large batch sizes

from curves.bspline_utils_jax import (
    Context,
    bspline_basis,
    curve_derivative,
    derivative_ctrl_pts,
    evaluate_curve,
    generate_knots,
)
# from igo.blockwise_mgigo_jax import mmog_igo_optimizer_mpc as solver
from igo.MPCsolverM22 import mmog_igo_optimizer_mpc as solver

EPS = 1e-8


# ═══════════════════════════════════════════════════════════════
#  Polar-coordinate helpers (JAX)
# ═══════════════════════════════════════════════════════════════

@jax.jit
def _cart_to_polar_jax(x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray):
    """Cartesian → polar  (ρ, δ, γ), vectorised over leading axis."""
    rho = jnp.sqrt(x**2 + y**2)
    delta = jnp.where(rho > EPS, jnp.arctan2(y, x) + jnp.pi, 0.0)
    gamma = jnp.arctan2(jnp.sin(delta - theta), jnp.cos(delta - theta))
    return rho, delta, gamma


# ═══════════════════════════════════════════════════════════════
#  Four ω̃ variants (JAX)
# ═══════════════════════════════════════════════════════════════


def _sinc(x: jnp.ndarray) -> jnp.ndarray:
    """sin(x)/x with limit 1 at x=0."""
    return jnp.where(jnp.abs(x) > EPS, jnp.sin(x) / x, 1.0)


def _tilde_omega_1(delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float) -> jnp.ndarray:
    return k2 * jnp.sin(gamma) + k3 * _sinc(2 * gamma) * delta


def _tilde_omega_2(delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float) -> jnp.ndarray:
    denom = (1 + jnp.tan(gamma / 2) ** 2) ** 2
    return k2 * jnp.sin(gamma) + k3 * jnp.cos(gamma) / denom * delta


def _tilde_omega_3(delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float) -> jnp.ndarray:
    term = (1 + jnp.tan(delta / 2) ** 2) * jnp.tan(delta / 2)
    return k2 * jnp.sin(gamma) + 2 * k3 * _sinc(2 * gamma) * term


def _tilde_omega_4(delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float) -> jnp.ndarray:
    denom = (1 + jnp.tan(gamma / 2) ** 2) ** 2
    term = (1 + jnp.tan(delta / 2) ** 2) * jnp.tan(delta / 2)
    return k2 * jnp.sin(gamma) + 2 * k3 * jnp.cos(gamma) / denom * term


def _tilde_omega(
    delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float, variant_idx: int
) -> jnp.ndarray:
    """Dispatch to one of four ω̃ variants (JAX-traceable via take)."""
    w = jnp.stack([
        _tilde_omega_1(delta, gamma, k2, k3),
        _tilde_omega_2(delta, gamma, k2, k3),
        _tilde_omega_3(delta, gamma, k2, k3),
        _tilde_omega_4(delta, gamma, k2, k3),
    ])
    return jnp.take(w, variant_idx, axis=0)


# ═══════════════════════════════════════════════════════════════
#  Four CLF gradient functions (JAX)
# ═══════════════════════════════════════════════════════════════


def _clf_grad_0(rho: jnp.ndarray, delta: jnp.ndarray,
                gamma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CLF 0 — state space S (→ ω̃₁, sinc-type coupling).

    V₀ = ρ² + ½(δ² + γ² + 2)² − 2 + (δ + γ)²
    """
    d2g2p2 = delta**2 + gamma**2 + 2.0
    dV_drho = 2.0 * rho
    dV_ddelta = 2.0 * delta * d2g2p2 + 2.0 * (delta + gamma)
    dV_dgamma = 2.0 * gamma * d2g2p2 + 2.0 * (delta + gamma)
    return dV_drho, dV_ddelta, dV_dgamma


def _clf_grad_1(rho: jnp.ndarray, delta: jnp.ndarray,
                gamma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CLF 1 — state space S₁ (→ ω̃₂, γ fast-decay coupling).

    V₁ = ρ² + (δ + sin γ)² + 4 tan²(γ/2)
    """
    sin_g = jnp.sin(gamma)
    tan_g2 = jnp.clip(jnp.tan(gamma / 2.0), -3.0, 3.0)
    dV_drho = 2.0 * rho
    dV_ddelta = 2.0 * (delta + sin_g)
    dV_dgamma = (2.0 * (delta + sin_g) * jnp.cos(gamma)
                 + 4.0 * tan_g2 * (1.0 + tan_g2**2))
    return dV_drho, dV_ddelta, dV_dgamma


def _clf_grad_2(rho: jnp.ndarray, delta: jnp.ndarray,
                gamma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CLF 2 — state space S₂ (→ ω̃₃, δ-enhanced coupling).

    V₂ = ρ² + δ² + (γ + ½ arctan(4 tan(δ/2)))²
    """
    tan_d2 = jnp.clip(jnp.tan(delta / 2.0), -5.0, 5.0)
    phi = gamma + 0.5 * jnp.arctan(4.0 * tan_d2)
    dphi_ddelta = (1.0 + tan_d2**2) / (1.0 + 16.0 * tan_d2**2)
    dV_drho = 2.0 * rho
    dV_ddelta = 2.0 * delta + 2.0 * phi * dphi_ddelta
    dV_dgamma = 2.0 * phi
    return dV_drho, dV_ddelta, dV_dgamma


def _clf_grad_3(rho: jnp.ndarray, delta: jnp.ndarray,
                gamma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CLF 3 — state space S₃ (→ ω̃₄, comprehensive coupling).

    V₃ = ρ² + A³ − 1 + B²
    A  = 4 tan²(δ/2) + 4 tan²(γ/2) + 1
    B  = 2 tan(δ/2) + 2 tan(γ/2)

    tan values are clipped to  TAN_MAX  to avoid overflow when
    γ ≈ ±π  (trajectories far from the reference controller).
    """
    TAN_MAX = 1.5  # tan(0.983 rad ≈ 56.3° half-angle) — exact for |δ|,|γ| < 112.6°
    tan_d2 = jnp.clip(jnp.tan(delta / 2.0), -TAN_MAX, TAN_MAX)
    tan_g2 = jnp.clip(jnp.tan(gamma / 2.0), -TAN_MAX, TAN_MAX)
    A = 4.0 * tan_d2**2 + 4.0 * tan_g2**2 + 1.0
    B = 2.0 * tan_d2 + 2.0 * tan_g2
    sec2_d2 = 1.0 + tan_d2**2
    sec2_g2 = 1.0 + tan_g2**2
    dV_drho = 2.0 * rho
    dV_ddelta = 12.0 * A**2 * tan_d2 * sec2_d2 + 4.0 * B * sec2_d2
    dV_dgamma = 12.0 * A**2 * tan_g2 * sec2_g2 + 4.0 * B * sec2_g2
    return dV_drho, dV_ddelta, dV_dgamma


def _clf_grad(
    rho: jnp.ndarray, delta: jnp.ndarray, gamma: jnp.ndarray,
    variant_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dispatch to one of four CLF gradient functions via jax.lax.switch."""
    def _case_0(args):
        return _clf_grad_0(*args)
    def _case_1(args):
        return _clf_grad_1(*args)
    def _case_2(args):
        return _clf_grad_2(*args)
    def _case_3(args):
        return _clf_grad_3(*args)
    return jax.lax.switch(variant_idx, [_case_0, _case_1, _case_2, _case_3],
                          (rho, delta, gamma))


# ═══════════════════════════════════════════════════════════════
#  IOC cost function (the objective for IGO)
# ═══════════════════════════════════════════════════════════════


@jax.jit
def cost_ioc(samples: jnp.ndarray, context: Context) -> jnp.ndarray:
    """IOC cost based on inverse optimal control theory.

    Uses one of four CLF designs (selected by ``context.ctrl_law_idx``)
    to construct a meaningful cost function for which the reference
    control law  k(x) = [v_ref, ω_ref]  is inverse optimal.

    The cost integrand is::

        l(x) + uᵀ R u

    where, following Freeman & Kokotović (1996)::

        l(x) = −L_gV·k − ½ kᵀR k + ¼ L_gV R⁻¹ L_gVᵀ

    with  R = diag(w₁, w₂)  and  L_gV = ∇V·g.

    Obstacle penalty and boundary conditions are added on top.
    """
    n_ctrl = context.n_ctrl
    deg = context.deg
    T = context.T

    # ---- build 3‑D control points:  [start] + opt_ctrl ----
    opt_ctrl = samples.reshape((n_ctrl - 1, 3))             # (n_ctrl‑1, 3)
    start_ctrl = jnp.array([context.start_pt[0],
                            context.start_pt[1], 0.0])
    ctrl_pts = jnp.concatenate(
        [start_ctrl[None, :], opt_ctrl], axis=0)             # (n_ctrl, 3)

    # clamp first & last  t  so the domain is exactly [0, T]
    ctrl_pts = ctrl_pts.at[0, 2].set(0.0)
    ctrl_pts = ctrl_pts.at[-1, 2].set(T)

    # ---- non‑uniform knot vector from sorted t values ----
    t_inp = jnp.sort(ctrl_pts[:, 2])                        # (n_ctrl,)  monotonic
    knots = generate_knots(t_inp, deg)

    # 2‑D spatial control points
    ctrl_xy = ctrl_pts[:, :2]                                # (n_ctrl, 2)

    n_eval = 200
    t_eval = jnp.linspace(0.0, T, n_eval)

    # position, 1st & 2nd derivatives  (w.r.t. physical time)
    curve = evaluate_curve(ctrl_xy, deg, knots, t_eval)      # (n_eval, 2)
    d1 = curve_derivative(ctrl_xy, deg, knots, t_eval)       # (n_eval, 2)
    dctrl1, knots1 = derivative_ctrl_pts(ctrl_xy, deg, knots)
    d2 = curve_derivative(dctrl1, deg - 1, knots1, t_eval)    # (n_eval, 2)

    x, y = curve[:, 0], curve[:, 1]
    dx, dy = d1[:, 0], d1[:, 1]
    d2x, d2y = d2[:, 0], d2[:, 1]

    # ---- reconstruct actual  θ, v, ω  from curve geometry ----
    speed_sq = dx ** 2 + dy ** 2
    v = jnp.sqrt(speed_sq)
    theta = jnp.arctan2(dy, dx)
    omega = (dx * d2y - dy * d2x) / (speed_sq + EPS)

    # ---- polar coordinates  (w.r.t. target = origin) ----
    rho, delta, gamma = _cart_to_polar_jax(x, y, theta)

    # ---- control‑law reference  v_ref, ω_ref ----
    v_ref = context.k1 * rho * jnp.cos(gamma)
    w_tilde = _tilde_omega(delta, gamma, context.k2, context.k3,
                           context.ctrl_law_idx)
    omega_ref = 0.5 * context.k1 * jnp.sin(2 * gamma) + w_tilde
    v_ref = jnp.where(context.is_sat, jnp.tanh(v_ref), v_ref)
    omega_ref = jnp.where(context.is_sat, jnp.tanh(omega_ref), omega_ref)
    # ---- CLF gradient for the selected variant ----
    dV_drho, dV_ddelta, dV_dgamma = _clf_grad(
        rho, delta, gamma, context.ctrl_law_idx)

    # ---- Lie derivative  L_gV = ∇V · g  ----
    #  Use  ρ_safe = max(ρ, RHO_MIN)  to bound the  1/ρ  singularity.
    # RHO_MIN = 0.01
    # rho_safe = jnp.maximum(rho, RHO_MIN)
    LgV1 = (-dV_drho * jnp.cos(gamma) + (dV_ddelta + dV_dgamma) * jnp.sin(gamma) / rho)
    LgV2 = -dV_dgamma

    # ---- Lie derivative  L_fV = ∇V · f  ----
    # LfV = dV_drho * v * jnp.cos(gamma) + dV_ddelta * omega + dV_dgamma * omega
    #     = 0

    # ---- IOC control weights  R = diag(w₁, w₂) ----
    w1 = context.w1
    w2 = context.w2

    # l(x, u) = - ∇V(x)^T (f(x) + g(x) u) + (u-k)^T R (u-k)
    #  since LfV = 0 for the reference trajectory, the CLF‑based cost reduces to
    #         =  - LgV1 * v_ref - LgV2 * omega_ref + w1 * (v - v_ref)**2 + w2 * (omega - omega_ref)**2
    Vdot_ref = LgV1 * v_ref + LgV2 * omega_ref
    cost_density =  -Vdot_ref + w1 * (v - v_ref)**2 + w2 * (omega - omega_ref)**2
    cost_density = w1 * (v - v_ref)**2 + w2 * (omega - omega_ref)**2
    # penalise CLF increase (Vdot_ref > 0) with a quadratic penalty
    cost_density += 100.0 * jnp.where(Vdot_ref > 0, Vdot_ref**2 + 1, 0.0)
# ---------------------------------------------------------------------------------
    # # ---- state-dependent cost  l(x)  (Freeman & Kokotović, 1996) ----
    # #  l(x) = −L_gV·k − ½ kᵀR k + ¼ L_gV R⁻¹ L_gVᵀ
    # l_x = (-(LgV1 * v_ref + LgV2 * omega_ref)
    #        - 0.5 * (w1 * v_ref**2 + w2 * omega_ref**2)
    #        + 0.25 * (LgV1**2 / w1_safe + LgV2**2 / w2_safe))

    # #  Enforce  l(x) ≥ 0  (required by FK theorem for meaningful cost).
    # l_x = jnp.maximum(l_x, 0.0)

    # # ---- total integrand  l(x) + uᵀ R u ----
    # cost_density = l_x + w1 * v**2 + w2 * omega**2
# ---------------------------------------------------------------------------------

    # ---- obstacle penalty (preserved from original) ----
    penalty_region = (x - context.r1) ** 2 + (y - context.r2) ** 2 < 0.25
    penalty = jnp.where(penalty_region,
                        500.0 / (1.0 + (x - context.r1)**2 + (y - context.r2)**2) + 500.0,
                        0.0)
    cost_density += penalty

    # ---- trapezoidal integration ----
    dt = T / (n_eval - 1)
    integral = dt * (
        0.5 * cost_density[0]
        + jnp.sum(cost_density[1:-1])
        + 0.5 * cost_density[-1]
    )

    # ════ boundary penalties ════

    # initial tangent → start_theta
    d0 = d1[0]                                              # (2,)
    d0_norm = jnp.sqrt(jnp.sum(d0 ** 2) + EPS)
    d0_v1 = jnp.array([jnp.cos(context.start_theta), jnp.sin(context.start_theta)])
    # d0_v2 = jnp.array([jnp.cos(context.start_theta), -jnp.sin(context.start_theta)])
    tangent_penalty = 50.0 * jnp.sum((d0 / d0_norm - d0_v1) ** 2)
    # tangent_penalty2 = 50.0 * jnp.sum((d0 / d0_norm - d0_v2) ** 2)
    # tangent_penalty = jnp.minimum(tangent_penalty1, tangent_penalty2)
    # end position → end_pt
    # end_penalty = 60.0 * jnp.sum((curve[-1] - context.end_pt) ** 2)

    return integral + tangent_penalty  # + end_penalty


# ═══════════════════════════════════════════════════════════════
#  Optimal-curve entry point
# ═══════════════════════════════════════════════════════════════


def optimal_curve(
    start_point: tuple[float, float, float],
    end_point: tuple[float, float, float],
    k1: float = 1.0,
    k2: float = 2.0,
    k3: float = 0.5,
    r1: float = 1.0,
    r2: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    ctrl_law_idx: int = 0,
    is_sat: bool = True,
    n_ctrl: int = 6,
    deg: int = 3,
    T: float = 6.0,
    tot_iters: int = 600,
    b_samples: int = 200,
    b0_elite: int = 80,
    dt_step: float = 0.15,
    t_restart: int = 200,
    n_eval_curve: int = 500,
    seed: int = 0,
) -> dict:
    """Compute the optimal B-spline curve via inverse optimal control.

    Parameters
    ----------
    start_point : (x, y, θ)
    end_point   : (x, y, θ)
    k1, k2, k3  : control-law gains.
    r1, r2      : obstacle-centre coordinates  (x_obs, y_obs).
    w1, w2      : IOC control weights  R = diag(w₁, w₂),  must be > 0.
    ctrl_law_idx: 0–3, selects ω̃ variant and corresponding CLF.
    n_ctrl      : number of B-spline control points.
    deg         : B-spline degree.
    T           : time horizon.
    tot_iters   : IGO iterations.
    b_samples   : samples per iteration.
    b0_elite    : elite samples.
    dt_step     : IGO step size α_t.
    t_restart   : reset period for mixture weights.
    n_eval_curve: number of evaluation points for the returned trajectory.
    seed        : PRNG seed.

    Returns
    -------
    dict with keys:
        t_eval      : (n_eval,)  time values.
        traj        : (n_eval, 3)  x, y, θ  along the optimized curve.
        ctrl_pts    : (n_ctrl, 3)  optimised B-spline control points.
        cost        : final IOC cost value.
    """
    ctx = Context(
        T=T,
        n_ctrl=n_ctrl,
        deg=deg,
        start_pt=jnp.array(start_point[:2]),
        start_theta=float(start_point[2]),
        end_pt=jnp.array(end_point[:2]),
        end_theta=float(end_point[2]),
        k1=k1,
        k2=k2,
        k3=k3,
        r1=r1,
        r2=r2,
        w1=w1,
        w2=w2,
        ctrl_law_idx=ctrl_law_idx,
        is_sat=is_sat,
    )

    key_solver = jax.random.PRNGKey(seed)

    # IGO optimises n_ctrl‑1 control points (all except the fixed first)
    D = 3       # control‑point dimension  (x, y, t)
    M = n_ctrl - 1
    dims = (D,) * M
    k_components = 3
    dim_max = max(dims)

    # ---- initial control points (straight line + uniform t) ----
    # Shape: (M, K, D_max) = (n_ctrl‑1, k_components, 3)
    initial_mu = jnp.zeros((M, k_components, dim_max), dtype=jnp.float32)
    for i in range(M):
        alpha = (i + 1) / M
        # (x, y): straight line from start to end
        # t:     uniform partition of [0, T]
        val = jnp.array([
            (1 - alpha) * ctx.start_pt[0] + alpha * ctx.end_pt[0],
            (1 - alpha) * ctx.start_pt[1] + alpha * ctx.end_pt[1],
            alpha * T,
        ])
        initial_mu = initial_mu.at[i, 0].set(val)

    initial_l_inv = jnp.tile(
        jnp.eye(dim_max, dtype=jnp.float32)[None, None, :, :],
        (M, k_components, 1, 1),
    ) * 1.0
    initial_v = jnp.zeros((M, k_components - 1), dtype=jnp.float32)

    print(f"[optimal_curve] M={M}, dims={dims}, K={k_components}, D_max={dim_max}")
    print(f"[optimal_curve] initial_mu shape={initial_mu.shape}")
    print(f"[optimal_curve] calling solver...")

    final_mu, final_l_inv, final_pi, final_v = solver(
        key_solver,
        tot_iters,
        dt_step,
        M,
        k_components,
        b_samples,
        b0_elite,
        dims,
        t_restart,
        cost_ioc,
        initial_mu,
        initial_l_inv,
        initial_v,
        ctx,
    )

    # ---- extract best control points ----
    opt_ctrl = final_mu[:, 0, :]                            # (n_ctrl‑1, 3)

    # prepend fixed first control point  (start_x, start_y, t=0)
    start_ctrl = jnp.array([ctx.start_pt[0], ctx.start_pt[1], 0.0])
    ctrl_pts_full = jnp.concatenate(
        [start_ctrl[None, :], opt_ctrl], axis=0)             # (n_ctrl, 3)
    # clamp last t to T
    ctrl_pts_full = ctrl_pts_full.at[-1, 2].set(T)

    # ---- build final trajectory ----
    t_inp = jnp.sort(ctrl_pts_full[:, 2])
    knots = generate_knots(t_inp, deg)
    t_eval = jnp.linspace(0.0, T, n_eval_curve)
    traj_xy = evaluate_curve(ctrl_pts_full[:, :2], deg, knots, t_eval)

    # derive θ from tangent
    d1 = curve_derivative(ctrl_pts_full[:, :2], deg, knots, t_eval)
    theta_traj = jnp.arctan2(d1[:, 1], d1[:, 0])
    traj = jnp.column_stack([traj_xy, theta_traj])            # (n_eval, 3)

    # ---- final cost ----
    flat = opt_ctrl.reshape(-1)                               # 3·(n_ctrl‑1)
    final_cost = float(cost_ioc(flat, ctx))

    return {
        "t_eval": np.array(t_eval),
        "traj": np.array(traj),
        "ctrl_pts": np.array(ctrl_pts_full),
        "cost": final_cost,
    }
