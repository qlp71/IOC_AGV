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
    delta: jnp.ndarray, gamma: jnp.ndarray, k2: float, k3: float, variant_idx: jnp.ndarray
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
#  IOC cost function (the objective for IGO)
# ═══════════════════════════════════════════════════════════════


@jax.jit
def cost_ioc(samples: jnp.ndarray, context: Context) -> jnp.ndarray:
    """IOC cost:  ∫‖u(t) – u_ref(t)‖² dt  +  boundary penalties.

    The B‑spline uses 3‑D control points  (xᵢ, yᵢ, tᵢ).  The ``tᵢ``
    values define a non‑uniform partition of the time horizon and
    serve as the parameter values for knot generation, so the
    B‑spline parameter **is** physical time.

    From the 1st & 2nd derivatives of  (x(t), y(t))  we reconstruct::

        θ   = atan2(ẏ, ẋ)
        v   = √(ẋ² + ẏ²)
        ω   = (ẋ·ÿ – ẏ·ẍ) / (ẋ² + ẏ²)

    The reference controls come from the closed‑loop law::

        v_ref = k₁ ρ cos γ
        ω_ref = ½ k₁ sin(2γ) + ω̃(δ, γ)
    """
    n_ctrl = context.n_ctrl
    deg = context.deg
    T = context.T

    # ---- build 3‑D control points:  [start] + opt_ctrl ----
    #  start  = (start_x,  start_y,  t=0)
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
    # v_sign = dx * jnp.cos(theta) + dy * jnp.sin(theta)
    # v = jnp.where(v > EPS, v * jnp.sign(v_sign), 0.0)  # restore sign to v, set to 0 if speed is very low
    omega = (dx * d2y - dy * d2x) / (speed_sq + EPS)
    v_omega = d2y * jnp.cos(theta) - d2x * jnp.sin(theta) # v·ω = ẋ·ÿ – ẏ·ẍ

    # ---- polar coordinates  (w.r.t. target = origin) ----
    rho, delta, gamma = _cart_to_polar_jax(x, y, theta)
    # rho = jnp.sqrt(x ** 2 + y ** 2)
    # delta = jnp.where(rho > EPS, jnp.arctan2(y, x) + jnp.pi, 0.0)
    # gamma = jnp.arctan2(jnp.sin(delta - theta), jnp.cos(delta - theta))

    # ---- control‑law reference  v_ref, ω_ref ----
    v_ref = context.k1 * rho * jnp.cos(gamma)
    w_tilde = _tilde_omega(delta, gamma, context.k2, context.k3,
                           context.ctrl_law_idx)
    omega_ref = 0.5 * context.k1 * jnp.sin(2 * gamma) + w_tilde
    v_omega_ref = v_ref * omega_ref

    # ---- squared‑deviation cost density ----
    cost_density = (v - v_ref) ** 2 + (v_omega - v_omega_ref) ** 2 + rho ** 2 * 0.01
    # cost_density = (v - v_ref) ** 2 + (omega - omega_ref) ** 2

    # 在 (-2, 0) 附近 0.5 的圆内增加一个 1000 的惩罚项，鼓励避开该区域
    penalty_region = (x - context.r1) ** 2 + (y - context.r2) ** 2 < 0.25
    penalty = jnp.where(penalty_region, 500.0 / (1 + (x - context.r1) ** 2 + (y - context.r2) ** 2) + 500.0, 0.0)
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
    dir_desired = jnp.array([jnp.cos(context.start_theta),
                             jnp.sin(context.start_theta)])
    tangent_penalty = 50.0 * jnp.sum((d0 / d0_norm - dir_desired) ** 2)

    # end position → end_pt
    end_penalty = 60.0 * jnp.sum((curve[-1] - context.end_pt) ** 2)

    return integral + tangent_penalty # + end_penalty


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
    ctrl_law_idx: int = 0,
    n_ctrl: int = 6,
    deg: int = 3,
    T: float = 5.0,
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
    r1, r2      : control weights in the IOC cost  R = diag(r₁, r₂).
    ctrl_law_idx: 0–3, selects ω̃ variant.
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
        ctrl_law_idx=ctrl_law_idx,
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
