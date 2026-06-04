from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Context:
    """
    Static context for IOC optimization.
    """

    # spline
    T: float
    n_ctrl: int
    deg: int

    # fixed first control point
    start_pt: jnp.ndarray  # (2,)
    start_theta: float
    end_pt: jnp.ndarray  # (2,)
    end_theta: float
    k1: float
    k2: float
    k3: float
    r1: float = 1.0
    r2: float = 1.0
    ctrl_law_idx: int = 0

    # pytree support
    def tree_flatten(self):
        children = (
            self.start_pt,
            self.start_theta,
            self.end_pt,
            self.end_theta,
            self.k1,
            self.k2,
            self.k3,
            self.r1,
            self.r2,
            self.ctrl_law_idx,
        )

        aux = (
            self.T,
            self.n_ctrl,
            self.deg,
        )

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):

        (
            T,
            n_ctrl,
            deg,
        ) = aux

        (
            start_pt,
            start_theta,
            end_pt,
            end_theta,
            k1,
            k2,
            k3,
            r1,
            r2,
            ctrl_law_idx,
        ) = children

        return cls(
            T=T,
            n_ctrl=n_ctrl,
            deg=deg,
            start_pt=start_pt,
            start_theta=start_theta,
            end_pt=end_pt,
            end_theta=end_theta,
            k1=k1,
            k2=k2,
            k3=k3,
            r1=r1,
            r2=r2,
            ctrl_law_idx=ctrl_law_idx,
        )
    
# ═══════════════════════════════════════════════════════════════
#  JAX B-spline utilities
# ═══════════════════════════════════════════════════════════════


def generate_knots(t_inp: jnp.ndarray, deg: int) -> jnp.ndarray:
    """Build a clamped knot vector from parameter values.

    Uses only vectorised operations — no ``lax.fori_loop`` with dynamic
    slices, so the function is fully JIT-compatible.

    Parameters
    ----------
    t_inp : (n,) array
        Parameter values, monotonically increasing, e.g. linspace(0, T, n).
    deg : int
        B-spline degree.

    Returns
    -------
    knots : (n + deg + 1,) array
        Clamped knot vector: first ``deg+1`` knots = t_inp[0],
        last ``deg+1`` knots = t_inp[-1].
    """
    n = len(t_inp)
    p = deg
    m = n + p + 1
    t_max = t_inp[-1]

    knots = jnp.zeros(m)
    # right clamped tail
    knots = knots.at[-p - 1:].set(t_max)

    # Interior knots via cumulative-sum sliding-window means.
    # For j = 1, …, n-p-1:  knots[p+j] = mean(t_inp[j : j+p]).
    if n - p > 1:
        cumsum = jnp.cumsum(t_inp)                # length n
        # pad so that cumsum_pad[i] = sum(t_inp[:i])
        cumsum_pad = jnp.pad(cumsum, (1, 0))      # length n+1
        # sums of windows [j, j+p)  for j = 1 … n-p-1
        win_sums = cumsum_pad[1 + p:n] - cumsum_pad[1:n - p]
        win_means = win_sums / p
        knots = knots.at[p + 1:n].set(win_means)

    return knots


def bspline_basis(deg: int, t: jnp.ndarray, knots: jnp.ndarray) -> jnp.ndarray:
    """Evaluate all B-spline basis functions of given degree at parameter *t*.

    Uses the Cox–de Boor recurrence.  For a clamped knot vector with *t* at the
    right endpoint the last basis function is set to 1 and all others to 0.

    Parameters
    ----------
    deg : int
        B-spline degree (non-negative).
    t : scalar array
        Parameter value.
    knots : (m,) array
        Knot vector.

    Returns
    -------
    N : (n_basis,) array
        Basis function values (one per control point).
    """
    n_knots = len(knots)
    n_basis = n_knots - deg - 1

    at_end = t >= knots[-1] - 1e-8

    # degree 0
    N = jnp.where((knots[:-1] <= t) & (t < knots[1:]), 1.0, 0.0)

    # Cox–de Boor recurrence
    for d in range(1, deg + 1):
        new_len = n_knots - d - 1

        left_denom = knots[d:n_knots - 1] - knots[:n_knots - d - 1]
        left = jnp.where(
            jnp.abs(left_denom) > 1e-12,
            (t - knots[:n_knots - d - 1]) / left_denom,
            0.0,
        )

        right_denom = knots[d + 1:] - knots[1:n_knots - d]
        right = jnp.where(
            jnp.abs(right_denom) > 1e-12,
            (knots[d + 1:] - t) / right_denom,
            0.0,
        )

        N = left * N[:new_len] + right * N[1:new_len + 1]

    end_result = jnp.zeros(n_basis).at[-1].set(1.0)
    return jnp.where(at_end, end_result, N)


def evaluate_curve(
    ctrl_pts: jnp.ndarray,
    deg: int,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a B-spline curve at multiple parameter values.

    Parameters
    ----------
    ctrl_pts : (n_ctrl, dim) array
        Control points.
    deg : int
        B-spline degree.
    knots : (m,) array
        Knot vector.
    t_eval : (n_eval,) array
        Parameter values at which to evaluate.

    Returns
    -------
    curve : (n_eval, dim) array
        Curve points.
    """

    def _eval_one(t):
        N = bspline_basis(deg, t, knots)
        return N @ ctrl_pts

    return jax.vmap(_eval_one)(t_eval)


def curve_derivative(
    ctrl_pts: jnp.ndarray,
    deg: int,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
) -> jnp.ndarray:
    """First derivative of a B-spline curve (analytic formula).

    The derivative of a degree-*p* B-spline is a degree-(*p*‑1) B-spline
    whose control points are scaled forward differences of the original
    control points.

    Parameters
    ----------
    ctrl_pts : (n_ctrl, dim) array
    deg : int
    knots : (m,) array
    t_eval : (n_eval,) array

    Returns
    -------
    deriv : (n_eval, dim) array
    """
    if deg < 1:
        return jnp.zeros_like(evaluate_curve(ctrl_pts, deg, knots, t_eval))

    dctrl, inner_knots = derivative_ctrl_pts(ctrl_pts, deg, knots)
    return evaluate_curve(dctrl, deg - 1, inner_knots, t_eval)


def derivative_ctrl_pts(
    ctrl_pts: jnp.ndarray, deg: int, knots: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Control points and knot vector of the first-derivative B-spline.

    Parameters
    ----------
    ctrl_pts : (n_ctrl, dim)
    deg : int  (must be ≥ 1)
    knots : (m,)

    Returns
    -------
    dctrl : (n_ctrl‑1, dim)   derivative control points
    dknots : (m‑2,)           derivative knot vector  = knots[1:-1]
    """
    p = deg
    denom = knots[p + 1:-1] - knots[1:-(p + 1)]
    safe_denom = jnp.where(jnp.abs(denom) > 1e-12, denom, 1e-12)
    dctrl = p * (ctrl_pts[1:] - ctrl_pts[:-1]) / safe_denom[:, None]
    dknots = knots[1:-1]
    return dctrl, dknots