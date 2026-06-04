import numpy as np


EPS = 1e-12

# ═══════════════════════════════════════════════════════════════════════════
#  B-spline utilities
# ═══════════════════════════════════════════════════════════════════════════

def _bspline_basis_all(deg: int, t: float, knots: np.ndarray) -> np.ndarray:
    """Evaluate all deg-*deg* B-spline basis functions at *t*.

    Returns an array of length ``len(knots) - deg - 1`` (one per control point).
    """
    n_knots = len(knots)
    n_ctrl = n_knots - deg - 1

    # right endpoint: the clamped B-spline interpolates the last control point
    if t >= knots[-1] - EPS:
        result = np.zeros(n_ctrl)
        result[-1] = 1.0
        return result

    n0 = n_knots - 1  # number of deg-0 basis functions
    N = np.zeros((deg + 1, n_knots - 1))

    # deg 0
    for i in range(n0):
        N[0, i] = 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    # recurrence
    for k in range(1, deg + 1):
        for i in range(n0 - k):
            d_left = knots[i + k] - knots[i]
            d_right = knots[i + k + 1] - knots[i + 1]
            left = (t - knots[i]) / d_left if d_left > EPS else 0.0
            right = (knots[i + k + 1] - t) / d_right if d_right > EPS else 0.0
            N[k, i] = left * N[k - 1, i] + right * N[k - 1, i + 1]

    return N[deg, :n_ctrl]


def generate_knots(t_inp: np.ndarray, deg: int) -> np.ndarray:
    """Build a clamped knot vector from parameter values *t_inp* (averaging)."""
    u = np.sort(np.asarray(t_inp, dtype=float))
    n = len(u)
    p = deg
    knots = np.zeros(n + p + 1)
    knots[: p + 1] = u[0]
    knots[-(p + 1) :] = u[-1]
    if n > p:
        for j in range(1, n - p):
            knots[p + j] = np.mean(u[j : j + p])
    else:
        knots[p + 1 : -p - 1] = np.linspace(u[0], u[-1], n - p)
    return knots


def evaluate_curve(
    ctrl_pts: np.ndarray, deg: int, knots: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    """Evaluate B-spline curve at each parameter in *t_eval*.

    *ctrl_pts* – shape (n_ctrl, dim)
    Returns shape (len(t_eval), dim).
    """
    t_eval = np.atleast_1d(t_eval)
    n_ctrl, dim = ctrl_pts.shape
    result = np.empty((len(t_eval), dim))
    n_basis = len(knots) - deg - 1
    for idx, t in enumerate(t_eval):
        N = _bspline_basis_all(deg, t, knots)
        n = min(len(N), n_ctrl)
        result[idx] = N[:n] @ ctrl_pts[:n]
    return result


def curve_derivative(
    ctrl_pts: np.ndarray, deg: int, knots: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    """First derivative of B-spline curve (numerical)."""
    h = 1e-6
    if len(t_eval) > 1:
        f_plus = evaluate_curve(ctrl_pts, deg, knots, t_eval + h)
        f_minus = evaluate_curve(ctrl_pts, deg, knots, t_eval - h)
        return (f_plus - f_minus) / (2 * h)
    # single point
    f_plus = evaluate_curve(ctrl_pts, deg, knots, np.array([t_eval[0] + h]))
    f_minus = evaluate_curve(ctrl_pts, deg, knots, np.array([t_eval[0] - h]))
    return (f_plus - f_minus) / (2 * h)
