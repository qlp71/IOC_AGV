#!/usr/bin/env python3
"""
MRS (Multi-Robot System) B-spline Trajectory Visualization.

Interactive visualization of an MRS center trajectory (B-spline) and the
resulting robot trajectories.  Control points are draggable in the x-y plane;
parameters can be tuned via text boxes and a t_inp range axis.

Usage:
    python mrs_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib

matplotlib.use("QtAgg")

from curves.bspline_utils import _bspline_basis_all, generate_knots
# ═══════════════════════════════════════════════════════════════════════════
#  Main interactive class
# ═══════════════════════════════════════════════════════════════════════════

ROBOT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

class MRSVisualizer:
    """Interactive MRS trajectory visualizer with B-spline fitting."""

    def __init__(self):
        # ── parameters ──
        self.n_pts = 4
        self.deg = 2
        self.T = 5.0
        self.N_robots = 2

        # t_inp: non-uniform time points
        self.t_inp = np.linspace(0, self.T, self.n_pts)

        # control points (x, y, theta) – shape (n_pts, 3)
        self.ctrl_pts = self._default_ctrl_pts()

        # robot relative offsets (x_r, y_r) – shape (N_robots, 2)
        self.r_offset = np.array([[1.0, 0.5], [-0.5, 1.0]])

        # dragging state
        self._dragging_ctrl_idx = None       # index of ctrl pt being dragged
        self._dragging_ctrl_theta = False    # dragging arrow tip → adjust θ
        self._dragging_tinp_idx = None
        self._dragging_robot_idx = None

        # cache for B-spline basis matrices (rebuild only when knots change)
        self._knots_hash = None
        self._init_done = False

        # evaluation samples
        self._t_dense = np.linspace(0, self.T, 100)

        # ── build UI ──
        self._build_ui()
        self._update_all()
        self._update_tinp_markers()
        self._init_done = True

    # ── default control points (figure-8) ─────────────────────

    def _default_ctrl_pts(self) -> np.ndarray:
        t = np.linspace(0, self.T, self.n_pts)
        x = np.linspace(0, 5.0, self.n_pts) - 5.0
        y = np.linspace(0, 1.0, self.n_pts) - 1.0 + 2.0 * np.sin(2 * np.pi * t / self.T)
        theta = np.sin(2 * np.pi * t / self.T) * np.pi / 4
        return np.column_stack([x, y, theta])

    # ═══════════════════════════════════════════════════════════
    #  UI layout
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        self.fig = plt.figure("MRS B-spline", figsize=(16, 9))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("MRS — B-spline Trajectory")

        # -- left: x-y trajectory (large) --
        self.ax_traj = self.fig.add_axes((0.075, 0.15, 0.4, 0.85))
        self.ax_traj.set_aspect("equal")
        self.ax_traj.set_xlabel("x")
        self.ax_traj.set_ylabel("y")
        self.ax_traj.set_title("MRS centre trajectory (B-spline) · drag control pts",
                               fontsize=10)
        self.ax_traj.grid(True, alpha=0.3)

        # -- left-bottom: controls --
        self._build_controls()

        # -- right: 6 time-series plots --
        self._build_time_series()

    def _build_controls(self):
        """Text boxes for n_pts, deg, T, N; t_inp axis."""
        bw, bh = 0.06, 0.025
        x0, y_base = 0.075, 0.15

        # row 1: n_pts, deg, T, N
        labels = ["n_pts", "deg", "T", "N"]
        initials = [str(self.n_pts), str(self.deg),
                    str(self.T), str(self.N_robots)]
        callbacks = [self._on_num_pts, self._on_deg,
                     self._on_T, self._on_N]
        self._textboxes = {}
        for i, (label, init, cb) in enumerate(zip(labels, initials, callbacks)):
            x = x0 + i * 0.09 + 0.03
            ax = self.fig.add_axes((x, y_base + 0.065, bw, bh))
            tb = TextBox(ax, label + "  ", initial=init, textalignment="right")
            tb.on_submit(cb)
            self._textboxes[label] = tb

        # row 2: t_inp range axis label
        ax_tlabel = self.fig.add_axes((x0, y_base + 0.025, 0.36, 0.025))
        ax_tlabel.axis("off")
        ax_tlabel.text(0.5, 0.5, "t_inp  (drag markers on the axis below)",
                       ha="center", va="center", fontsize=9)

        # t_inp axis
        self.ax_tinp = self.fig.add_axes((x0, y_base, 0.36, 0.03))
        self.ax_tinp.set_xlim(-0.02 * self.T, self.T * 1.02)
        self.ax_tinp.set_ylim(-1.2, 1.2)
        self.ax_tinp.set_yticks([])
        self.ax_tinp.set_xlabel("t [s]", fontsize=8)
        self.ax_tinp.axhline(0, color="gray", lw=1.5, zorder=0)
        self.ax_tinp.axvline(0, color="black", lw=1.2, zorder=0)
        self._tinp_T_line = self.ax_tinp.axvline(
            self.T, color="black", lw=1.2, zorder=0)

        self._tinp_markers = None
        self._tinp_lines = []

    def _build_time_series(self):
        """Right column: t-x_c, t-y_c, t-theta_c, t-x_i, t-y_i, t-theta_i."""
        y_axis = ["$x_c(t)$", "$y_c(t)$", r"$\theta_c(t)$",
                  "$x_i(t)$", "$y_i(t)$", r"$\theta_i(t)$"]
        self._ts_axes = []
        self._ts_lines = []
        self._ts_scatters = []

        x0, y_top = 0.55, 0.85
        w, h = 0.4, 0.12
        gap = 0.03
        n = len(y_axis)

        for i, y_ in enumerate(y_axis):
            y = y_top - i * (h + gap)
            ax = self.fig.add_axes((x0, y, w, h))
            ax.set_ylabel(y_, fontsize=9)
            ax.set_xlabel("t [s]", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            self._ts_axes.append(ax)

    # ═══════════════════════════════════════════════════════════
    #  Compute trajectories
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _build_basis(deg: int, knots: np.ndarray,
                     t_eval: np.ndarray) -> np.ndarray:
        """Build basis matrix of shape (len(t_eval), n_ctrl)."""
        n_ctrl = len(knots) - deg - 1
        N = np.empty((len(t_eval), n_ctrl))
        for idx, t in enumerate(t_eval):
            N[idx] = _bspline_basis_all(deg, t, knots)
        return N

    def _compute(self):
        """Compute centre & robot trajectories.  Stores results on self."""
        knots = generate_knots(self.t_inp, self.deg)
        kh = hash(knots.tobytes())

        # rebuild basis matrices only when knots change
        if kh != self._knots_hash:
            self._N_dense = self._build_basis(self.deg, knots, self._t_dense)
            self._N_tinp = self._build_basis(self.deg, knots, self.t_inp)
            self._knots_hash = kh

        # centre trajectory (fast: BLAS matmul → one-shot, no Python loop)
        centre = self._N_dense @ self.ctrl_pts
        centre_tinp = self._N_tinp @ self.ctrl_pts
        x_c, y_c, theta_c = centre[:, 0], centre[:, 1], centre[:, 2]

        # derivative via numpy gradient (no extra basis evaluations)
        d_centre = np.gradient(centre, self._t_dense, axis=0)
        dx_c, dy_c, dtheta_c = d_centre[:, 0], d_centre[:, 1], d_centre[:, 2]

        # robot trajectories
        robot_traj = []
        robot_theta = []
        for i in range(self.N_robots):
            xr, yr = self.r_offset[i]
            cos_t = np.cos(theta_c)
            sin_t = np.sin(theta_c)
            x_i = x_c + cos_t * xr - sin_t * yr
            y_i = y_c + sin_t * xr + cos_t * yr
            robot_traj.append(np.column_stack([x_i, y_i]))

            cos_t_pi2 = np.cos(theta_c + np.pi / 2)
            sin_t_pi2 = np.sin(theta_c + np.pi / 2)
            dx_i = dx_c + (cos_t_pi2 * xr - sin_t_pi2 * yr) * dtheta_c
            dy_i = dy_c + (sin_t_pi2 * xr + cos_t_pi2 * yr) * dtheta_c
            robot_theta.append(np.arctan2(dy_i, dx_i))

        # robot positions at t_inp
        robot_tinp = []
        for i in range(self.N_robots):
            xr, yr = self.r_offset[i]
            cos_t = np.cos(centre_tinp[:, 2])
            sin_t = np.sin(centre_tinp[:, 2])
            x_i = centre_tinp[:, 0] + cos_t * xr - sin_t * yr
            y_i = centre_tinp[:, 1] + sin_t * xr + cos_t * yr
            robot_tinp.append(np.column_stack([x_i, y_i]))

        self._cache = {
            "knots": knots,
            "centre": centre,
            "centre_tinp": centre_tinp,
            "x_c": x_c, "y_c": y_c, "theta_c": theta_c,
            "d_centre": d_centre,
            "robot_traj": robot_traj,
            "robot_theta": robot_theta,
            "robot_tinp": robot_tinp,
        }

    # ═══════════════════════════════════════════════════════════
    #  Draw / update artists
    # ═══════════════════════════════════════════════════════════

    def _update_all(self):
        self._compute()
        self._update_traj_plot()
        self._update_time_series_plots()
        self.fig.canvas.draw_idle()

    def _update_traj_plot(self):
        ax = self.ax_traj
        c = self._cache
        pts = self.ctrl_pts

        need_recreate = (
            not hasattr(self, "_traj_ctrl_arrows")
            or len(self._traj_ctrl_arrows) != self.n_pts
            or len(getattr(self, "_traj_robot_lines", [])) != self.N_robots
        )

        if need_recreate:
            self._remove_traj_artists()
            self._create_traj_artists()

        # --- in-place data updates ---
        ci = c["centre_tinp"]

        self._traj_centre_line.set_data(c["x_c"], c["y_c"])
        self._traj_ctrl_scatter.set_offsets(pts[:, :2])
        self._traj_interp_scatter.set_offsets(ci[:, :2])

        for i in range(self.n_pts):
            x, y, th = pts[i]
            dx = 0.35 * np.cos(th)
            dy = 0.35 * np.sin(th)
            self._traj_ctrl_arrows[i].set_data([x, x + dx], [y, y + dy])

        for i in range(self.N_robots):
            rt = c["robot_traj"][i]
            ri = c["robot_tinp"][i]
            self._traj_robot_lines[i].set_data(rt[:, 0], rt[:, 1])
            self._traj_robot_scatter[i].set_offsets(ri)
            for j in range(self.n_pts):
                idx = i * self.n_pts + j
                self._traj_robot_conn_lines[idx].set_data(
                    [ci[j, 0], ri[j, 0]], [ci[j, 1], ri[j, 1]])

        for i in range(self.N_robots):
            xr, yr = self.r_offset[i]
            x0, y0, th0 = c["centre"][0]
            x_abs = x0 + np.cos(th0) * xr - np.sin(th0) * yr
            y_abs = y0 + np.sin(th0) * xr + np.cos(th0) * yr
            getattr(self, f"_robot_offset_dot_{i}").set_center((x_abs, y_abs))

        # auto-range
        all_x = np.concatenate([c["x_c"]] +
                               [c["robot_traj"][i][:, 0] for i in range(self.N_robots)])
        all_y = np.concatenate([c["y_c"]] +
                               [c["robot_traj"][i][:, 1] for i in range(self.N_robots)])
        m = 1.5
        ax.set_xlim(all_x.min() - m, all_x.max() + m)
        ax.set_ylim(all_y.min() - m, all_y.max() + m)

    def _remove_traj_artists(self):
        for attr in ("_traj_centre_line", "_traj_ctrl_scatter",
                     "_traj_interp_scatter"):
            val = getattr(self, attr, None)
            if val is not None:
                val.remove()
        for attr in ("_traj_ctrl_arrows", "_traj_robot_lines",
                     "_traj_robot_scatter", "_traj_robot_conn_lines"):
            for a in getattr(self, attr, []):
                a.remove()
        for i in range(getattr(self, "N_robots", 0)):
            name = f"_robot_offset_dot_{i}"
            dot = getattr(self, name, None)
            if dot is not None:
                dot.remove()

    def _create_traj_artists(self):
        ax = self.ax_traj
        c = self._cache
        pts = self.ctrl_pts
        ci = c["centre_tinp"]

        (self._traj_centre_line,) = ax.plot(
            c["x_c"], c["y_c"], "b-", lw=2.5, label="MRS centre", zorder=4)

        self._traj_ctrl_scatter = ax.scatter(
            pts[:, 0], pts[:, 1], c="red", s=70, zorder=8,
            picker=8, marker="s", label="ctrl pts")

        self._traj_ctrl_arrows = []
        for i in range(self.n_pts):
            x, y, th = pts[i]
            dx = 0.35 * np.cos(th)
            dy = 0.35 * np.sin(th)
            arr = Line2D([x, x + dx], [y, y + dy], color="darkred",
                         lw=2.0, zorder=9, picker=5)
            ax.add_line(arr)
            self._traj_ctrl_arrows.append(arr)

        self._traj_interp_scatter = ax.scatter(
            ci[:, 0], ci[:, 1], c="blue", s=40, zorder=7,
            marker="o", label="interp pts",
            edgecolors="darkblue", linewidths=1)

        self._traj_robot_lines = []
        self._traj_robot_scatter = []
        self._traj_robot_conn_lines = []
        for i in range(self.N_robots):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            rt = c["robot_traj"][i]
            (l,) = ax.plot(rt[:, 0], rt[:, 1], color=color, lw=1.5,
                          ls="--", label=f"Robot {i+1}", zorder=5)
            self._traj_robot_lines.append(l)

            ri = c["robot_tinp"][i]
            s = ax.scatter(ri[:, 0], ri[:, 1], color=color, s=25, zorder=7,
                          marker="o")
            self._traj_robot_scatter.append(s)

            for j in range(self.n_pts):
                cl = Line2D([ci[j, 0], ri[j, 0]], [ci[j, 1], ri[j, 1]],
                           color=color, lw=0.5, alpha=0.4, ls=":", zorder=3)
                ax.add_line(cl)
                self._traj_robot_conn_lines.append(cl)

        for i in range(self.N_robots):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            xr, yr = self.r_offset[i]
            x0, y0, th0 = c["centre"][0]
            x_abs = x0 + np.cos(th0) * xr - np.sin(th0) * yr
            y_abs = y0 + np.sin(th0) * xr + np.cos(th0) * yr
            dot = Circle((x_abs, y_abs), 0.18, fc=color, ec="black",
                        lw=1.5, zorder=12, picker=6,
                        label=f"Robot {i+1} offset")
            ax.add_patch(dot)
            setattr(self, f"_robot_offset_dot_{i}", dot)

        ax.legend(loc="upper right", fontsize=7).set_zorder(20)

    def _update_time_series_plots(self):
        c = self._cache
        t = self._t_dense
        t_inp = self.t_inp
        ci = c["centre_tinp"]

        # check if we need to (re)create artists
        need_create = (
            not hasattr(self, "_ts_lines")
            or len(self._ts_lines) != 6
        )
        if not need_create:
            expected_lines = [1, 1, 1] + [self.N_robots] * 3
            need_create = any(
                len(self._ts_lines[i]) != expected_lines[i] for i in range(6))

        if need_create:
            self._create_ts_artists()

        # --- in-place data updates ---
        # axis 0: t-x_c
        self._ts_lines[0][0].set_data(t, c["x_c"])
        self._ts_scatters[0][0].set_offsets(np.column_stack([t_inp, ci[:, 0]]))
        self._ts_scatters[0][1].set_offsets(np.column_stack([t_inp, self.ctrl_pts[:, 0]]))

        # axis 1: t-y_c
        self._ts_lines[1][0].set_data(t, c["y_c"])
        self._ts_scatters[1][0].set_offsets(np.column_stack([t_inp, ci[:, 1]]))
        self._ts_scatters[1][1].set_offsets(np.column_stack([t_inp, self.ctrl_pts[:, 1]]))

        # axis 2: t-theta_c
        self._ts_lines[2][0].set_data(t, c["theta_c"])
        self._ts_scatters[2][0].set_offsets(np.column_stack([t_inp, ci[:, 2]]))
        self._ts_scatters[2][1].set_offsets(np.column_stack([t_inp, self.ctrl_pts[:, 2]]))

        # axes 3-5: robot trajectories
        for ax_i, field in enumerate(["x", "y", "theta"], start=3):
            for r_i in range(self.N_robots):
                if field == "theta":
                    data = c["robot_theta"][r_i]
                else:
                    data = c["robot_traj"][r_i][:, 0 if field == "x" else 1]
                self._ts_lines[ax_i][r_i].set_data(t, data)
                if field != "theta":
                    rti = c["robot_tinp"][r_i]
                    col = 0 if field == "x" else 1
                    self._ts_scatters[ax_i][r_i].set_offsets(
                        np.column_stack([t_inp, rti[:, col]]))

        # auto-scale each ts axis (skip on first call since data already set)
        for ax in self._ts_axes:
            ax.relim()
            ax.autoscale_view()

    def _create_ts_artists(self):
        """Create time-series artists (once, then updated in-place)."""
        c = self._cache
        t = self._t_dense
        t_inp = self.t_inp
        ci = c["centre_tinp"]

        # clear any existing
        for attr in ("_ts_lines", "_ts_scatters"):
            for group in getattr(self, attr, []):
                for a in group:
                    a.remove()

        self._ts_lines = [[] for _ in range(6)]
        self._ts_scatters = [[] for _ in range(6)]

        # axis 0: t-x_c
        (l,) = self._ts_axes[0].plot(t, c["x_c"], "-", color="tab:blue", lw=1.2, label="$x_c$")
        self._ts_lines[0].append(l)
        self._ts_scatters[0].append(
            self._ts_axes[0].scatter(t_inp, ci[:, 0], c="blue", s=25, zorder=5))
        self._ts_scatters[0].append(
            self._ts_axes[0].scatter(t_inp, self.ctrl_pts[:, 0], c="red", s=25, marker="s", zorder=5))

        # axis 1: t-y_c
        (l,) = self._ts_axes[1].plot(t, c["y_c"], "-", color="tab:orange", lw=1.2, label="$y_c$")
        self._ts_lines[1].append(l)
        self._ts_scatters[1].append(
            self._ts_axes[1].scatter(t_inp, ci[:, 1], c="blue", s=25, zorder=5))
        self._ts_scatters[1].append(
            self._ts_axes[1].scatter(t_inp, self.ctrl_pts[:, 1], c="red", s=25, marker="s", zorder=5))

        # axis 2: t-theta_c
        (l,) = self._ts_axes[2].plot(t, c["theta_c"], "-", color="tab:green", lw=1.2, label=r"$\theta_c$")
        self._ts_lines[2].append(l)
        self._ts_scatters[2].append(
            self._ts_axes[2].scatter(t_inp, ci[:, 2], c="blue", s=25, zorder=5))
        self._ts_scatters[2].append(
            self._ts_axes[2].scatter(t_inp, self.ctrl_pts[:, 2], c="red", s=25, marker="s", zorder=5))

        # axes 3-5: robot trajectories
        field_labels = ["$x_i$", "$y_i$", r"$\theta_i$"]
        for ax_i, flabel in enumerate(field_labels, start=3):
            ax = self._ts_axes[ax_i]
            for r_i in range(self.N_robots):
                color = ROBOT_COLORS[r_i % len(ROBOT_COLORS)]
                if ax_i < 5:  # x or y
                    col = 0 if ax_i == 3 else 1
                    data = c["robot_traj"][r_i][:, col]
                    rti = c["robot_tinp"][r_i][:, col]
                else:
                    data = c["robot_theta"][r_i]
                    rti = None

                (l,) = ax.plot(t, data, "-", color=color, lw=1.2,
                              label=f"R{r_i+1} {flabel}")
                self._ts_lines[ax_i].append(l)
                if rti is not None:
                    self._ts_scatters[ax_i].append(
                        ax.scatter(t_inp, rti, c=color, s=20, zorder=5))

        for ax in self._ts_axes:
            ax.legend(fontsize=6, loc="upper right")

    def _update_tinp_markers(self):
        """Update draggable markers on the t_inp axis (in-place when possible)."""
        need_create = (
            not hasattr(self, "_tinp_markers")
            or self._tinp_markers is None
            or len(self._tinp_lines) != self.n_pts
        )
        if need_create:
            for ln in getattr(self, "_tinp_lines", []):
                ln.remove()
            if self._tinp_markers is not None:
                self._tinp_markers.remove()

            self._tinp_lines = []
            for ti in self.t_inp:
                ln = self.ax_tinp.axvline(ti, color="steelblue", lw=1.2,
                                          alpha=0.7, zorder=2)
                self._tinp_lines.append(ln)
            self._tinp_markers = self.ax_tinp.scatter(
                self.t_inp, np.zeros(self.n_pts),
                c="steelblue", s=60, zorder=5, picker=8, marker="o",
                edgecolors="darkblue", linewidths=1.5)
        else:
            for i, ti in enumerate(self.t_inp):
                self._tinp_lines[i].set_xdata([ti, ti])
            if self._tinp_markers is not None:
                self._tinp_markers.set_offsets(np.column_stack([self.t_inp, np.zeros(self.n_pts)]))
        self.fig.canvas.draw_idle()

    # ═══════════════════════════════════════════════════════════
    #  Event handling
    # ═══════════════════════════════════════════════════════════

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.inaxes is None:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode != "":
            return

        # -- check t_inp markers --
        if event.inaxes == self.ax_tinp:
            for i, ti in enumerate(self.t_inp):
                if i == 0 or i == self.n_pts - 1:
                    continue  # endpoints fixed
                if abs(event.xdata - ti) < self.T * 0.025:
                    self._dragging_tinp_idx = i
                    return
            return

        # -- check ctrl point arrows (theta drag) --
        if event.inaxes == self.ax_traj:
            pts = self.ctrl_pts
            for i in range(self.n_pts):
                x, y, th = pts[i]
                ax, ay = x + 0.35 * np.cos(th), y + 0.35 * np.sin(th)
                if np.hypot(event.xdata - ax, event.ydata - ay) < 0.25:
                    self._dragging_ctrl_idx = i
                    self._dragging_ctrl_theta = True
                    return

            # -- check ctrl point bodies --
            for i in range(self.n_pts):
                x, y, _ = pts[i]
                if np.hypot(event.xdata - x, event.ydata - y) < 0.4:
                    self._dragging_ctrl_idx = i
                    self._dragging_ctrl_theta = False
                    return

            # -- check robot offset markers --
            for i in range(self.N_robots):
                name = f"_robot_offset_dot_{i}"
                dot = getattr(self, name, None)
                if dot is not None:
                    contains, _ = dot.contains(event)
                    if contains:
                        self._dragging_robot_idx = i
                        return

    def _on_release(self, event):
        self._dragging_ctrl_idx = None
        self._dragging_ctrl_theta = False
        self._dragging_tinp_idx = None
        self._dragging_robot_idx = None

    def _on_motion(self, event):
        if event.inaxes is None or event.xdata is None:
            return

        # -- drag t_inp marker --
        if self._dragging_tinp_idx is not None:
            idx = self._dragging_tinp_idx
            new_t = np.clip(event.xdata,
                            self.t_inp[idx - 1] + 0.01,
                            self.t_inp[idx + 1] - 0.01)
            self.t_inp = self.t_inp.copy()
            self.t_inp[idx] = new_t
            self._update_tinp_markers()
            self._update_all()
            return

        # -- drag control point --
        if self._dragging_ctrl_idx is not None:
            idx = self._dragging_ctrl_idx
            if self._dragging_ctrl_theta:
                xc, yc, _ = self.ctrl_pts[idx]
                new_th = np.arctan2(event.ydata - yc, event.xdata - xc)
                self.ctrl_pts[idx, 2] = new_th
            else:
                self.ctrl_pts[idx, 0] = event.xdata
                self.ctrl_pts[idx, 1] = event.ydata
            self._update_all()
            return

        # -- drag robot offset --
        if self._dragging_robot_idx is not None:
            i = self._dragging_robot_idx
            c = self._cache
            xc0, yc0, th0 = c["centre"][0]
            dx = event.xdata - xc0
            dy = event.ydata - yc0
            # inverse rotation
            ct, st = np.cos(th0), np.sin(th0)
            self.r_offset[i, 0] = ct * dx + st * dy
            self.r_offset[i, 1] = -st * dx + ct * dy
            self._update_all()
            return

    # ═══════════════════════════════════════════════════════════
    #  TextBox callbacks
    # ═══════════════════════════════════════════════════════════

    def _on_num_pts(self, text):
        try:
            v = int(text)
            if v < 2:
                return
            if v > 50:
                v = 50
            if v == self.n_pts:
                return
            self.n_pts = v
            old_t = self.t_inp
            self.t_inp = np.linspace(0, self.T, self.n_pts)
            if len(old_t) > 2:
                self.t_inp[1:-1] = np.interp(
                    np.linspace(0, 1, self.n_pts)[1:-1],
                    np.linspace(0, 1, len(old_t)),
                    old_t)
            # clamp deg if needed
            if self.deg >= self.n_pts:
                self.deg = max(1, self.n_pts - 1)
                self._textboxes["deg"].set_val(str(self.deg))
            self.ctrl_pts = self._default_ctrl_pts()
            self._update_tinp_markers()
            self._update_all()
        except ValueError:
            pass

    def _on_deg(self, text):
        try:
            v = int(text)
            if v < 1:
                return
            max_deg = min(10, self.n_pts - 1)
            if v > max_deg:
                v = max_deg
                self._textboxes["deg"].set_val(str(v))
            if v == self.deg:
                return
            self.deg = v
            self._update_all()
        except ValueError:
            pass

    def _on_T(self, text):
        try:
            v = float(text)
            if v <= 0:
                return
            self.T = v
            self.t_inp = np.linspace(0, self.T, self.n_pts)
            self.ctrl_pts = self._default_ctrl_pts()
            self._t_dense = np.linspace(0, self.T, 300)
            self.ax_tinp.set_xlim(-0.02 * self.T, self.T * 1.02)
            self._tinp_T_line.set_xdata([self.T, self.T])
            self._update_tinp_markers()
            self._update_all()
        except ValueError:
            pass

    def _on_N(self, text):
        try:
            v = int(text)
            if v < 0:
                return
            if v > 10:
                v = 10
            if v == self.N_robots:
                return
            old = self.N_robots
            self.N_robots = v
            if v > old:
                extra = v - old
                new_offsets = np.random.uniform(-2, 2, (extra, 2))
                self.r_offset = np.vstack([self.r_offset, new_offsets])
            else:
                self.r_offset = self.r_offset[:v]
            self._update_all()
        except ValueError:
            pass

    # ═══════════════════════════════════════════════════════════
    #  Entry point
    # ═══════════════════════════════════════════════════════════

    def run(self):
        self._connect_events()
        plt.show()

