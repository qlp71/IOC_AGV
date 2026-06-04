#!/usr/bin/env python3
"""
Interactive simulation of inverse optimal control for a differential-drive
mobile robot. Demonstrates four control-law variants (ω̃) and the effect of
gains (k₁, k₂, k₃) on closed-loop trajectories.

Usage:
    python simulation.py

Controls:
    - Drag green circle: change start position (x₀, y₀)
    - Drag red circle:   change target position
    - Sliders:           k₁, k₂, k₃ gains and initial orientation θ₀
    - Radio buttons:     select control-law variant 1–4
    - Scroll on plot:    zoom  |  Middle-drag: pan
    - Checkbox:          toggle tanh saturation of v, ω (v_max=1.0, ω_max=1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import matplotlib

matplotlib.use("QtAgg")

from ioc_utils import optimal_curve

EPS = 1e-8
R_ARROW = 0.4


# Physical limits v = tanh(v / V_MAX) * V_MAX, ω = tanh(ω / W_MAX) * W_MAX
V_MAX = 1.0
W_MAX = 1.0

# ═══════════════════════════════════════════════════════════════
#  Mathematical utilities
# ═══════════════════════════════════════════════════════════════

def sinc(x):
    """sin(x)/x with correct limit at x=0 (scalar)."""
    return 1.0 if abs(x) < EPS else np.sin(x) / x


def norm_angle(a):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))


def cart_to_polar(x, y, theta):
    """Cartesian state (x, y, θ) → polar state (ρ, δ, γ)."""
    rho = np.sqrt(x**2 + y**2)
    if rho < EPS:
        return 0.0, 0.0, norm_angle(-theta)
    delta = norm_angle(np.arctan2(y, x) + np.pi)
    gamma = norm_angle(delta - theta)
    return rho, delta, gamma


# ═══════════════════════════════════════════════════════════════
#  Four control-law variants for ω̃
# ═══════════════════════════════════════════════════════════════

def tilde_omega_1(delta, gamma, k2, k3):
    """ω̃ = k₂ sin(γ) + k₃ sinc(2γ) δ"""
    return k2 * np.sin(gamma) + k3 * sinc(2 * gamma) * delta


def tilde_omega_2(delta, gamma, k2, k3):
    """ω̃ = k₂ sin(γ) + k₃ cos(γ) / (1 + tan²(γ/2))² · δ"""
    denom = (1 + np.tan(gamma / 2) ** 2) ** 2
    return k2 * np.sin(gamma) + k3 * np.cos(gamma) / denom * delta


def tilde_omega_3(delta, gamma, k2, k3):
    """ω̃ = k₂ sin(γ) + 2 k₃ sinc(2γ) (1 + tan²(δ/2)) tan(δ/2)"""
    term = (1 + np.tan(delta / 2) ** 2) * np.tan(delta / 2)
    return k2 * np.sin(gamma) + 2 * k3 * sinc(2 * gamma) * term


def tilde_omega_4(delta, gamma, k2, k3):
    """ω̃ = k₂ sin(γ) + 2 k₃ cos(γ)/(1+tan²(γ/2))² · (1+tan²(δ/2)) tan(δ/2)"""
    denom = (1 + np.tan(gamma / 2) ** 2) ** 2
    term = (1 + np.tan(delta / 2) ** 2) * np.tan(delta / 2)
    return k2 * np.sin(gamma) + 2 * k3 * np.cos(gamma) / denom * term


TILDE_OMEGA = [tilde_omega_1, tilde_omega_2, tilde_omega_3, tilde_omega_4]
VARIANT_LABELS = [f"Variant {i + 1}" for i in range(4)]


# ═══════════════════════════════════════════════════════════════
#  Simulation engine
# ═══════════════════════════════════════════════════════════════

def simulate(x0, y0, theta0, xt, yt, k1, k2, k3,
             variant_idx, dt=0.01, max_steps=5000, tol=1e-3,
             saturate=False):
    """
    Simulate the closed-loop system from start pose to target.

    Parameters
    ----------
    saturate : bool
        If True, apply tanh-based saturation to v and ω using V_MAX, W_MAX.

    Returns a dict with full time-histories of Cartesian states, polar
    states, and control inputs.
    """
    x, y, theta = x0 - xt, y0 - yt, theta0

    xs, ys, thetas = [x0], [y0], [theta0]
    vs, omegas = [], []
    rhos, deltas, gammas = [], [], []
    ts = [0.0]

    for _ in range(max_steps):
        rho, delta, gamma = cart_to_polar(x, y, theta)
        if rho < tol:
            break

        v = k1 * rho * np.cos(gamma)
        w_tilde = TILDE_OMEGA[variant_idx](delta, gamma, k2, k3)
        omega = 0.5 * k1 * np.sin(2 * gamma) + w_tilde

        if saturate:
            v = np.tanh(v / V_MAX) * V_MAX
            omega = np.tanh(omega / W_MAX) * W_MAX

        vs.append(v)
        omegas.append(omega)
        rhos.append(rho)
        deltas.append(delta)
        gammas.append(gamma)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        ts.append(ts[-1] + dt)
        xs.append(x + xt)
        ys.append(y + yt)
        thetas.append(theta)

    return {
        "xs": np.array(xs), "ys": np.array(ys), "thetas": np.array(thetas),
        "vs": np.array(vs), "omegas": np.array(omegas),
        "rhos": np.array(rhos), "deltas": np.array(deltas),
        "gammas": np.array(gammas),
        "ts": np.array(ts),
    }


# ═══════════════════════════════════════════════════════════════
#  Interactive GUI
# ═══════════════════════════════════════════════════════════════

class InteractiveSim:
    """Main window with draggable poses, sliders and radio buttons."""

    def __init__(self):
        # -- Default state --
        self.x0, self.y0, self.theta0 = 5.0, 3.0, np.deg2rad(60)
        self.xt, self.yt = 0.0, 0.0
        self.k1, self.k2, self.k3 = 1.0, 2.0, 0.5
        self.r1, self.r2 = -4.0, 0.0
        self.variant_idx = 0
        self.saturate = False
        self._dragging = None          # 'start' | 'target' | None
        self.ioc_result = None         # dict from optimal_curve()
        self._ioc_computing = False

        # -- Build UI --
        self.fig = plt.figure("IOC-AGV", figsize=(17, 10))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("IOC-AGV — Inverse Optimal Control Simulation")
        self._build_axes()
        self._build_artists()
        self._build_widgets()
        self._connect_events()
        self._update_trajectory()

    # ── Layout ───────────────────────────────────────────────

    def _build_axes(self):
        # Trajectory (left, large)
        self.ax_traj = self.fig.add_axes((0.04, 0.30, 0.44, 0.66))
        self.ax_traj.set_aspect("equal")
        self.ax_traj.set_xlabel("x")
        self.ax_traj.set_ylabel("y")
        self.ax_traj.set_title(
            "Drag start (green) / target (red)  ·  "
            "adjust sliders below  ·  pick variant →",
            fontsize=10,
        )
        self.ax_traj.grid(True, alpha=0.3)

        # v, ω  time-series (top-right)
        self.ax_vw = self.fig.add_axes((0.54, 0.70, 0.43, 0.26))
        self.ax_w = self.ax_vw.twinx()
        self.ax_vw.set_xlabel("t [s]")
        self.ax_vw.set_ylabel("v [m/s]", color="tab:blue")
        self.ax_w.set_ylabel("ω [rad/s]", color="tab:red")
        self.ax_vw.tick_params(axis="y", colors="tab:blue")
        self.ax_w.tick_params(axis="y", colors="tab:red")
        self.ax_vw.set_title("Control inputs", fontsize=10)
        self.ax_vw.grid(True, alpha=0.3)

        # x, y, θ  time-series (middle-right)
        self.ax_xyz = self.fig.add_axes((0.54, 0.38, 0.43, 0.26))
        self.ax_xyz.set_xlabel("t [s]")
        self.ax_xyz.set_ylabel("state")
        self.ax_xyz.set_title("Cartesian states  x, y, θ", fontsize=10)
        self.ax_xyz.grid(True, alpha=0.3)

        # ρ, δ, γ  time-series (bottom-right)
        self.ax_pdg = self.fig.add_axes((0.54, 0.06, 0.43, 0.26))
        self.ax_pdg.set_xlabel("t [s]")
        self.ax_pdg.set_ylabel("state")
        self.ax_pdg.set_title("Polar states  ρ, δ, γ", fontsize=10)
        self.ax_pdg.grid(True, alpha=0.3)

    def _build_artists(self):
        # -- Trajectory plot --
        (self.line_traj,) = self.ax_traj.plot(
            [], [], "b-", lw=2.0, label="trajectory", zorder=3)
        self.quiver_poses = None

        # IOC overlay
        (self.line_ioc,) = self.ax_traj.plot(
            [], [], "--", color="darkorange", lw=2.0,
            label="IOC B-spline", zorder=4)
        (self.ctrl_polygon,) = self.ax_traj.plot(
            [], [], "-.", color="gray", lw=0.8, alpha=0.6,
            label="control polygon", zorder=2)
        self.ctrl_scatter = self.ax_traj.scatter(
            [], [], c="darkorange", s=30, marker="o",
            zorder=8, label="ctrl pts")

        self.pt_start = Circle(
            (self.x0, self.y0), 0.15, fc="limegreen", ec="darkgreen",
            lw=2, zorder=10, picker=5, label="start")
        self.arrow_start = self._make_arrow(
            self.x0, self.y0, self.theta0, "darkgreen")

        self.pt_target = Circle(
            (self.xt, self.yt), 0.15, fc="red", ec="darkred",
            lw=2, zorder=10, picker=5, label="target")
        self.arrow_target = self._make_arrow(
            self.xt, self.yt, 0.0, "darkred")

        self.ax_traj.add_patch(self.pt_start)
        self.ax_traj.add_patch(self.pt_target)
        self.ax_traj.add_line(self.arrow_start)
        self.ax_traj.add_line(self.arrow_target)
        self.ax_traj.legend(
            handles=[self.line_traj, self.line_ioc,
                     self.ctrl_polygon, self.ctrl_scatter,
                     self.pt_start, self.pt_target],
            loc="upper right", fontsize=7,
        ).set_zorder(20)

        # -- v / ω plot (dual y-axes) --
        (self.line_v,) = self.ax_vw.plot(
            [], [], "tab:blue", lw=1.5, label="v")
        (self.line_w,) = self.ax_w.plot(
            [], [], "tab:red", lw=1.5, label="ω")
        # -- x, y, θ plot --
        (self.line_x,) = self.ax_xyz.plot(
            [], [], "tab:blue", lw=1.0, label="x")
        (self.line_y,) = self.ax_xyz.plot(
            [], [], "tab:red", lw=1.0, label="y")
        (self.line_theta,) = self.ax_xyz.plot(
            [], [], "tab:green", lw=1.0, label="θ")

        # -- ρ, δ, γ plot --
        (self.line_rho,) = self.ax_pdg.plot(
            [], [], "tab:blue", lw=1.0, label="ρ")
        (self.line_delta,) = self.ax_pdg.plot(
            [], [], "tab:red", lw=1.0, label="δ")
        (self.line_gamma,) = self.ax_pdg.plot(
            [], [], "tab:green", lw=1.0, label="γ")

        # IOC time-series (dashed)
        (self.line_ioc_v,) = self.ax_vw.plot(
            [], [], "--", color="darkorange", lw=1.0, alpha=0.7, label="IOC v")
        (self.line_ioc_w,) = self.ax_w.plot(
            [], [], "--", color="brown", lw=1.0, alpha=0.7, label="IOC ω")
        (self.line_ioc_x,) = self.ax_xyz.plot(
            [], [], "--", color="darkorange", lw=1.0, alpha=0.7, label="IOC x")
        (self.line_ioc_y,) = self.ax_xyz.plot(
            [], [], "--", color="brown", lw=1.0, alpha=0.7, label="IOC y")
        (self.line_ioc_theta,) = self.ax_xyz.plot(
            [], [], "--", color="purple", lw=1.0, alpha=0.7, label="IOC θ")
        (self.line_ioc_rho,) = self.ax_pdg.plot(
            [], [], "--", color="darkorange", lw=1.0, alpha=0.7, label="IOC ρ")
        (self.line_ioc_delta,) = self.ax_pdg.plot(
            [], [], "--", color="brown", lw=1.0, alpha=0.7, label="IOC δ")
        (self.line_ioc_gamma,) = self.ax_pdg.plot(
            [], [], "--", color="purple", lw=1.0, alpha=0.7, label="IOC γ")

        # legends (built after all lines are created)
        lines_vw = [self.line_v, self.line_w, self.line_ioc_v, self.line_ioc_w]
        self.ax_vw.legend(lines_vw, [str(l.get_label()) for l in lines_vw],
                          loc="upper right", fontsize=6)
        lines_xyz = [self.line_x, self.line_y, self.line_theta,
                     self.line_ioc_x, self.line_ioc_y, self.line_ioc_theta]
        self.ax_xyz.legend(lines_xyz, [str(l.get_label()) for l in lines_xyz],
                           loc="upper right", fontsize=6)
        lines_pdg = [self.line_rho, self.line_delta, self.line_gamma,
                     self.line_ioc_rho, self.line_ioc_delta, self.line_ioc_gamma]
        self.ax_pdg.legend(lines_pdg, [str(l.get_label()) for l in lines_pdg],
                           loc="upper right", fontsize=6)
        
        # obstacle overlay (circle at (r1, r2) with radius 0.5)
        self.obstacle_patch = Circle(
            (self.r1, self.r2), 0.5, fc="none", ec="gray", lw=2, ls="-", fill=True, alpha=0.5, zorder=9)
        self.ax_traj.add_patch(self.obstacle_patch)

    @staticmethod
    def _make_arrow(x, y, theta, color):
        dx = R_ARROW * np.cos(theta)
        dy = R_ARROW * np.sin(theta)
        return Line2D([x, x + dx], [y, y + dy],
                       color=color, lw=2.5, zorder=9)

    # ── Widgets ──────────────────────────────────────────────

    def _build_widgets(self):
        # Saturation toggle
        ax_sat = self.fig.add_axes((0.04, 0.35, 0.15, 0.04))
        self.check_sat = CheckButtons(
            ax_sat, ["tanh saturate"], actives=[self.saturate])
        self.check_sat.on_clicked(self._on_saturate)

        # IOC button
        ax_ioc_btn = self.fig.add_axes((0.04, 0.01, 0.12, 0.035))
        self.btn_ioc = Button(ax_ioc_btn, "Compute IOC",
                              color="lightcoral", hovercolor="tomato")
        self.btn_ioc.on_clicked(self._on_compute_ioc)

        # IOC status text
        self.ax_ioc_status = self.fig.add_axes((0.17, 0.005, 0.08, 0.045))
        self.ax_ioc_status.axis("off")
        self.ioc_status_text = self.ax_ioc_status.text(
            0, 0.5, "", va="center", fontfamily="monospace", fontsize=7,
            color="darkorange", transform=self.ax_ioc_status.transAxes)

        # Control weights r₁, r₂
        ax_r1 = self.fig.add_axes((0.04, 0.29, 0.18, 0.02))
        self.slider_r1 = Slider(
            ax_r1, "r₁", -5.0, 5.0, valinit=self.r1, valstep=0.5)
        self.slider_r1.on_changed(self._on_slider)

        ax_r2 = self.fig.add_axes((0.04, 0.32, 0.18, 0.02))
        self.slider_r2 = Slider(
            ax_r2, "r₂", -5.0, 5.0, valinit=self.r2, valstep=0.5)
        self.slider_r2.on_changed(self._on_slider)

        # Gain k₁
        ax_k1 = self.fig.add_axes((0.04, 0.22, 0.18, 0.025))
        self.slider_k1 = Slider(
            ax_k1, "k₁", 0.1, 5.0, valinit=self.k1, valstep=0.01)
        self.slider_k1.on_changed(self._on_slider)

        # Gain k₂
        ax_k2 = self.fig.add_axes((0.04, 0.16, 0.18, 0.025))
        self.slider_k2 = Slider(
            ax_k2, "k₂", 0.0, 10.0, valinit=self.k2, valstep=0.01)
        self.slider_k2.on_changed(self._on_slider)

        # Gain k₃
        ax_k3 = self.fig.add_axes((0.04, 0.10, 0.18, 0.025))
        self.slider_k3 = Slider(
            ax_k3, "k₃", 0.0, 10.0, valinit=self.k3, valstep=0.01)
        self.slider_k3.on_changed(self._on_slider)

        # Initial orientation θ₀
        ax_th0 = self.fig.add_axes((0.04, 0.04, 0.18, 0.025))
        self.slider_th0 = Slider(
            ax_th0, "θ₀ [deg]", -180, 180,
            valinit=np.rad2deg(self.theta0), valstep=1)
        self.slider_th0.on_changed(self._on_slider)

        # Radio buttons — control-law variant
        ax_radio = self.fig.add_axes((0.27, 0.04, 0.18, 0.24))
        self.radio = RadioButtons(
            ax_radio, VARIANT_LABELS, active=self.variant_idx)
        self.radio.on_clicked(self._on_radio)

        # Info readout
        self.ax_info = self.fig.add_axes((0.46, 0.04, 0.06, 0.24))
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0, 0.95, "", va="top", fontfamily="monospace", fontsize=8,
            transform=self.ax_info.transAxes)

    # ── Event handling ───────────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect(
            "button_press_event", self._on_press)
        self.fig.canvas.mpl_connect(
            "button_release_event", self._on_release)
        self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.ax_traj:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode != "":
            return
        for name, pt in [("start", self.pt_start),
                         ("target", self.pt_target)]:
            contains, _ = pt.contains(event)
            if contains:
                self._dragging = name
                return

    def _on_release(self, event):
        self._dragging = None

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes != self.ax_traj:
            return
        x, y = event.xdata, event.ydata
        if self._dragging == "start":
            self.x0, self.y0 = x, y
            self.pt_start.set_center((x, y))
            self._update_arrow_line(
                self.arrow_start, x, y, self.theta0)
        elif self._dragging == "target":
            self.xt, self.yt = x, y
            self.pt_target.set_center((x, y))
            self._update_arrow_line(
                self.arrow_target, x, y, 0.0)
        self._update_trajectory()

    @staticmethod
    def _update_arrow_line(line, x, y, theta):
        dx = R_ARROW * np.cos(theta)
        dy = R_ARROW * np.sin(theta)
        line.set_data([x, x + dx], [y, y + dy])

    def _on_slider(self, val):
        self.k1 = self.slider_k1.val
        self.k2 = self.slider_k2.val
        self.k3 = self.slider_k3.val
        self.r1 = self.slider_r1.val
        self.r2 = self.slider_r2.val
        self.theta0 = np.deg2rad(self.slider_th0.val)
        self._update_arrow_line(
            self.arrow_start, self.x0, self.y0, self.theta0)
        self.ioc_result = None  # invalidate IOC cache
        self._update_ioc_status("")
        self._update_trajectory()

    def _on_saturate(self, label):
        self.saturate = self.check_sat.get_status()[0]
        self._update_trajectory()

    def _on_radio(self, label):
        self.variant_idx = VARIANT_LABELS.index(label)
        self.ioc_result = None
        self._update_ioc_status("")
        self._update_trajectory()

    # ── IOC computation ──────────────────────────────────────

    def _update_ioc_status(self, msg: str):
        self.ioc_status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _on_compute_ioc(self, event):
        if self._ioc_computing:
            return
        self._ioc_computing = True
        self._update_ioc_status("computing…")
        try:
            print(f"[ioc_viz] start=({self.x0:.2f},{self.y0:.2f},{self.theta0:.2f})")
            print(f"[ioc_viz] end=({self.xt:.2f},{self.yt:.2f},0.00)")
            print(f"[ioc_viz] gains: k1={self.k1}, k2={self.k2}, k3={self.k3}, r1={self.r1}, r2={self.r2}")
            print(f"[ioc_viz] variant_idx={self.variant_idx}")
            self.ioc_result = optimal_curve(
                start_point=(self.x0, self.y0, self.theta0),
                end_point=(self.xt, self.yt, 0.0),
                k1=self.k1, k2=self.k2, k3=self.k3,
                r1=self.r1, r2=self.r2,
                ctrl_law_idx=self.variant_idx,
                tot_iters=1000,
                t_restart=1000
            )
            print(f"[ioc_viz] IOC done, cost={self.ioc_result['cost']:.2f}")
            self._update_ioc_status(
                f"cost={self.ioc_result['cost']:.2f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_ioc_status(f"err: {e}")
            self.ioc_result = None
        finally:
            self._ioc_computing = False
        self._update_trajectory()

    def _update_obstacle_overlay(self):
        # update the magenta circle representing the obstacle at (r1, r2) with radius 1.0
        self.obstacle_patch.center = (self.r1, self.r2)
        self.fig.canvas.draw_idle()

    def _update_ioc_overlay(self):
        """Plot IOC B-spline curve, control points and control polygon."""
        if self.ioc_result is None:
            self.line_ioc.set_data([], [])
            self.ctrl_polygon.set_data([], [])
            self.ctrl_scatter.set_offsets(np.empty((0, 2)))
            for ln in [self.line_ioc_v, self.line_ioc_w,
                       self.line_ioc_x, self.line_ioc_y, self.line_ioc_theta,
                       self.line_ioc_rho, self.line_ioc_delta, self.line_ioc_gamma]:
                ln.set_data([], [])
            return

        r = self.ioc_result

        # ── trajectory ──
        self.line_ioc.set_data(r["traj"][:, 0], r["traj"][:, 1])

        # ── control points & polygon ──
        cp = r["ctrl_pts"]
        self.ctrl_polygon.set_data(cp[:, 0], cp[:, 1])
        self.ctrl_scatter.set_offsets(cp[:, :2])

        # ── time-series: compute v, ω, polar from IOC trajectory ──
        t_eval = r["t_eval"]
        xs = r["traj"][:, 0]
        ys = r["traj"][:, 1]
        thetas = r["traj"][:, 2]
        thetas = np.unwrap(thetas)  # unwrap for smooth differentiation
        # finite-difference v, ω
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 1.0
        dx = np.gradient(xs, dt)
        dy = np.gradient(ys, dt)
        dtheta = np.gradient(thetas, dt)
        v_ioc = np.sqrt(dx**2 + dy**2)
        omega_ioc = dtheta

        # polar
        rhos, deltas, gammas = [], [], []
        for x, y, th in zip(xs, ys, thetas):
            rho_i, delta_i, gamma_i = cart_to_polar(x, y, th)
            rhos.append(rho_i)
            deltas.append(delta_i)
            gammas.append(gamma_i)
        rhos = np.array(rhos)
        deltas = np.array(deltas)
        gammas = np.array(gammas)

        self.line_ioc_v.set_data(t_eval, v_ioc)
        self.line_ioc_w.set_data(t_eval, omega_ioc)
        self.line_ioc_x.set_data(t_eval, xs)
        self.line_ioc_y.set_data(t_eval, ys)
        self.line_ioc_theta.set_data(t_eval, thetas)
        self.line_ioc_rho.set_data(t_eval, rhos)
        self.line_ioc_delta.set_data(t_eval, deltas)
        self.line_ioc_gamma.set_data(t_eval, gammas)

    # ── Update all plots ─────────────────────────────────────

    def _update_trajectory(self):
        r = simulate(
            self.x0, self.y0, self.theta0,
            self.xt, self.yt,
            self.k1, self.k2, self.k3,
            self.variant_idx,
            saturate=self.saturate,
        )
        xs, ys, thetas = r["xs"], r["ys"], r["thetas"]
        vs, omegas = r["vs"], r["omegas"]
        rhos, deltas, gammas = r["rhos"], r["deltas"], r["gammas"]
        ts = r["ts"]
        n = len(xs)

        # ── Trajectory ──────────────────────────────────

        self.line_traj.set_data(xs, ys)

        # ── IOC overlay ──
        self._update_ioc_overlay()
        self._update_obstacle_overlay()

        if self.quiver_poses is not None:
            self.quiver_poses.remove()
            self.quiver_poses = None

        if n > 1:
            step = max(1, n // 30)
            sx, sy, st = xs[::step], ys[::step], thetas[::step]
            u = np.cos(st) * 0.15
            v = np.sin(st) * 0.15
            self.quiver_poses = self.ax_traj.quiver(
                sx, sy, u, v,
                angles="xy", scale_units="xy", scale=1,
                color="steelblue", alpha=0.50, width=0.004,
                zorder=5,
            )

        all_x = np.concatenate([xs, [self.x0, self.xt]])
        all_y = np.concatenate([ys, [self.y0, self.yt]])
        margin = 1.5
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_range = max(x_max - x_min, 2.0)
        y_range = max(y_max - y_min, 2.0)
        self.ax_traj.set_xlim(x_min - margin, x_min + x_range + margin)
        self.ax_traj.set_ylim(y_min - margin, y_min + y_range + margin)

        # ── v / ω ────────────────────────────────────────

        if len(vs) > 0:
            t_ctrl = ts[:-1]
            self.line_v.set_data(t_ctrl, vs)
            self.line_w.set_data(t_ctrl, omegas)
        else:
            self.line_v.set_data([], [])
            self.line_w.set_data([], [])
        self.ax_vw.relim()
        self.ax_vw.autoscale_view()
        self.ax_w.relim()
        self.ax_w.autoscale_view()

        # ── x, y, θ ──────────────────────────────────────

        self.line_x.set_data(ts, xs)
        self.line_y.set_data(ts, ys)
        self.line_theta.set_data(ts, thetas)
        self.ax_xyz.relim()
        self.ax_xyz.autoscale_view()

        # ── ρ, δ, γ ──────────────────────────────────────

        if len(rhos) > 0:
            t_pol = ts[:-1]
            self.line_rho.set_data(t_pol, rhos)
            self.line_delta.set_data(t_pol, deltas)
            self.line_gamma.set_data(t_pol, gammas)
        else:
            self.line_rho.set_data([], [])
            self.line_delta.set_data([], [])
            self.line_gamma.set_data([], [])
        self.ax_pdg.relim()
        self.ax_pdg.autoscale_view()

        # ── Info text ────────────────────────────────────

        lines = [
            f"Start : ({self.x0:+.2f}, {self.y0:+.2f})\n"
            f"θ₀ = {np.rad2deg(self.theta0):+.0f}°",
            f"Target: ({self.xt:+.2f}, {self.yt:+.2f})",
            f"k₁ = {self.k1:.2f}\nk₂ = {self.k2:.2f}\nk₃ = {self.k3:.2f}",
            f"r₁ = {self.r1:.2f}\nr₂ = {self.r2:.2f}",
            f"Law  : V{self.variant_idx + 1}",
            f"Sat  : {'ON' if self.saturate else 'OFF'}",
            f"Steps: {n}",
        ]
        if self.ioc_result is not None:
            lines.append(f"IOC  : cost={self.ioc_result['cost']:.2f}")
        self.info_text.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = InteractiveSim()
    sim.run()
