import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
from jax import random, vmap

from igo.blockwise_mgigo_jax import mmog_igo_optimizer_mpc as solver
from igo.utils import get_best_component_stats
from igo.plot_utils import _blend_with_previous_if_nonfinite, _safe_color_limits
from igo.utils import sample_joint_from_solver, build_joint_component_candidates


# 禁止JAX预分配大块显存
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 或者限制最多使用50%显存
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

@jax.jit
def cost_cos_cos(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = - jnp.cos(x1 * 3.14/4) * jnp.cos(x2 * 3.14/4)
    return res

@jax.jit
def cost_quadratic_half_plane(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # res = 2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    # res /= 10.0
    res = jnp.where(x1 - x2 < 2, 
                    -5.0 * (x1 - x2 - 2) + 20.0, 
                    (2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2) / 10.0)
    return res

@jax.jit
def cost_quadratic1(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = 2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    res /= 10.0
    return res

@jax.jit
def cost_quadratic2(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = 20.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    res /= 10.0
    return res

@jax.jit
def cost_quadratic_linear_constraint(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = jnp.where(jnp.abs(x1 - x2 - 2) > 0.3, 
                    5.0 * jnp.abs(x1 - x2 - 2) + 20.0,
                    (2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2) / 10.0)
    return res

@jax.jit
def cost_circle_constraint(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = jnp.where(jnp.abs(x1**2 + (x2 + 1)**2 - 16.0) > 3, 
                    5.0 * (jnp.abs(x1**2 + (x2 + 1)**2 - 16)) + 10.0, 
                    (2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2) / 10.0)
    return res

def exp_fn(x, ctx, cost, dt=0.05):
    return jnp.exp(- cost(x, ctx) * dt)

def calculate_countours(cost, x1_lim=(-12.0, 12.0), x2_lim=(-12.0, 12.0), resolution=280):
    x1 = np.linspace(start=x1_lim[0], stop=x1_lim[1], num=resolution)
    x2 = np.linspace(start=x2_lim[0], stop=x2_lim[1], num=resolution)
    xx, yy = np.meshgrid(x1, x2)
    grid_points = jnp.stack([jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1)
    zz = np.asarray(vmap(lambda p: exp_fn(p, None, cost))(grid_points)).reshape(xx.shape)
    return xx, yy, zz

def generate_animation_data(
    total_iterations=500,
    frame_stride=5,
    seed=5,
    m=1,
    dims=(2,),
    k=3,
    dt=0.15,
    b=100,
    b0=40,
    t0=100,
    cost=cost_cos_cos,
    sample_count=100,
):
    d_max = max(dims)
    if not (m == 1 and d_max == 2):
        raise ValueError("Animation currently supports single-block 2D only (m=1, dims=(2,)).")

    key = random.PRNGKey(seed)
    key, key_init = random.split(key)
    initial_mu = random.uniform(key_init, shape=(m, k, d_max), minval=-8.0, maxval=8.0)
    initial_l_inv = jnp.tile(jnp.eye(d_max, dtype=jnp.float32)[None, None, :, :], (m, k, 1, 1)) * 1.0
    initial_v = jnp.zeros((m, k - 1), dtype=jnp.float32)

    frame_data = []
    all_f_vals = []
    cur_mu = initial_mu
    cur_l_inv = initial_l_inv
    cur_v = initial_v
    warned_nonfinite_frame = False
    iter_list = list(range(0, total_iterations, frame_stride))
    if iter_list[-1] != total_iterations:
        iter_list.append(total_iterations)
    for t in tqdm.tqdm(iter_list, desc="IGO solver iterations"):
        key_solver = random.fold_in(key, 100000 + t)
        key_plot = random.fold_in(key, 100000 + t)

        final_mu, final_l_inv, final_pi, final_v = solver(key_solver, frame_stride, dt, m, k, b, b0, dims, t0, cost, cur_mu, cur_l_inv, cur_v, None)

        final_mu_np = _blend_with_previous_if_nonfinite(final_mu, cur_mu, clip_abs=1e3)
        final_l_inv_np = _blend_with_previous_if_nonfinite(final_l_inv, cur_l_inv, clip_abs=1e3)
        final_v_np = _blend_with_previous_if_nonfinite(final_v, cur_v, clip_abs=70.0)
        final_pi_np = np.asarray(final_pi)

        _, best_mu_np, best_cov_np, _ = get_best_component_stats(final_mu_np, final_l_inv_np, cost)

        raw_samples = sample_joint_from_solver(
            key_plot, final_mu_np, final_l_inv_np, final_pi_np, sample_count
        )
        samples_np = np.asarray(raw_samples)
        f_vals_np = np.asarray(vmap(lambda s: cost(s, None))(samples_np))

        finite_mask = np.isfinite(samples_np).all(axis=1) & np.isfinite(f_vals_np)
        if np.any(finite_mask):
            samples_np = samples_np[finite_mask]
            f_vals_np = f_vals_np[finite_mask]
        else:
            if not warned_nonfinite_frame:
                print(f"[warn] all samples non-finite at iter={t}")
                warned_nonfinite_frame = True
            samples_np = np.zeros((1, 2))
            f_vals_np = np.zeros((1,))

        means_np, covs_np = build_joint_component_candidates(final_mu_np, final_l_inv_np)

        frame_data.append({
            "t": t,
            "samples": samples_np,
            "f_vals": f_vals_np,
            "means": np.asarray(means_np),
            "covs": np.asarray(covs_np),
            "best_mu": best_mu_np,
            "best_cov": best_cov_np,
        })
        all_f_vals.append(f_vals_np)
        if t % t0 == 0:
            cur_mu, cur_l_inv, cur_v = final_mu_np, initial_l_inv, initial_v
        else:
            cur_mu, cur_l_inv, cur_v = final_mu_np, final_l_inv_np, final_v_np

    return frame_data, all_f_vals

def render_iteration_animation(output_root, frame_data, all_f_vals, cost, cost_name="cost", fps=15):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pbar = tqdm.tqdm(total=len(frame_data), desc="Saving animation")
    def progress_callback(i, n):
        pbar.update(i + 1 - pbar.n)

    f_all = np.concatenate(all_f_vals, axis=0)
    vmin, vmax = _safe_color_limits(f_all)

    xx, yy, zz = calculate_countours(cost)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    contour = ax.contour(xx, yy, zz, levels=32, cmap="Blues")
    ax.clabel(contour, inline=True, fontsize=7)

    first = frame_data[0]
    sc = ax.scatter(
        first["samples"][:, 0], first["samples"][:, 1], c=first["f_vals"],
        cmap="viridis_r", vmin=vmin, vmax=vmax, s=20, alpha=0.7, linewidths=0.0, zorder=1,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cost")

    mean_scatter = ax.scatter([], [], c="tab:orange", s=36, marker="D", edgecolors="black", linewidths=0.5, zorder=4)
    best_scatter = ax.scatter([], [], c="gold", s=180, marker="*", edgecolors="black", linewidths=0.9, zorder=6)
    title = ax.set_title(f"IGO optimization animation (iter={first['t']})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal", adjustable="box")
    
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, alpha=0.8, label="samples"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:orange", markeredgecolor="black", markersize=7, label="component means"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=1.2, label="3σ ellipses"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=11, label="best mean"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="best 3σ"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    def _ellipse_params(mean_xy, cov_xy, n_std=1.0):
        from igo.utils import _safe_eigh_numpy
        eigvals, eigvecs = _safe_eigh_numpy(cov_xy)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        w = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-12))
        h = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-12))
        a = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return tuple(mean_xy), w, h, a, True

    n_comp = first["means"].shape[0]
    comp_ellipses = []
    for i in range(n_comp):
        center, w, h, a, valid = _ellipse_params(first["means"][i], first["covs"][i])
        e = Ellipse(
            xy=center, width=w, height=h, angle=a, edgecolor="tab:orange",
            facecolor="none", linewidth=1.2, alpha=0.8, linestyle="-", zorder=3,
        )
        e.set_visible(valid)
        ax.add_patch(e)
        comp_ellipses.append(e)

    best_center, best_w, best_h, best_a, best_valid = _ellipse_params(first["best_mu"], first["best_cov"], n_std=3.0)
    best_ellipse = Ellipse(
        xy=best_center, width=best_w, height=best_h, angle=best_a, edgecolor="black",
        facecolor="none", linewidth=1.8, alpha=0.85, linestyle="--", zorder=5,
    )
    best_ellipse.set_visible(best_valid)
    ax.add_patch(best_ellipse)

    def update(frame_idx):
        fd = frame_data[frame_idx]
        title.set_text(f"IGO optimization animation (iter={fd['t']})")

        sc.set_offsets(fd["samples"])
        sc.set_array(fd["f_vals"])

        mean_scatter.set_offsets(fd["means"])
        best_scatter.set_offsets([fd["best_mu"]])

        for i in range(n_comp):
            center, w, h, a, valid = _ellipse_params(fd["means"][i], fd["covs"][i], n_std=3.0)
            e = comp_ellipses[i]
            e.set_center(center)
            e.width = w
            e.height = h
            e.angle = a
            e.set_visible(valid)

        b_center, b_w, b_h, b_a, b_valid = _ellipse_params(fd["best_mu"], fd["best_cov"], n_std=3.0)
        best_ellipse.set_center(b_center)
        best_ellipse.width = b_w
        best_ellipse.height = b_h
        best_ellipse.angle = b_a
        best_ellipse.set_visible(b_valid)

        return [sc, mean_scatter, best_scatter, best_ellipse, title] + comp_ellipses

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_data), interval=150, blit=False, repeat=False
    )

    mp4_path = output_root / (cost_name + ".mp4")
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='MGIGO'))
        # ani.save(mp4_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        ani.save(
            mp4_path,
            writer=writer,
            progress_callback=progress_callback,
        )
    except Exception as err:
        print(f"[error] Failed to save mp4, trying gif. Error: {err}")
        gif_path = output_root / (cost_name + ".gif")
        ani.save(gif_path, writer="pillow", fps=fps)
        mp4_path = gif_path

    plt.close(fig)

    print(f"Saved animation: {mp4_path}")
    return mp4_path

def save_contour_plot(output_root, cost, cost_name="cost"):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    xx, yy, zz = calculate_countours(cost)
    plt.figure(figsize=(5.0, 5.0), dpi=300)
    contour = plt.contour(xx, yy, zz, levels=32, cmap="Blues")
    plt.clabel(contour, inline=True, fontsize=7)
    plt.title(f"Cost contour: {cost_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.2)
    plt.xlim(-8.0, 8.0)
    plt.ylim(-8.0, 8.0)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    # plt.tight_layout()
    contour_path = output_root / f"{cost_name}_contour.png"
    plt.savefig(contour_path)
    plt.close()
    print(f"Saved contour plot: {contour_path}")
    return contour_path

TEST = [
    {"name": "cos_cos_test", "cost": cost_cos_cos},
    {"name": "quadratic_half_plane_test", "cost": cost_quadratic_half_plane},
    {"name": "quadratic_linear_test", "cost": cost_quadratic_linear_constraint},
    {"name": "quadratic_test1", "cost": cost_quadratic1},
    {"name": "quadratic_test2", "cost": cost_quadratic2},
    {"name": "quadratic_circle_test", "cost": cost_circle_constraint},
]

def save_all_contours(output_root):
    for test in TEST:
        save_contour_plot(output_root, test["cost"], cost_name=test["name"])

def main():
    test_idx = 0
    output_root = Path("output_igo_test")
    output_root.mkdir(parents=True, exist_ok=True)
    save_contour_plot(output_root, TEST[test_idx]["cost"], cost_name=TEST[test_idx]["name"])
    # breakpoint()
    k = 5
    dt = 0.15
    b = 300
    b0 = 100
    t0 = 200
    iter_tot = 600
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=iter_tot, frame_stride=5, seed=42,
        m=1, dims=(2,), k=k, dt=dt, b=b, b0=b0, t0=t0, cost=TEST[test_idx]["cost"], sample_count=100,
    )
    render_iteration_animation(output_root=output_root, frame_data=frame_data, all_f_vals=all_f_vals, cost=TEST[test_idx]["cost"], cost_name=TEST[test_idx]["name"], fps=15)

if __name__ == "__main__":
    main()
    # output_root = Path("output_igo_test")
    # save_all_contours(output_root)