# Hermite 样条

## 定义

Hermite 插值不仅插值函数值，还**同时插值导数值**。对于单段三次 Hermite 曲线（Ferguson 曲线）：

给定端点条件：$\gamma(0) = \mathbf{P}_0$，$\gamma(1) = \mathbf{P}_1$，$\gamma'(0) = \mathbf{v}_0$，$\gamma'(1) = \mathbf{v}_1$。

三次 Hermite 多项式为：

$$
\gamma(t) = H_0^3(t) \mathbf{P}_0 + H_1^3(t) \mathbf{P}_1 + H_2^3(t) \mathbf{v}_0 + H_3^3(t) \mathbf{v}_1, \quad t \in [0,1]
$$

其中 **Hermite 基函数**为：

$$
\begin{aligned}
H_0^3(t) &= 2t^3 - 3t^2 + 1 = (1 + 2t)(1-t)^2 \\
H_1^3(t) &= -2t^3 + 3t^2 = t^2(3-2t) \\
H_2^3(t) &= t^3 - 2t^2 + t = t(1-t)^2 \\
H_3^3(t) &= t^3 - t^2 = t^2(t-1)
\end{aligned}
$$

### 基函数满足的条件

在 $t=0$ 和 $t=1$ 处：

| 函数 | $t=0$ 值 | $t=1$ 值 | $t=0$ 导数 | $t=1$ 导数 |
|------|----------|----------|------------|------------|
| $H_0^3$ | 1 | 0 | 0 | 0 |
| $H_1^3$ | 0 | 1 | 0 | 0 |
| $H_2^3$ | 0 | 0 | 1 | 0 |
| $H_3^3$ | 0 | 0 | 0 | 1 |

这构成了标准的 Hermite 插值基。

---

## 矩阵形式

$$
\gamma(t) = \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
-3 & 3 & -2 & -1 \\
2 & -2 & 1 & 1
\end{bmatrix}
\begin{bmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \mathbf{v}_0 \\ \mathbf{v}_1 \end{bmatrix}
$$

### 与三次 Bézier 的等价关系

三次 Hermite 与三次 Bézier 可通过控制点转换互相表示：

$$
\begin{aligned}
\mathbf{P}_0^{\text{bez}} &= \mathbf{P}_0^{\text{her}} \\
\mathbf{P}_1^{\text{bez}} &= \mathbf{P}_0^{\text{her}} + \frac{1}{3}\mathbf{v}_0 \\
\mathbf{P}_2^{\text{bez}} &= \mathbf{P}_1^{\text{her}} - \frac{1}{3}\mathbf{v}_1 \\
\mathbf{P}_3^{\text{bez}} &= \mathbf{P}_1^{\text{her}}
\end{aligned}
$$

逆变换：

$$
\begin{aligned}
\mathbf{v}_0 &= 3(\mathbf{P}_1^{\text{bez}} - \mathbf{P}_0^{\text{bez}}) \\
\mathbf{v}_1 &= 3(\mathbf{P}_3^{\text{bez}} - \mathbf{P}_2^{\text{bez}})
\end{aligned}
$$

这说明两种表示在三次情形下是**等价的**，选择取决于应用场景对端点切向量的直观控制需求。

---

## 分段 Hermite 样条

给定节点 $a = t_0 < t_1 < \cdots < t_n = b$，每段 $[t_i, t_{i+1}]$ 上：

- 位置值 $\mathbf{P}_i$, $\mathbf{P}_{i+1}$
- 切向量 $\mathbf{v}_i$, $\mathbf{v}_{i+1}$

段内曲线为：

$$
\gamma_i(t) = H_0^3(\tau) \mathbf{P}_i + H_1^3(\tau) \mathbf{P}_{i+1} + h_i H_2^3(\tau) \mathbf{v}_i + h_i H_3^3(\tau) \mathbf{v}_{i+1}
$$

其中 $\tau = \frac{t - t_i}{h_i}$，$h_i = t_{i+1} - t_i$。因子 $h_i$ 来自导数缩放。

---

## Catmull-Rom 样条

**Catmull-Rom 样条**是 Hermite 样条的一个重要特例，其中切向量由相邻点差分自动生成（不需要用户指定导数）：

### 均匀 Catmull-Rom

对于等距节点（$h_i$ 均为常数）：

$$
\mathbf{v}_i = \frac{\mathbf{P}_{i+1} - \mathbf{P}_{i-1}}{2}
$$

即中心差分格式。这产生 $C^1$ 连续的插值曲线。

### 向心 Catmull-Rom（Centripetal）

Barry & Goldman (1988) 提出用非均匀参数化改善形状：

令 $t_0 = 0$，$t_i = t_{i-1} + \|\mathbf{P}_i - \mathbf{P}_{i-1}\|^\alpha$，其中 $\alpha = 0.5$ 为向心参数化（$\alpha = 0$ 为均匀，$\alpha = 1$ 为弦长）。

切向量：

$$
\mathbf{v}_i = (1 - c) \frac{\mathbf{P}_{i+1} - \mathbf{P}_{i-1}}{t_{i+1} - t_{i-1}}
$$

其中 $c$ 为张力参数（Cardinal spline 的推广）。

### 性质

- 插值所有控制点（曲线经过 $\mathbf{P}_i$）
- $C^1$ 连续（位置和速度连续）
- 局部控制（改一个点只影响 4 段）
- 不需要求解线性系统（显式公式）

---

## 五次 Hermite 样条

对于需要 $C^2$ 连续性的轨迹规划，使用五次 Hermite：

给定：$\gamma(0)$, $\gamma(1)$, $\gamma'(0)$, $\gamma'(1)$, $\gamma''(0)$, $\gamma''(1)$（共 6 个条件，确定 5 次多项式）。

基函数 $H_i^5(t)$ 满足类似三次情形的 Kronecker delta 条件。一般公式：

$$
\gamma(t) = \sum_{k=0}^5 c_k t^k
$$

系数由 6 个端点条件组成的线性方程组确定：

$$
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 3 & 4 & 5 \\
0 & 0 & 2 & 0 & 0 & 0 \\
0 & 0 & 2 & 6 & 12 & 20
\end{bmatrix}
\begin{bmatrix} c_0 \\ c_1 \\ c_2 \\ c_3 \\ c_4 \\ c_5 \end{bmatrix}
=
\begin{bmatrix}
\gamma(0) \\ \gamma(1) \\ \gamma'(0) \\ \gamma'(1) \\ \gamma''(0) \\ \gamma''(1)
\end{bmatrix}
$$

---

## 轨迹优化中的应用

Hermite 表示在机器人轨迹规划中特别有用，因为：

### 1. 直接控制边界状态

轨迹段 $[t_i, t_{i+1}]$ 的起点和终点位置、速度、加速度可直接指定为优化约束。

### 2. 最小化 snap/jerk 轨迹

对于无人机等 differentially flat 系统，常需求解：

$$
\min_{\gamma} \int_0^T \|\gamma^{(k)}(t)\|^2 dt
$$

使用分段多项式（每段 Hermite 形式），此问题化为**二次规划（QP）**：

$$
\min_{\mathbf{c}} \mathbf{c}^\top \mathbf{Q} \mathbf{c} \quad \text{s.t.} \quad \mathbf{A}\mathbf{c} = \mathbf{b}
$$

其中 $\mathbf{c}$ 为各段系数拼接，$\mathbf{Q}$ 可预计算（与段长有关），约束 $\mathbf{A}\mathbf{c} = \mathbf{b}$ 编码端点条件和段间连续性。

### 3. MINCO 参数化

**MINCO**（Minimum Control effort）是近年轨迹优化中的高效参数化（Zhou et al., 2019），本质上是：

> 将多段多项式轨迹的连续性条件隐式编码进参数化，使得无约束优化就可以产生 $C^k$ 连续轨迹。

其在 Hermite 表示的基础上做了巧妙的时空解耦，避免了显式约束，大幅提高了优化效率。

---

## 与 B-spline 的比较

| 特性 | Hermite | B-spline |
|------|---------|----------|
| 基函数 | Hermite 基 | B-spline 基 |
| 插值对象 | 位置 + 导数 | 仅位置 |
| 导数控制 | 直接指定 | 通过控制点间接决定 |
| 连续性 | 由导数匹配决定 | 由节点重数决定（更灵活） |
| 计算 | 封闭解 | 需要递推 / 矩阵 |
| 典型应用 | 轨迹规划、动画 | CAD、曲面建模 |

---

## Cardinal 样条

Cardinal 样条是 Catmull-Rom 的推广，引入张力参数 $c \in [0,1]$：

$$
\mathbf{v}_i = (1 - c) \frac{\mathbf{P}_{i+1} - \mathbf{P}_{i-1}}{t_{i+1} - t_{i-1}}
$$

- $c = 0$：Catmull-Rom（最大弯曲）
- $c = 0.5$：标准 Cardinal
- $c \to 1$：趋近于分段线性（张力极大，无弯曲）

张力参数提供了额外的一维形状控制。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 端点导数可直接指定 | 需要提供导数信息（或从数据估计） |
| 封闭解，无需解方程组 | 高次时可能振荡 |
| 与 Bézier 等价（三次） | 没有 B-spline 的局部支撑优势 |
| 非常适合轨迹规划 | 全局连续性需手动匹配 |
