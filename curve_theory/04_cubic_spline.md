# 三次样条（Cubic Spline）

## 定义

给定节点 $a = x_0 < x_1 < \cdots < x_n = b$ 及对应的函数值 $\{y_i\}_{i=0}^n$，三次样条插值 $s(x)$ 满足：

1. $s(x_i) = y_i, \quad i = 0,\dots,n$（插值条件）
2. 在每段 $[x_i, x_{i+1}]$ 上，$s(x)$ 是三次多项式
3. $s \in C^2[a,b]$（位置、一阶导数、二阶导数全局连续）

---

## 分段表示

在区间 $[x_i, x_{i+1}]$ 上，令 $h_i = x_{i+1} - x_i$，$s(x)$ 可写为：

$$
s_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
$$

其中系数由插值条件和连续性条件确定。

### 用节点值和二阶导数表达

令 $M_i = s''(x_i)$（待定的二阶导数值），则在 $[x_i, x_{i+1}]$ 上：

$$
\begin{aligned}
a_i &= y_i \\
b_i &= \frac{y_{i+1} - y_i}{h_i} - \frac{h_i}{6}(2M_i + M_{i+1}) \\
c_i &= \frac{M_i}{2} \\
d_i &= \frac{M_{i+1} - M_i}{6h_i}
\end{aligned}
$$

这是分段 Hermite 插值的形式（分别插值位置和二阶导数）。

---

## 三弯矩方程（确定 $M_i$）

由 $s'$ 在内部节点的连续性 $s_{i-1}'(x_i) = s_i'(x_i)$，导出：

$$
h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1} = 6\left(\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right)
$$

即**三弯矩方程**，$i = 1,\dots,n-1$。

写成矩阵形式：

$$
\begin{bmatrix}
2(h_0+h_1) & h_1 & 0 & \cdots & 0 \\
h_1 & 2(h_1+h_2) & h_2 & \cdots & 0 \\
0 & h_2 & 2(h_2+h_3) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & h_{n-1} \\
0 & 0 & 0 & h_{n-1} & 2(h_{n-2}+h_{n-1})
\end{bmatrix}
\begin{bmatrix} M_1 \\ M_2 \\ M_3 \\ \vdots \\ M_{n-1} \end{bmatrix}
= 6 \begin{bmatrix}
\frac{y_2-y_1}{h_1} - \frac{y_1-y_0}{h_0} \\
\frac{y_3-y_2}{h_2} - \frac{y_2-y_1}{h_1} \\
\vdots
\end{bmatrix}
$$

这是**三对角**系统，可用 Thomas 算法在 $O(n)$ 时间内精确求解。

---

## 边界条件

$n+1$ 个未知数 $M_0,\dots,M_n$ 只有 $n-1$ 个方程，需要额外 2 个边界条件。

### 1. 自然边界条件（Natural Spline）

$$
M_0 = 0, \quad M_n = 0
$$

两端二阶导数为零。这产生著名的**自然三次样条**。

### 2. 夹持边界条件（Clamped Spline）

给定端点一阶导数 $s'(a) = \alpha$，$s'(b) = \beta$：

$$
\begin{aligned}
2h_0 M_0 + h_0 M_1 &= 6\left(\frac{y_1 - y_0}{h_0} - \alpha\right) \\
h_{n-1} M_{n-1} + 2h_{n-1} M_n &= 6\left(\beta - \frac{y_n - y_{n-1}}{h_{n-1}}\right)
\end{aligned}
$$

### 3. 周期边界条件

若 $y_0 = y_n$ 且假设 $s'(a) = s'(b)$，$s''(a) = s''(b)$。

### 4. "Not-a-knot" 边界条件

假设 $s$ 在 $x_1$ 和 $x_{n-1}$ 处也是 $C^3$（即前两段和后两段分别由同一三次多项式定义），常用于无额外信息时的默认选择。

---

## 弯曲能量最小化：变分原理

自然三次样条最深刻的理论性质是：

> 在所有满足插值条件且 $f \in H^2[a,b]$ 的函数中，自然三次样条**唯一地**最小化弯曲能量：

$$
\min_f \int_a^b |f''(x)|^2 dx
$$

**变分证明**：

设 $\eta(x)$ 为任意 $H^2$ 扰动满足 $\eta(x_i) = 0$。考虑：

$$
J(s + \varepsilon \eta) = \int_a^b (s'' + \varepsilon \eta'')^2 dx
$$

一阶变分：

$$
\frac{d}{d\varepsilon} J(s + \varepsilon \eta)\Big|_{\varepsilon=0} = 2\int_a^b s'' \eta'' dx
$$

分部积分（利用 $s$ 的分段性质）：

$$
\int_a^b s'' \eta'' dx = \underbrace{[s'' \eta']_{a}^{b}}_{\text{边界项}} - \underbrace{\int_a^b s''' \eta' dx}_{\text{再次分部积分}}
$$

由于 $s^{(4)} = 0$ 在每段内部（三次多项式），再次分部积分得到边界项和分段节点处的跳跃项。自然边界条件 $s''(a) = s''(b) = 0$ 和 $s \in C^2$ 使得所有项消失，故一阶变分为零 → 驻点。

二阶变分：$\frac{d^2}{d\varepsilon^2} J = 2\int (\eta'')^2 dx > 0$，故为严格极小值。

**物理意义**：自然三次样条对应通过给定数据点的弹性细梁（elastica）的平衡形状。

---

## 光滑样条（Smoothing Spline）

当数据含有噪声时，纯插值不合适。引入**光滑样条**（Reinsch, 1967）：

$$
\min_{s \in H^2} \sum_{i=1}^m (s(x_i) - y_i)^2 + \lambda \int_a^b |s''(x)|^2 dx
$$

其中 $\lambda > 0$ 为光滑参数：
- $\lambda \to 0$：趋近于插值样条
- $\lambda \to \infty$：趋近于线性最小二乘

光滑样条的显式解也是自然三次样条，节点在数据点处。这可以通过 RKHS（再生核 Hilbert 空间）理论严格推导。

### 光滑参数的选择

- **GCV（广义交叉验证）**：

$$
\mathrm{GCV}(\lambda) = \frac{\frac{1}{m}\|(\mathbf{I} - \mathbf{S}_\lambda)\mathbf{y}\|^2}{\left[\frac{1}{m}\operatorname{tr}(\mathbf{I} - \mathbf{S}_\lambda)\right]^2}
$$

  其中 $\mathbf{S}_\lambda$ 为光滑矩阵（$\hat{\mathbf{y}} = \mathbf{S}_\lambda \mathbf{y}$）。

- **REML**（限制最大似然）：在贝叶斯框架下将光滑样条视为 Gaussian 过程

---

## B-spline 与三次样条的关系

三次样条是三次 B-spline 的特殊情况（插值问题），而 B-spline 提供了更灵活的基表示。实际上，任何三次样条都可以唯一地表示为三次 B-spline 展开：

$$
s(x) = \sum_i c_i N_{i,3}(x)
$$

其中系数 $c_i$ 由插值条件确定。B-spline 基的优势在于局部支撑和更好的数值稳定性（尤其对于大规模问题）。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| $C^2$ 全局光滑 | 全局插值（改一点影响整体） |
| 三对角系统 $O(n)$ 求解 | 外推不可靠 |
| 弯曲能量最小（自然样条） | 无局部形状控制 |
| 理论基础优美（变分法） | 对异常值敏感（需用光滑样条） |
| 简单直接 | 不能精确表示圆锥曲线 |

---

## 实现要点

### Thomas 算法（三对角求解器）

对于系统 $a_i M_{i-1} + b_i M_i + c_i M_{i+1} = d_i$：

```python
def thomas_solve(a, b, c, d):
    n = len(d)
    c_prime = [c[0] / b[0]]
    d_prime = [d[0] / b[0]]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[-1]
        c_prime.append(c[i] / denom if i < n-1 else 0)
        d_prime.append((d[i] - a[i] * d_prime[-1]) / denom)
    x = [0] * n
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x
```

计算复杂度：$O(n)$，空间中只需存储三对角带。
