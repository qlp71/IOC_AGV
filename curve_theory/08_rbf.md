# 径向基函数（RBF）曲线

## 定义

径向基函数方法将函数表示为以数据点为中心的径向函数的线性组合：

$$
f(\mathbf{x}) = \sum_{i=1}^n w_i \phi(\|\mathbf{x} - \mathbf{c}_i\|)
$$

其中：
- $\phi: \mathbb{R}_{\geq 0} \to \mathbb{R}$ 为**径向基函数**（仅依赖于距离）
- $\mathbf{c}_i$ 为**中心点**（通常取为数据点 $\mathbf{x}_i$）
- $w_i$ 为权重系数
- $\|\cdot\|$ 通常为 Euclidean 范数

对于曲线情形（参数曲线 $\gamma: \mathbb{R} \to \mathbb{R}^d$），$\mathbf{x} \in \mathbb{R}$ 为参数，$\mathbf{c}_i \in \mathbb{R}$ 为参数域中的中心。

---

## 常见径向基函数

### 1. Gaussian

$$
\phi(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right) = \exp(-\varepsilon^2 r^2)
$$

其中 $\varepsilon = 1/(\sqrt{2}\sigma)$ 为形状参数。$C^\infty$，正定。

### 2. Multiquadric（MQ）

$$
\phi(r) = \sqrt{1 + (\varepsilon r)^2}
$$

广义 multiquadric：$\phi(r) = (1 + (\varepsilon r)^2)^{\nu/2}$，$\nu \in \mathbb{R}_{\neq 0, -2, -4,\dots}$。

### 3. Inverse Multiquadric（IMQ）

$$
\phi(r) = \frac{1}{\sqrt{1 + (\varepsilon r)^2}}
$$

正定，$C^\infty$。

### 4. Thin-plate Spline（TPS）

$$
\phi(r) = r^2 \log r \quad \text{（二维）}
$$

$\phi(r) = r^{2k-d} \log r$（$d$ 为偶数），$\phi(r) = r^{2k-d}$（$d$ 为奇数），其中 $2k-d > 0$。

这是**条件正定**函数，需要附加多项式项。

### 5. Wendland 紧支撑函数

$$
\phi(r) = (1 - r)_+^4 (4r + 1) \quad \text{（} C^2 \text{，维度} \leq 3\text{）}
$$

紧支撑 RBF 产生**稀疏插值矩阵**，适合大规模问题。

### 基函数选择指南

| $\phi$ | 光滑性 | 正定性 | 矩阵性态 | 典型场景 |
|--------|--------|--------|----------|----------|
| Gaussian | $C^\infty$ | 正定 | 依赖 $\varepsilon$ | 通用、高维 |
| MQ | $C^\infty$ | 条件正定（1阶） | 依赖 $\varepsilon$ | 散乱数据插值 |
| IMQ | $C^\infty$ | 正定 | 较好 | 光滑插值 |
| TPS | $C^2$ | 条件正定（$m$ 阶） | 好 | 几何/物理插值 |
| Wendland | $C^{2k}$ | 正定 | 稀疏 | 大规模问题 |

---

## RBF 插值

### 精确插值

给定数据 $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$，要求 $f(\mathbf{x}_i) = \mathbf{y}_i$：

$$
\begin{bmatrix}
\phi(\|\mathbf{x}_1 - \mathbf{x}_1\|) & \cdots & \phi(\|\mathbf{x}_1 - \mathbf{x}_n\|) \\
\vdots & \ddots & \vdots \\
\phi(\|\mathbf{x}_n - \mathbf{x}_1\|) & \cdots & \phi(\|\mathbf{x}_n - \mathbf{x}_n\|)
\end{bmatrix}
\begin{bmatrix} w_1 \\ \vdots \\ w_n \end{bmatrix}
= \begin{bmatrix} \mathbf{y}_1 \\ \vdots \\ \mathbf{y}_n \end{bmatrix}
$$

即 $\mathbf{A} \mathbf{w} = \mathbf{y}$。

### 条件正定与多项式增广

对于条件正定函数（如 MQ、TPS），需要在展开中添加多项式项以保证解的唯一性：

$$
f(\mathbf{x}) = \sum_{i=1}^n w_i \phi(\|\mathbf{x} - \mathbf{x}_i\|) + \sum_{j=1}^m \beta_j p_j(\mathbf{x})
$$

其中 $\{p_j\}$ 为 $\leq k$ 次多项式空间的基（$k$ 取决于 $\phi$ 的条件正定阶数）。附加正交性约束 $\sum_i w_i p_j(\mathbf{x}_i) = 0$。

增广系统：

$$
\begin{bmatrix}
\mathbf{A} & \mathbf{P} \\
\mathbf{P}^\top & \mathbf{0}
\end{bmatrix}
\begin{bmatrix} \mathbf{w} \\ \boldsymbol{\beta} \end{bmatrix}
= \begin{bmatrix} \mathbf{y} \\ \mathbf{0} \end{bmatrix}
$$

其中 $P_{ij} = p_j(\mathbf{x}_i)$。对于曲线拟合（$\mathbf{x} \in \mathbb{R}$），通常附加 $\{1, x\}$（线性多项式项，对应 TPS 的条件正定阶数为 2）。

---

## 形状参数 $\varepsilon$ 的影响

形状参数 $\varepsilon$ 控制基函数的"宽度"：

- $\varepsilon \to 0$：基函数变平（宽），矩阵 $\mathbf{A}$ **趋于病态**（所有行趋同），但逼近精度提高
- $\varepsilon \to \infty$：基函数变尖（窄），矩阵 $\mathbf{A} \to \mathbf{I}$（条件数→1），但逼近能力下降

这是 RBF 理论中的核心**精度-稳定性权衡**（trade-off principle，Schaback, 1995）。

### 最优 $\varepsilon$ 选择

- **交叉验证**：在数据上做 LOOCV 选择 $\varepsilon$
- **Rippa 方法**：LOOCV 对 RBF 有 $O(n^2)$ 的快速计算公式
- **自适应 $\varepsilon$**：对不同中心使用不同的 $\varepsilon_i$（variable shape parameter）

---

## 数值稳定性

### 条件数

RBF 插值矩阵 $\mathbf{A}$ 的条件数随 $n$ 增加和 $\varepsilon \to 0$ 迅速增长：

$$
\kappa(\mathbf{A}) \sim \exp(c / \varepsilon)
$$

### 缓解方法

1. **稳定化求解**：对正定 RBF 使用 Cholesky + 对角线扰动（Tikhonov 正则化）
2. **RBF-QR**（Fornberg & Piret, 2007）：对 Gaussian RBF 在 $\varepsilon \to 0$ 的极限使用 Chebyshev 基展开，避免病态
3. **Contour-Padé**：复平面围道积分方法
4. **局部 RBF**：只用 $k$ 个最近邻中心（$k \ll n$），稀疏化矩阵

---

## 最小二乘 RBF 拟合

当数据量 $n$ 很大时，不使用全部数据点作为中心，而选择 $m \ll n$ 个中心：

$$
\min_{\mathbf{w}} \sum_{i=1}^n \left\|\sum_{j=1}^m w_j \phi(\|\mathbf{x}_i - \mathbf{c}_j\|) - \mathbf{y}_i\right\|^2
$$

即求解超定系统：

$$
\min_{\mathbf{w}} \|\mathbf{\Phi} \mathbf{w} - \mathbf{y}\|^2
$$

其中 $\Phi_{ij} = \phi(\|\mathbf{x}_i - \mathbf{c}_j\|) \in \mathbb{R}^{n \times m}$。正规方程：

$$
\mathbf{\Phi}^\top \mathbf{\Phi} \mathbf{w} = \mathbf{\Phi}^\top \mathbf{y}
$$

计算复杂度 $O(n m^2 + m^3)$，当 $m \ll n$ 时远小于精确插值的 $O(n^3)$。

---

## RBF 与样条的联系

### Thin-plate spline 变分性质

TPS 是以下变分问题的解：

$$
\min_f \sum_{i=1}^n (f(\mathbf{x}_i) - y_i)^2 + \lambda \int_{\mathbb{R}^d} \|\nabla^2 f\|^2 d\mathbf{x}
$$

这直接推广了自然三次样条（见 `04_cubic_spline.md`）到高维。实际上，$\phi(r) = r^2 \log r$ 是 biharmonic 算子 $\Delta^2$ 的 Green 函数。

### 再生核 Hilbert 空间（RKHS）

每个正定核 $\phi$ 对应一个唯一的 RKHS $\mathcal{H}_\phi$。RBF 插值等价于在 $\mathcal{H}_\phi$ 中寻找具有最小范数的插值函数：

$$
\min_{f \in \mathcal{H}_\phi} \|f\|_{\mathcal{H}_\phi} \quad \text{s.t.} \quad f(\mathbf{x}_i) = y_i, \; i=1,\dots,n
$$

这提供了 RBF 方法的泛函分析统一视角。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 对散乱数据天然适用（无需网格） | 稠密矩阵（$O(n^3)$ 求解） |
| 高维自然推广（维度无关） | 条件数随 $n$ 快速恶化 |
| 变分最优性（RKHS 框架） | 形状参数选择敏感 |
| 光滑性极高（$C^\infty$ 基可选） | 外推差（基函数局部衰减后无信息） |
| TPS 对应的物理直觉（弯曲能量） | 紧支撑 RBF 降低精度 |
| 任意维度统一 | 大 $n$ 需要局部化/稀疏化技巧 |

---

## 大规模 RBF 的加速方法

- **快速多极子法（FMM）**：$O(n \log n)$ 矩阵-向量乘，用于迭代求解
- **局部 RBF（RBF-FD）**：结合有限差分的局部近似，产生稀疏系统
- **分区法（Partition of Unity RBF）**：将域分解为重叠子域，每子域独立 RBF 插值后加权混合
