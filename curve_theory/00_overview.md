# 曲线拟合与逼近：统一视角

## 核心问题

给定一组数据点 $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^m \subset \mathbb{R}^d \times \mathbb{R}^d$（或参数曲线 $\gamma: [a,b] \to \mathbb{R}^d$ 的采样），我们希望找到一个函数 $f$ 使得：

$$
f(\mathbf{x}_i) \approx \mathbf{y}_i, \quad i = 1,\dots,m
$$

并且 $f$ 具有良好的光滑性、可微性、数值稳定性等附加性质。

## 泛函分析统一框架

几乎所有曲线拟合方法都可以纳入以下框架：

### 1. 选择基函数空间

选定一组基函数 $\{\phi_j\}_{j=1}^n \subset V$，其中 $V$ 是某个函数空间（如 $C^k[a,b]$, $L^2[a,b]$, Sobolev 空间 $H^s$ 等）。

### 2. 表达为线性（或非线性）展开

$$
\gamma(t) = \sum_{j=1}^n \mathbf{c}_j \phi_j(t)
$$

其中 $\mathbf{c}_j \in \mathbb{R}^d$ 为控制系数（控制点）。对于有理情形（如 NURBS），形式为：

$$
\gamma(t) = \frac{\sum_j w_j \mathbf{c}_j \phi_j(t)}{\sum_j w_j \phi_j(t)}
$$

### 3. 通过优化确定系数

最常见的是最小二乘问题：

$$
\min_{\mathbf{c}_j} \sum_{i=1}^m \left\|\gamma(t_i) - \mathbf{p}_i\right\|^2
$$

或带正则化的变分问题：

$$
\min_{\gamma} \underbrace{\sum_{i=1}^m \|\gamma(t_i) - \mathbf{p}_i\|^2}_{\text{数据保真}} + \lambda \underbrace{\int \|\gamma^{(k)}(t)\|^2 dt}_{\text{光滑性惩罚}}
$$

### 4. 理论保证

- **稠密性（逼近能力）**：$\overline{V} = C([a,b])$ 或 $\overline{V} = L^2([a,b])$（如 Weierstrass 定理、Universal Approximation Theorem）
- **收敛速率**：与基函数的逼近阶（approximation order）有关，通常 $O(h^{p+1})$ 量级
- **稳定性**：条件数 $\kappa(\mathbf{A})$ 决定了数值求解的可靠性

---

## 各方法分类与特征对比

| 方法 | 基函数 | 局部性 | 连续性 | 拟合方式 | 典型应用 |
|------|--------|--------|--------|----------|----------|
| Bézier | Bernstein 多项式 | ✗ 全局 | $C^\infty$ 内部 | 最小二乘 / 控制点插值 | 字体设计、插图 |
| B-spline | B-spline 基 $N_{i,p}$ | ✓ 强 | $C^{p-k}$（节点重数 $k$） | 带惩罚的最小二乘 | CAD/CAM、轨迹规划 |
| NURBS | 有理 B-spline | ✓ 强 | 同 B-spline | 加权最小二乘 | 工业 CAD 标准 |
| Cubic Spline | 分段三次多项式 | ✓ 中 | $C^2$ | 三对角线性系统 | 数据插值 |
| Hermite | Hermite 基 | ✓ | 可指定 | 端点条件封闭解 | 机器人轨迹、动画 |
| Fourier | $\{\cos(kx), \sin(kx)\}$ | ✗ 全局 | $C^\infty$ | FFT + 截断 | 周期信号、频域分析 |
| Wavelet | 小波基 $\psi_{j,k}$ | ✓ 强 | 可调 | 阈值收缩 | 信号去噪、多尺度 |
| RBF | $\phi(\|x - c_i\|)$ | 中 | $C^\infty$（Gaussian） | 线性方程组 | 散乱点插值、高维 |
| Neural Network | $\sigma(w^\top x + b)$ | ✓ 强 | 可调 | SGD / Adam | 通用函数逼近、PINN |

---

## 工程选型决策树

```
需要精确圆锥曲线？
├── 是 → NURBS
└── 否
    ├── 需要局部控制？
    │   ├── 是
    │   │   ├── 需要指定端点导数？ → Hermite / Catmull-Rom
    │   │   └── 需要高阶连续性？ → B-spline
    │   └── 否
    │       ├── 数据是周期的？ → Fourier
    │       ├── 数据是散乱高维的？ → RBF
    │       ├── 需要多尺度分析？ → Wavelet
    │       └── 极高复杂度？ → Neural Network
    └── 只需插值，不要多余自由度？ → Cubic Spline
```

---

## 数学工具速查

### 函数空间

- $C^k[a,b]$：$k$ 阶连续可微函数空间
- $L^2[a,b]$：平方可积函数空间，内积 $\langle f, g\rangle = \int_a^b f(t)g(t)dt$
- Sobolev 空间 $H^s[a,b]$：导数也在 $L^2$ 中的函数空间，范数 $\|f\|_{H^s}^2 = \sum_{k=0}^s \|f^{(k)}\|_{L^2}^2$

### 常用范数

- **$L^\infty$（一致范数）**：$\|f\|_\infty = \max_{x\in[a,b]}|f(x)|$ —— Weierstrass 逼近定理使用的范数
- **$L^2$ 范数**：$\|f\|_2 = \left(\int_a^b |f(x)|^2 dx\right)^{1/2}$ —— 最小二乘的自然范数
- **离散 $\ell^2$ 范数**：$\|f\|_{\ell^2} = \left(\sum_{i=1}^m |f(x_i) - y_i|^2\right)^{1/2}$ —— 实际计算使用的范数

### 逼近阶

若基函数空间包含所有 $\leq p$ 次多项式，且函数 $f$ 足够光滑，则有：

$$
\mathrm{dist}(f, V_h) \leq C h^{p+1} \|f^{(p+1)}\|_\infty
$$

其中 $h$ 为网格尺寸。这解释了为什么高次基函数可以更快收敛。

---

## 各文档索引

| 文档 | 内容 |
|------|------|
| [01_bezier.md](01_bezier.md) | Bézier 曲线：Bernstein 基、de Casteljau 算法、最小二乘拟合 |
| [02_bspline.md](02_bspline.md) | B-spline：Cox-de Boor 递推、节点向量、局部支撑、惩罚最小二乘 |
| [03_nurbs.md](03_nurbs.md) | NURBS：有理形式、权重几何意义、圆锥曲线精确表示 |
| [04_cubic_spline.md](04_cubic_spline.md) | 三次样条：弯曲能量最小化、变分推导、三对角系统 |
| [05_hermite.md](05_hermite.md) | Hermite 样条：端点导数控制、Catmull-Rom、轨迹优化 |
| [06_fourier.md](06_fourier.md) | Fourier 曲线：三角基展开、DFT 拟合、Gibbs 现象 |
| [07_wavelet.md](07_wavelet.md) | 小波曲线：多分辨分析、DWT、阈值去噪、消失矩 |
| [08_rbf.md](08_rbf.md) | RBF 曲线：核函数、插值矩阵、条件数分析 |
| [09_neural_network.md](09_neural_network.md) | 神经网络：Universal Approximation Theorem、PINN、基展开视角 |

---

## 参考文献

- Piegl, L., & Tiller, W. (1996). *The NURBS Book*. Springer.
- de Boor, C. (1978). *A Practical Guide to Splines*. Springer.
- Farin, G. (2002). *Curves and Surfaces for CAGD*. Morgan Kaufmann.
- Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
- Wendland, H. (2004). *Scattered Data Approximation*. Cambridge.
- Hornik, K. (1991). Approximation Capabilities of Multilayer Feedforward Networks. *Neural Networks*.
