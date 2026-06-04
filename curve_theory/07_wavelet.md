# 小波曲线（Wavelet）

## 核心思想

Fourier 分析用全局的正弦/余弦基，使用时域和频域信息不能同时局部化。小波变换通过**缩放（scaling）和平移（shifting）**一个母小波 $\psi$，实现时频局部化：

$$
\psi_{j,k}(x) = 2^{-j/2} \psi(2^{-j} x - k), \quad j, k \in \mathbb{Z}
$$

其中 $j$ 控制尺度（频率），$k$ 控制位置。$2^{-j/2}$ 为 $L^2$ 归一化因子。

---

## 离散小波变换（DWT）

函数 $f \in L^2(\mathbb{R})$ 的小波展开为：

$$
f(x) = \sum_{k \in \mathbb{Z}} c_{J_0,k} \phi_{J_0,k}(x) + \sum_{j \leq J_0} \sum_{k \in \mathbb{Z}} d_{j,k} \psi_{j,k}(x)
$$

其中：
- $\phi$ 为**尺度函数**（scaling function），对应低频近似
- $\psi$ 为**母小波**，对应高频细节
- $c_{J_0,k}$ 为近似系数，$d_{j,k}$ 为细节系数

系数由内积给出：

$$
\begin{aligned}
c_{j,k} &= \langle f, \phi_{j,k} \rangle = \int f(x) \phi_{j,k}(x) dx \\
d_{j,k} &= \langle f, \psi_{j,k} \rangle = \int f(x) \psi_{j,k}(x) dx
\end{aligned}
$$

---

## 多分辨分析（MRA）

多分辨分析（Mallat, 1989）是小波理论的数学基础。一个 MRA 由嵌套子空间序列组成：

$$
\cdots \subset V_2 \subset V_1 \subset V_0 \subset V_{-1} \subset \cdots \subset L^2(\mathbb{R})
$$

满足：

1. **稠密性**：$\overline{\bigcup_j V_j} = L^2(\mathbb{R})$
2. **分离性**：$\bigcap_j V_j = \{0\}$
3. **尺度关系**：$f(x) \in V_j \iff f(2x) \in V_{j-1}$
4. **平移不变性**：$f(x) \in V_0 \iff f(x-k) \in V_0$，$\forall k \in \mathbb{Z}$
5. **Riesz 基**：存在 $\phi \in V_0$ 使得 $\{\phi(x-k)\}_{k \in \mathbb{Z}}$ 构成 $V_0$ 的 Riesz 基

小波空间 $W_j$ 定义为 $V_{j-1} = V_j \oplus W_j$（$W_j$ 为 $V_j$ 在 $V_{j-1}$ 中的正交补）：

$$
f_{j-1} = \underbrace{f_j}_{\text{粗糙近似}} + \underbrace{g_j}_{\text{细节}}
$$

### 双尺度方程

尺度函数和小波函数满足：

$$
\begin{aligned}
\phi(x) &= \sqrt{2} \sum_k h_k \phi(2x - k) \\
\psi(x) &= \sqrt{2} \sum_k g_k \phi(2x - k)
\end{aligned}
$$

其中 $\{h_k\}$ 为低通滤波器系数，$\{g_k\}$ 为高通滤波器系数（$g_k = (-1)^k h_{1-k}$ 对于正交小波）。

---

## Mallat 算法（快速离散小波变换）

Mallat 算法利用双尺度方程将 DWT 实现为级联滤波器组，计算复杂度为 $O(N)$。

### 分解（Decomposition）

从最细尺度 $j$ 的近似系数 $\{c_{j,k}\}$ 出发：

$$
\begin{aligned}
c_{j+1,k} &= \sum_m h_{m-2k} c_{j,m} \quad \text{（低通，下采样）}\\
d_{j+1,k} &= \sum_m g_{m-2k} c_{j,m} \quad \text{（高通，下采样）}
\end{aligned}
$$

### 重构（Reconstruction）

从粗尺度向上恢复：

$$
c_{j-1,k} = \sum_m h_{k-2m} c_{j,m} + \sum_m g_{k-2m} d_{j,m}
$$

### 计算复杂度

每一层分解/重构的运算量为 $O(N_j)$，$N_j$ 为该层的系数个数。$\sum_j N_j = N + N/2 + N/4 + \cdots \leq 2N$，故总复杂度为 $O(N)$。比 FFT 的 $O(N \log N)$ 还快。

---

## 小波阈值去噪（曲线拟合的核心应用）

小波在曲线/信号拟合中最强大的应用是**非线性阈值去噪**（Donoho & Johnstone, 1994）。

### 算法（小波收缩，WaveShrink）

1. 对含噪数据 $\mathbf{y} = \mathbf{f} + \boldsymbol{\varepsilon}$ 做 DWT，得到系数 $\{d_{j,k}\}$
2. 对细节系数施加阈值函数 $\eta_\lambda(\cdot)$
3. 逆 DWT 重建去噪信号

### 硬阈值

$$
\eta_\lambda^{\text{hard}}(d) = \begin{cases}
d, & |d| \geq \lambda \\
0, & |d| < \lambda
\end{cases}
$$

### 软阈值

$$
\eta_\lambda^{\text{soft}}(d) = \operatorname{sign}(d) \cdot \max(|d| - \lambda, 0)
$$

软阈值产生更光滑的结果（收缩而非截断），且是 $L^2$ 风险意义上的渐近最优。

### 阈值选择

- **Universal 阈值**（Donoho & Johnstone）：

$$
\lambda = \sigma \sqrt{2 \log N}
$$

  其中 $\sigma$ 为噪声标准差（可从最细尺度的小波系数中位数估计：$\hat{\sigma} = \operatorname{MAD}(d_{1,k}) / 0.6745$）。

- **SURE 阈值**（Stein's Unbiased Risk Estimate）：

  对每个尺度自适应选择 $\lambda_j$，最小化 Stein 无偏风险估计。

- **Bayes 阈值**：在贝叶斯框架下推导阈值（如 Laplace 先验对应软阈值）

### 为什么小波去噪有效

光滑函数的小波系数是**稀疏的**：只有少数大系数对应奇异性/边缘，大部分系数很小。加性噪声在所有系数上均匀散布。阈值操作保留大系数（信号）而去除小系数（噪声），从而实现自适应平滑——在平坦区域强去噪，在突变处（尖峰、边缘）保持细节。

Fourier 方法无法做到这一点：信号的不连续性会散布到所有 Fourier 系数上。

---

## 消失矩

小波 $\psi$ 具有 $N$ 阶消失矩，若：

$$
\int x^k \psi(x) dx = 0, \quad k = 0, 1, \dots, N-1
$$

### 意义

1. **稀疏性**：对 $C^N$ 光滑函数，小波系数以 $\sim 2^{-j(N+1/2)}$ 的速度衰减
2. **多项式抑制**：$\leq N-1$ 次多项式成分完全被 $\psi$ 的消失矩"滤除"，只出现在近似系数中
3. **奇异检测**：$N$ 阶消失矩使小波对 $\leq N-1$ 阶导数的间断敏感

### 常见小波族

| 小波 | 消失矩 | 支撑宽度 | 对称性 | 正交性 |
|------|--------|----------|--------|--------|
| Haar | 1 | 1 | 对称 | 正交 |
| Daubechies dbN | N | 2N-1 | 不对称 | 正交 |
| Symlet | N | 2N-1 | 近似对称 | 正交 |
| Coiflet | 2N | 6N-1 | 近似对称 | 正交 |
| Biorthogonal | 可调 | 可调 | 对称 | 双正交 |
| Meyer | ∞ | 无限 | 对称 | 正交 |

---

## 小波与样条的联系

B-spline 可以构造 Battle-Lemarié 小波：

- 尺度函数 $\phi$ 取为 $m$ 阶 B-spline
- 对应的半正交小波具有 $m+1$ 阶消失矩
- 结合了样条的光滑性和小波的多尺度特性

这体现了各种基函数方法之间的深层联系：B-spline → 尺度函数 → MRA → 小波。

---

## 曲线拟合中的应用

### 1. 非参数回归（小波平滑）

$$
\hat{f}(x) = \sum_k \hat{c}_{J_0,k} \phi_{J_0,k}(x) + \sum_{j \leq J_0} \sum_k \eta_\lambda(\hat{d}_{j,k}) \psi_{j,k}(x)
$$

即：DWT → 阈值 → IDWT。这是**空间自适应**的平滑器，在函数变化剧烈处自动减少平滑量。

### 2. 多尺度曲线编辑

可以在不同分辨率级别独立编辑曲线形状，这在 CAD 和动画中可用于 coarse-to-fine 的形状设计。

### 3. 压缩表示

由于光滑曲线的小波系数稀疏，只需保留少量大系数即可高保真重建（JPEG2000 标准的数学基础）。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 时频局部化（同时定位时间和频率） | 平移非不变（DWT 非平移协变） |
| $O(N)$ 复杂度（比 FFT 快） | 理论门槛高（MRA 框架） |
| 自适应平滑（保留边缘） | 离散尺度非连续（二进尺度） |
| 稀疏表示（压缩感知） | 边界处理复杂 |
| 非线性阈值接近 minimax 最优 | 小波选择依赖问题 |
| 多尺度分析天然 | 二维+设计复杂（张量积局限） |

---

## 平移不变小波变换

为克服 DWT 的平移敏感性，可使用：

- **平稳小波变换（SWT / undecimated DWT）**：不做下采样，所有子带保持原长，$O(N \log N)$
- **双树复小波变换（DT-CWT）**：近似平移不变 + 方向选择性
- **最大重叠离散小波变换（MODWT）**：SWT 的等效形式
