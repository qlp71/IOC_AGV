# Fourier 曲线

## 定义

Fourier 级数将周期函数展开为三角基的线性组合。对于周期为 $T$ 的函数 $f$：

$$
f(x) = \frac{a_0}{2} + \sum_{k=1}^\infty \left(a_k \cos\left(\frac{2\pi k x}{T}\right) + b_k \sin\left(\frac{2\pi k x}{T}\right)\right)
$$

其中 Fourier 系数由正交投影给出：

$$
\begin{aligned}
a_k &= \frac{2}{T} \int_0^T f(x) \cos\left(\frac{2\pi k x}{T}\right) dx, \quad k = 0,1,2,\dots \\
b_k &= \frac{2}{T} \int_0^T f(x) \sin\left(\frac{2\pi k x}{T}\right) dx, \quad k = 1,2,\dots
\end{aligned}
$$

### 复指数形式

利用 Euler 公式 $e^{i\theta} = \cos\theta + i\sin\theta$：

$$
f(x) = \sum_{k=-\infty}^\infty c_k e^{2\pi i k x / T}, \quad c_k = \frac{1}{T} \int_0^T f(x) e^{-2\pi i k x / T} dx
$$

其中 $c_k = \frac{a_k - i b_k}{2}$（$k>0$），$c_{-k} = \overline{c_k}$（对于实函数），$c_0 = a_0/2$。

---

## 有限维截断逼近

实际中使用截断 Fourier 级数（三角多项式）作为逼近：

$$
f_n(x) = \frac{a_0}{2} + \sum_{k=1}^n \left(a_k \cos\left(\frac{2\pi k x}{T}\right) + b_k \sin\left(\frac{2\pi k x}{T}\right)\right)
$$

这是 $2n+1$ 维函数空间中的最佳 $L^2$ 逼近（由正交投影的性质）：

$$
\|f - f_n\|_{L^2} = \min_{g \in \mathcal{T}_n} \|f - g\|_{L^2}
$$

其中 $\mathcal{T}_n = \operatorname{span}\{1, \cos(\omega x), \sin(\omega x), \dots, \cos(n\omega x), \sin(n\omega x)\}$ 为次数 $\leq n$ 的三角多项式空间。

---

## 离散 Fourier 变换（DFT）与拟合

### 离散情形

实际数据为等距采样点 $\{(x_j, y_j)\}_{j=0}^{N-1}$，$x_j = j \cdot T/N$。离散 Fourier 系数由 DFT 给出：

$$
Y_k = \sum_{j=0}^{N-1} y_j e^{-2\pi i j k / N}, \quad k = 0,\dots,N-1
$$

逆变换（IDFT）：

$$
y_j = \frac{1}{N} \sum_{k=0}^{N-1} Y_k e^{2\pi i j k / N}
$$

系数与连续 Fourier 系数的关系（假设 $y_j = f(x_j)$ 且 $f$ 带限）：

$$
Y_k \approx N c_k \quad \text{（当 $f$ 的频谱限制在 Nyquist 频率内）}
$$

### 使用 DFT 做曲线拟合

1. 对数据 $\{y_j\}$ 执行 FFT，得频谱 $\{Y_k\}$
2. 截断高频成分（保留 $|k| \leq n$ 的系数，$n \ll N/2$）
3. 逆 FFT 得光滑曲线（或直接用三角函数展开）

计算复杂度：$O(N \log N)$（FFT）。

---

## 逼近性质

### 1. 稠密性（Fejér 定理）

三角多项式在 $C(\mathbb{T})$（周期连续函数空间）中稠密。在 $L^2$ 中则更强：$\overline{\mathcal{T}} = L^2(\mathbb{T})$。

### 2. 收敛速率

若 $f \in C^p$ 且 $f^{(p)}$ 分段连续，则：

$$
|a_k|, |b_k| = O(k^{-p})
$$

换句话说：**函数越光滑，Fourier 系数衰减越快**。这意味着光滑函数只需很少的 Fourier 系数即可高精度逼近。

具体地（利用分部积分）：

$$
c_k = \frac{1}{ik} \widehat{f'}(k) = \cdots = \frac{1}{(ik)^p} \widehat{f^{(p)}}(k)
$$

故 $|c_k| \leq \frac{\|f^{(p)}\|_{L^1}}{|k|^p}$。

### 3. $L^2$ 最优性

截断 Fourier 级数 $f_n$ 是 $f$ 在 $\mathcal{T}_n$ 中的**最佳 $L^2$ 逼近**（最小二乘意义上的最优线性逼近），因为三角基是 $L^2$ 的正交基。

### 4. Parseval 等式

$$
\frac{1}{T}\int_0^T |f(x)|^2 dx = \frac{a_0^2}{4} + \frac{1}{2}\sum_{k=1}^\infty (a_k^2 + b_k^2) = \sum_{k=-\infty}^\infty |c_k|^2
$$

这为截断误差提供了精确的量化工具：保留的系数能量占比直接决定了逼近精度。

---

## Gibbs 现象

若 $f$ 有跳跃间断点，则：

1. Fourier 级数在间断点处收敛到跳跃两侧的算术平均
2. 在间断点附近出现过冲（overshoot），幅度约为跳跃的 $\sim 9\%$
3. 过冲不随 $n$ 增加而消失（只变窄）

这是 Fourier 级数处理非光滑函数时的根本局限。数学上对应 Dirichlet 核 $\frac{\sin((n+1/2)x)}{\sin(x/2)}$ 的旁瓣。

**缓解方法**：
- 使用平滑窗函数（Fejér 求和 = Cesàro 平均，$\sigma$-因子法）
- 转向小波或多尺度方法（见 `07_wavelet.md`）

---

## 带惩罚的 Fourier 拟合

为避免过拟合噪声，对 Fourier 系数施加 Tikhonov 正则化：

$$
\min_{\{a_k, b_k\}} \sum_{j=0}^{N-1} \left|y_j - f_n(x_j)\right|^2 + \lambda \sum_{k=1}^n k^{2p}(a_k^2 + b_k^2)
$$

惩罚项 $\sum k^{2p}(a_k^2 + b_k^2)$ 等价于 $\int |f^{(p)}|^2 dx$（利用导数与 Fourier 系数的关系 $\widehat{f^{(p)}}(k) = (ik)^p \hat{f}(k)$）。

这产生频域中的收缩估计：

$$
\tilde{c}_k = \frac{c_k^{\text{LS}}}{1 + \lambda k^{2p}}
$$

其中 $c_k^{\text{LS}}$ 为普通最小二乘系数。$\lambda$ 越大，高频系数被衰减越强。

---

## 周期曲线参数方程

对于闭曲线（如轮廓、轨道），分别对每个坐标分量做 Fourier 展开：

$$
\begin{aligned}
x(t) &= \frac{a_{0x}}{2} + \sum_{k=1}^n \left(a_{kx} \cos(2\pi k t) + b_{kx} \sin(2\pi k t)\right) \\
y(t) &= \frac{a_{0y}}{2} + \sum_{k=1}^n \left(a_{ky} \cos(2\pi k t) + b_{ky} \sin(2\pi k t)\right)
\end{aligned}
$$

参数 $t \in [0,1]$ 为归一化弧长（或实际弧长参数化）。

在计算机视觉中，这被称为**椭圆 Fourier 描述子**（Elliptic Fourier Descriptors，Kuhl & Giardina, 1982），广泛用于形状分析和识别。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 正交基（$L^2$ 最佳逼近） | 全局支撑（改一点影响所有系数） |
| FFT 计算 $O(N \log N)$ | 不适合非周期数据（需加窗或周期化） |
| 无限光滑（$C^\infty$） | Gibbs 现象（间断附近过冲） |
| 频域分析自然（谱方法） | 无局部形状控制 |
| 导数/积分简单（乘以/除以频率） | 收敛速率依赖函数光滑性 |
| Parseval 等式精确量化误差 | 对非等距采样需用非均匀 FFT（NUFFT） |

---

## 频域滤波

Fourier 表示允许直接在频域进行滤波操作：

- **低通滤波**：截断 $|k| > K$，保留光滑趋势
- **高通滤波**：去除 $|k| < K$，提取细节/边缘
- **带通/带阻**：选择特定频段保留或去除

滤波后的曲线通过逆 FFT 重建。这使得 Fourier 方法在信号处理和周期性数据分析中不可替代。
