# 神经网络曲线拟合

## 与经典方法的本质联系

从逼近论角度看，神经网络与其他曲线方法属于同一框架：

$$
f(\mathbf{x}) \approx \sum_{i=1}^N a_i \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)
$$

这本质上仍是**基函数展开**，只是基函数 $\sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$ 本身是可学习的（包含可训练参数 $\mathbf{w}_i$, $b_i$），而经典方法（Bézier、B-spline、Fourier）的基函数是固定的。

---

## Universal Approximation Theorem（万能逼近定理）

### 经典版本（Cybenko, 1989; Hornik et al., 1989）

> 设 $\sigma$ 为非多项式连续函数，则单隐层前馈网络
> $$
> \mathcal{N}_N(\mathbf{x}) = \sum_{i=1}^N a_i \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)
> $$
> 在 $C(K)$（紧集上的连续函数空间）中稠密。

即：$\forall f \in C(K), \forall \varepsilon > 0$，存在 $N$ 和参数 $\{a_i, \mathbf{w}_i, b_i\}$ 使得 $\sup_{\mathbf{x} \in K} |f(\mathbf{x}) - \mathcal{N}_N(\mathbf{x})| < \varepsilon$。

### 定量版本（Barron, 1993）

若 $f$ 的 Fourier 表示满足 $C_f = \int \|\omega\| |\hat{f}(\omega)| d\omega < \infty$，则：

$$
\|f - \mathcal{N}_N\|_{L^2}^2 \leq \frac{C_f^2}{N}
$$

误差以 $O(1/N)$ 速率收敛，且**与输入维度无关**（突破了经典逼近论的维度诅咒！这是神经网络的核心理论优势）。

### 深度网络的表达优势

浅层网络（1 隐层）即可万能逼近，但所需神经元数量可能指数级增长。深度网络（多层）可以**组合性地**表示函数：

- 浅层：$O(\exp(d))$ 神经元
- 深层：$O(\text{poly}(d))$ 神经元

这解释了深度的必要性（如 ReLU 网络的分段线性区域数以指数增长于层数）。

---

## 激活函数与基函数对应

| 激活函数 | 对应经典基 | 特点 |
|----------|-----------|------|
| ReLU $\max(0, x)$ | 线性样条（0 次 B-spline） | 分片线性，二阶导为 0 |
| ReLU$^k$ $\max(0, x)^k$ | $k$ 次 B-spline | $C^{k-1}$ 分片多项式 |
| Sigmoid / tanh | Sigmoidal 基 | 全局支撑，$C^\infty$ |
| 高斯 $\exp(-x^2)$ | Gaussian RBF | 局部支撑，$C^\infty$ |
| Softplus $\log(1+e^x)$ | 光滑 ReLU 近似 | $C^\infty$，处处非零导数 |
| Sine | Fourier 基 | 周期特征 |

### ReLU 网络 = 分段线性函数

这是最重要的联系：ReLU 网络 $f(\mathbf{x})$ 是**分段线性连续函数**（Continuous Piecewise Linear, CPWL）。区域的划分由所有神经元的"开关"状态决定。

在 $\mathbb{R}$ 上（曲线情形），ReLU 网络的每一段是线性函数，转折点是 $\{-b_i / w_i\}$。这本质上是**自适应节点的一阶 B-spline**，其中节点位置由优化决定而非预先固定。

---

## 曲线拟合的训练方法

### 损失函数

给定数据 $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^m$：

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^m \|f_{\boldsymbol{\theta}}(\mathbf{x}_i) - \mathbf{y}_i\|^2
$$

### 优化方法

- **SGD / Adam**：随机梯度下降及其自适应变体，$O(m \cdot p)$ 每次迭代（$p$ 为参数个数）
- **L-BFGS**：对中小规模（$m \sim 10^3$–$10^5$）更有效，但内存消耗大
- **Gauss-Newton / KFAC**：利用 Fisher 信息的二阶方法，对过参数化网络效果好

### 正则化

防止过拟合的常用策略：

1. **权重衰减（$L^2$ 正则化）**：

$$
\mathcal{L}_{\text{reg}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \lambda \|\mathbf{w}\|^2
$$

2. **早停（Early Stopping）**：在验证误差最小时停止训练。隐式等价于限制参数范数增长。

3. **Dropout**：训练时随机丢弃神经元，等价于集成大量子网络的预测。

4. **谱归一化 / 梯度惩罚**：约束网络 Lipschitz 常数以控制振荡。

---

## 神经网络 vs 经典方法的效率对比

| 维度 | 经典方法 | 神经网络 |
|------|----------|----------|
| 基函数 | 固定（如 Bernstein、B-spline） | 可学习（自适应） |
| 参数数 | $O(n)$（控制点） | $O(p)$（权重+偏置） |
| 拟合方式 | 线性最小二乘 / QP | 非线性非凸优化（SGD） |
| 凸性 | 凸（最小二乘） | 非凸 |
| 理论保证 | 完备（逼近阶、稳定性） | 仍在发展中（NTK、双下降） |
| 计算复杂度 | $O(n^3)$（稠密）–$O(n)$（稀疏） | $O(\text{epochs} \cdot m \cdot p)$ |
| 高维适应性 | 维度诅咒（除 RBF 外） | 突破维度诅咒（Barron） |
| 可解释性 | 强（几何直观） | 弱 |

---

## 物理信息神经网络（PINN）

对于曲线拟合 + 物理约束的场景，PINN（Raissi et al., 2019）提供统一框架：

$$
\mathcal{L}_{\text{PINN}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{数据拟合}} + \lambda_{\text{phys}} \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{物理残差}}
$$

例如，要求拟合曲线最小化弯曲能量：

$$
\mathcal{L}_{\text{phys}} = \frac{1}{m} \sum_{i=1}^m \|\ddot{f}_{\boldsymbol{\theta}}(x_i) - 0\|^2
$$

（即为自然样条的变分原理在神经网络框架下的实现）。PINN 的优点是可以将任意微分约束自然地融入拟合过程。

---

## 神经算子与 DeepONet

对于"学习一族曲线"的任务（如学习 PDE 解算子），可以使用：

- **DeepONet**（Lu et al., 2021）：学习函数空间之间的映射
- **FNO**（Fourier Neural Operator）：在 Fourier 频域参数化算子
- **NOMAD**：结合样条和注意力的算子学习

这些方法将曲线表示从"拟合单条曲线"提升到"学习曲线族的结构"。

---

## 神经隐式表示（Neural Implicit Representation）

另一种表示曲线的方式是通过隐式函数 $F: \mathbb{R}^d \to \mathbb{R}$ 的零水平集：

$$
\Gamma = \{ \mathbf{x} \in \mathbb{R}^d : F_{\boldsymbol{\theta}}(\mathbf{x}) = 0 \}
$$

$F_{\boldsymbol{\theta}}$ 由神经网络参数化（如 SIREN 使用周期激活函数）。优点：

- 自动处理拓扑变化
- 可以表示任意 genus 的曲面
- 分辨率无关（连续表示）
- 适合三维重建和形状补全

### 位置编码（Positional Encoding）

标准 MLP 难以拟合高频细节（"spectral bias" — 神经网络偏向学习低频函数）。Mildenhall et al. (2020) 提出的位置编码将输入映射到高频 Fourier 特征：

$$
\gamma(x) = [x, \sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \dots]
$$

这使得 MLP 能有效表达高频细节。从 RKHS 角度看，位置编码的神经网络等价于具有特定 NTK（神经正切核）的核方法，该核在频率域具有均匀的谱密度。

---

## NTK（神经正切核）视角

在无限宽极限下，随机初始化的神经网络等价于 GP（Gaussian Process），其训练动态由 NTK 控制（Jacot et al., 2018）：

$$
K_{\text{NTK}}(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{\boldsymbol{\theta}}\left[ \left\langle \frac{\partial f_{\boldsymbol{\theta}}(\mathbf{x})}{\partial \boldsymbol{\theta}}, \frac{\partial f_{\boldsymbol{\theta}}(\mathbf{x}')}{\partial \boldsymbol{\theta}} \right\rangle \right]
$$

这建立了神经网络与核方法/RKHS 的深层联系，也为理解"过参数化网络的隐式正则化"提供了数学工具。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 逼近能力极强（突破维度诅咒） | 非凸优化（不保证全局最优） |
| 基函数自适应学习 | 训练不稳定（需大量调参） |
| 高维数据处理自然 | 过参数化导致过拟合风险 |
| 硬件加速（GPU/TPU） | 可解释性差 |
| 灵活融入物理约束（PINN） | 理论保证仍不完整 |
| 连续/隐式表示 | 推理速度慢于封闭公式 |
