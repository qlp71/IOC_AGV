# B-spline 曲线

## 定义

$p$ 次（阶数为 $p+1$）B-spline 曲线定义为：

$$
\gamma(t) = \sum_{i=0}^{n-1} \mathbf{P}_i N_{i,p}(t), \quad t \in [t_{p}, t_{n}]
$$

其中 $\mathbf{P}_i \in \mathbb{R}^d$ 为控制点，$N_{i,p}(t)$ 为 $p$ 次 B-spline 基函数。

---

## 节点向量

B-spline 基由**节点向量**（knot vector）唯一确定：

$$
\mathbf{T} = [t_0, t_1, \dots, t_{n+p}]
$$

为非递减序列：$t_i \leq t_{i+1}$。

### 节点向量类型

#### 1. 均匀（Uniform）

$$
t_i = i, \quad i = 0,\dots,n+p
$$

#### 2. 开放均匀（Open Uniform / Clamped）

两端节点重复 $p+1$ 次，保证曲线端点插值首尾控制点：

$$
\mathbf{T} = [\underbrace{0,\dots,0}_{p+1}, t_{p+1},\dots,t_{n-1}, \underbrace{1,\dots,1}_{p+1}]
$$

这是 CAD 中最常用的类型。

#### 3. 非均匀（Non-Uniform）

节点间距不固定，提供更大的灵活性。

### 节点重数与连续性

若内部节点 $t_k$ 的重数为 $m$，则 B-spline 在该处的连续性为 $C^{p-m}$。特别地：

- $m=1$（单节点）：$C^{p-1}$ 连续
- $m=p$（$p$ 重节点）：$C^0$ 连续（位置连续，导数不连续）
- $m=p+1$：曲线在此处插值控制点（类似 Bézier 端点行为）

---

## Cox-de Boor 递推定义

B-spline 基函数通过如下递推定义：

### 零次基（$p=0$，分段常数）

$$
N_{i,0}(t) = \begin{cases}
1, & t \in [t_i, t_{i+1}) \\
0, & \text{否则}
\end{cases}
$$

### 高次基递推（$p \geq 1$）

$$
N_{i,p}(t) = \frac{t - t_i}{t_{i+p} - t_i} N_{i,p-1}(t) + \frac{t_{i+p+1} - t}{t_{i+p+1} - t_{i+1}} N_{i+1,p-1}(t)
$$

约定 $0/0 = 0$（当分母为零时）。

---

## B-spline 基的核心性质

### 1. 局部支撑（Local Support）

$$
\operatorname{supp}(N_{i,p}) = [t_i, t_{i+p+1}]
$$

这是 B-spline 最重要的性质：每个基函数只在有限区间内非零。

**推论**：修改控制点 $\mathbf{P}_i$ 仅影响曲线在 $[t_i, t_{i+p+1}]$ 上的形状。这是 CAD/CAM 中局部编辑能力的数学基础。

### 2. 非负性与单位分解

$$
N_{i,p}(t) \geq 0, \quad \sum_{i=0}^{n-1} N_{i,p}(t) = 1 \quad \forall t \in [t_p, t_n]
$$

后者保证了**仿射不变性**和**强凸包性质**——每一段曲线落在对应 $p+1$ 个控制点的凸包内（比 Bézier 的全局凸包更强）。

### 3. 导数

$$
N_{i,p}'(t) = \frac{p}{t_{i+p} - t_i} N_{i,p-1}(t) - \frac{p}{t_{i+p+1} - t_{i+1}} N_{i+1,p-1}(t)
$$

B-spline 曲线的导数仍然是 B-spline（次数降 1），控制点为：

$$
\mathbf{Q}_i = \frac{p}{t_{i+p+1} - t_{i+1}} (\mathbf{P}_{i+1} - \mathbf{P}_i)
$$

### 4. 逼近阶

若被逼近函数 $f \in C^{p+1}$，节点间距为 $h$，则：

$$
\|f - \sum_i f(t_i^*) N_{i,p}\|_\infty = O(h^{p+1})
$$

其中 $t_i^*$ 为适当的节点内点（如 Greville 横标）。

---

## 拟合方法

### 1. 最小二乘拟合

给定数据点 $\{\mathbf{q}_j\}_{j=1}^m$ 及参数 $\{u_j\}$，固定节点向量和次数 $p$，求解：

$$
\min_{\mathbf{P}} \|\mathbf{N} \mathbf{P} - \mathbf{Q}\|_F^2
$$

其中 $\mathbf{N} \in \mathbb{R}^{m \times n}$，$N_{ji} = N_{i,p}(u_j)$。

**关键差异**：由于局部支撑，$\mathbf{N}$ 是**稀疏带状矩阵**，每行至多 $p+1$ 个非零元。因此 $\mathbf{N}^\top \mathbf{N}$ 也是带状的，可用稀疏 Cholesky 分解高效求解。对于大规模问题，这是巨大的计算优势（vs Bézier 的稠密矩阵）。

正规方程：

$$
\mathbf{N}^\top \mathbf{N} \mathbf{P} = \mathbf{N}^\top \mathbf{Q}
$$

实际更推荐直接求解稀疏最小二乘问题 $\min \|\mathbf{N} \mathbf{P} - \mathbf{Q}\|$，可通过 QR 分解（对带状矩阵特别有效）避免正规方程的平方条件数。

### 2. 带光滑惩罚的拟合（P-spline）

引入差分惩罚以抑制过拟合（Eilers & Marx, 1996）：

$$
\min_{\mathbf{P}} \underbrace{\|\mathbf{N} \mathbf{P} - \mathbf{Q}\|^2}_{\text{数据保真}} + \lambda \underbrace{\|\mathbf{D}^{(k)} \mathbf{P}\|^2}_{\text{粗糙度惩罚}}
$$

其中 $\mathbf{D}^{(k)}$ 为 $k$ 阶差分矩阵。例如 $k=2$：

$$
\mathbf{D}^{(2)} =
\begin{bmatrix}
1 & -2 & 1 & 0 & \cdots \\
0 & 1 & -2 & 1 & \cdots \\
\vdots & & \ddots & \ddots & \ddots
\end{bmatrix}
$$

这使得 $\|\mathbf{D}^{(2)} \mathbf{P}\|^2 \approx \int \|\gamma''(t)\|^2 dt$。

解的结构（广义岭回归形式）：

$$
\hat{\mathbf{P}} = (\mathbf{N}^\top \mathbf{N} + \lambda \mathbf{D}^{(k)\top} \mathbf{D}^{(k)})^{-1} \mathbf{N}^\top \mathbf{Q}
$$

### 3. 节点选择

节点位置和数量是影响拟合质量的关键超参数：

- **等距节点**：简单但不一定最优
- **分位数节点**：按数据密度分布节点
- **自适应节点**：在曲率大的地方加密节点（需要非线性优化）

通常用交叉验证选择 $\lambda$（惩罚参数）和节点数。

### 4. 插值（Interpolation）

当 $m = n$ 且 $\mathbf{N}$ 可逆时，可直接求解 $\mathbf{N}\mathbf{P} = \mathbf{Q}$。根据 Schoenberg-Whitney 定理，$\mathbf{N}$ 可逆的充要条件是：

$$
t_i < u_i < t_{i+p+1}, \quad \forall i
$$

即每个节点区间内有恰好一个数据参数。

---

## B-spline 与 Bézier 的关系

Bézier 是 B-spline 的特例：取节点向量为

$$
\mathbf{T} = [\underbrace{0,\dots,0}_{p+1}, \underbrace{1,\dots,1}_{p+1}]
$$

（无内部节点），B-spline 基退化为 Bernstein 基，B-spline 曲线即为 Bézier 曲线。

**Boehm 节点插入算法**：在 B-spline 中插入新节点（不改变曲线几何形状），反复插入直至所有节点重数为 $p+1$，便得到 Bézier 表示。这实现了 B-spline → 分段 Bézier 的精确转换。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 局部控制（改一点只影响局部） | 理论理解门槛较高 |
| 稀疏结构（计算效率高） | 节点选择是超参数 |
| $C^{p-1}$ 连续性可调节 | 不能精确表示圆锥曲线 |
| 数值稳定（低次多项式分段） | 端点插值需 clamp 节点 |
| 任意次数的统一框架 | 实现比 Bézier 复杂 |
| 强凸包性质 | — |

---

## 机器人轨迹规划中的应用

在轨迹优化中，常用 **三次 B-spline**（$p=3$, $C^2$ 连续）或 **五次 B-spline**（$p=5$, $C^4$ 连续，jerk 连续）：

- 控制点 $\mathbf{P}_i$ 为优化变量
- 速度 $\dot{\gamma}(t)$ 和加速度 $\ddot{\gamma}(t)$ 通过导数公式解析给出
- 碰撞约束可利用凸包性质：若某个凸包不与障碍物相交，则对应段安全
- 稀疏 Hessian 结构使得大规模 MPC 问题可实时求解
