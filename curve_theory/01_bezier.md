# Bézier 曲线

## 定义

$n$ 次 Bézier 曲线定义为 Bernstein 基的线性组合：

$$
\gamma(t) = \sum_{i=0}^n \mathbf{P}_i B_i^n(t), \quad t \in [0,1]
$$

其中 $\mathbf{P}_i \in \mathbb{R}^d$ 为控制点，Bernstein 基函数为：

$$
B_i^n(t) = \binom{n}{i} t^i (1-t)^{n-i}, \quad i = 0,1,\dots,n
$$

---

## Bernstein 基的性质

### 1. 非负性与单位分解

$$
B_i^n(t) \geq 0, \quad \sum_{i=0}^n B_i^n(t) = 1 \quad \forall t \in [0,1]
$$

**证明**（单位分解）：由二项式定理，

$$
1 = (t + (1-t))^n = \sum_{i=0}^n \binom{n}{i} t^i (1-t)^{n-i} = \sum_{i=0}^n B_i^n(t)
$$

这直接导出 **凸包性质**：曲线完全落在控制点凸包内。

### 2. 端点插值

$$
B_i^n(0) = \begin{cases} 1 & i=0 \\ 0 & i>0 \end{cases}, \quad
B_i^n(1) = \begin{cases} 1 & i=n \\ 0 & i<n \end{cases}
$$

因此 $\gamma(0) = \mathbf{P}_0$，$\gamma(1) = \mathbf{P}_n$。

### 3. 对称性

$$
B_i^n(t) = B_{n-i}^n(1-t)
$$

### 4. 递推关系

$$
B_i^n(t) = (1-t) B_i^{n-1}(t) + t B_{i-1}^{n-1}(t)
$$

其中约定 $B_{-1}^{n-1} = B_n^{n-1} = 0$。

### 5. 导数

$$
\frac{d}{dt} B_i^n(t) = n \left(B_{i-1}^{n-1}(t) - B_i^{n-1}(t)\right)
$$

由此得 Bézier 曲线的导数（仍为 Bézier 形式）：

$$
\gamma'(t) = n \sum_{i=0}^{n-1} (\mathbf{P}_{i+1} - \mathbf{P}_i) B_i^{n-1}(t)
$$

端点导数：

$$
\gamma'(0) = n(\mathbf{P}_1 - \mathbf{P}_0), \quad \gamma'(1) = n(\mathbf{P}_n - \mathbf{P}_{n-1})
$$

高阶导数同理可递推。

---

## de Casteljau 算法

这是 Bézier 曲线的核心求值算法，数值稳定且几何直观。

### 算法

给定 $t \in [0,1]$，令 $\mathbf{P}_i^{(0)} = \mathbf{P}_i$，然后对 $r = 1,\dots,n$：

$$
\mathbf{P}_i^{(r)} = (1-t) \mathbf{P}_i^{(r-1)} + t \mathbf{P}_{i+1}^{(r-1)}, \quad i = 0,\dots,n-r
$$

则 $\mathbf{P}_0^{(n)} = \gamma(t)$。

### 计算复杂度

$O(n^2)$ 次线性插值。对于常用低次 Bézier（$n=3$），仅需 6 次插值。

### de Casteljau 分割

取中间结果 $\mathbf{P}_0^{(0)}, \mathbf{P}_0^{(1)}, \dots, \mathbf{P}_0^{(n)}$ 为左子曲线控制点，$\mathbf{P}_0^{(n)}, \mathbf{P}_1^{(n-1)}, \dots, \mathbf{P}_n^{(0)}$ 为右子曲线控制点。这给出了曲线在 $t$ 处的精确分割。

---

## 升阶

$n$ 次 Bézier 曲线可用 $n+1$ 次 Bézier 曲线精确表示。新控制点为：

$$
\mathbf{P}_i^{\text{new}} = \frac{i}{n+1} \mathbf{P}_{i-1} + \left(1 - \frac{i}{n+1}\right) \mathbf{P}_i, \quad i = 0,\dots,n+1
$$

升阶在拟合中用于统一不同次数曲线的表示。

---

## 拟合方法

### 1. 最小二乘拟合

给定数据点 $\{\mathbf{q}_j\}_{j=1}^m$ 及对应参数 $\{t_j\}_{j=1}^m \subset [0,1]$（通常用弦长参数化），最小化：

$$
\min_{\mathbf{P}_i} \sum_{j=1}^m \left\| \sum_{i=0}^n \mathbf{P}_i B_i^n(t_j) - \mathbf{q}_j \right\|^2
$$

记矩阵 $\mathbf{B} \in \mathbb{R}^{m \times (n+1)}$，其中 $B_{ji} = B_i^n(t_j)$，则问题化为：

$$
\min_{\mathbf{P}} \|\mathbf{B} \mathbf{P} - \mathbf{Q}\|_F^2
$$

正规方程：

$$
\mathbf{B}^\top \mathbf{B} \mathbf{P} = \mathbf{B}^\top \mathbf{Q}
$$

这是一个 $(n+1) \times (n+1)$ 的稠密系统（Bernstein 基无局部支撑，$\mathbf{B}^\top \mathbf{B}$ 是满矩阵）。

### 2. 参数化策略

参数 $\{t_j\}$ 的选择对拟合质量影响显著：

- **均匀参数化**：$t_j = \frac{j-1}{m-1}$，简单但可能导致振荡
- **弦长参数化**：

$$
t_1 = 0, \quad t_j = t_{j-1} + \frac{\|\mathbf{q}_j - \mathbf{q}_{j-1}\|}{\sum_{k=2}^m \|\mathbf{q}_k - \mathbf{q}_{k-1}\|}
$$

  更贴合数据分布，常用

- **向心参数化**：分母用 $\sqrt{\|\mathbf{q}_j - \mathbf{q}_{j-1}\|}$，对急转弯更鲁棒

### 3. 带光滑性约束的拟合

为避免高阶 Bézier 振荡，添加曲率惩罚：

$$
\min_{\mathbf{P}_i} \sum_{j=1}^m \left\|\gamma(t_j) - \mathbf{q}_j\right\|^2 + \lambda \int_0^1 \|\gamma''(t)\|^2 dt
$$

曲率能量的离散形式可利用 Bézier 导数公式精确计算。

---

## 几何性质

### Variation Diminishing 性质

Bézier 曲线与任意直线的交点数不会超过其控制多边形与该直线的交点数。这意味着：

> 曲线不会比控制多边形更"振荡"。

这是 Bernstein 基的变号缩减性质（Variation Diminishing Property）的直接推论。

### 仿射不变性

对任意仿射变换 $T(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$：

$$
T(\gamma(t)) = \sum_{i=0}^n T(\mathbf{P}_i) B_i^n(t)
$$

即变换控制点等价于变换曲线。

### 升阶的极限

无穷次升阶后，控制多边形收敛于曲线自身（这一点与 Bernstein 多项式逼近连续函数的 Weierstrass 定理本质相同）。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 几何直观（控制多边形逼近曲线） | 全局控制（改一点全变） |
| 凸包性质（安全约束） | 高次时振荡（Runge 现象） |
| de Casteljau 数值稳定 | 基无局部支撑 |
| 端点插值 | 不能精确表示圆锥曲线 |
| 仿射不变性 | $\mathbf{B}^\top \mathbf{B}$ 稠密 |

---

## 常见实现：三次 Bézier

工程中最常用的是三次 Bézier（$n=3$），由 4 个控制点定义：

$$
\gamma(t) = (1-t)^3 \mathbf{P}_0 + 3(1-t)^2 t \mathbf{P}_1 + 3(1-t) t^2 \mathbf{P}_2 + t^3 \mathbf{P}_3
$$

矩阵形式：

$$
\gamma(t) = \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & 0 \\
-3 & 3 & 0 & 0 \\
3 & -6 & 3 & 0 \\
-1 & 3 & -3 & 1
\end{bmatrix}
\begin{bmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \mathbf{P}_2 \\ \mathbf{P}_3 \end{bmatrix}
$$

这对应三次 Hermite 表示，在 `05_hermite.md` 中有更详细的讨论。
