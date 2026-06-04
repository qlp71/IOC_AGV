# NURBS 曲线

## 定义

NURBS（Non-Uniform Rational B-Spline）是 B-spline 的有理推广：

$$
\gamma(t) = \frac{\sum_{i=0}^{n-1} w_i \mathbf{P}_i N_{i,p}(t)}{\sum_{i=0}^{n-1} w_i N_{i,p}(t)}, \quad t \in [t_p, t_n]
$$

其中：
- $\mathbf{P}_i \in \mathbb{R}^d$：控制点
- $w_i > 0$：权重（weights）
- $N_{i,p}(t)$：$p$ 次 B-spline 基函数（与 `02_bspline.md` 中定义相同）

通常在实际计算中，使用齐次坐标表示：

$$
\gamma^w(t) = \sum_{i=0}^{n-1} \begin{bmatrix} w_i \mathbf{P}_i \\ w_i \end{bmatrix} N_{i,p}(t) \in \mathbb{R}^{d+1}
$$

然后将前 $d$ 维除以最后一维（透视除法）得到 $\gamma(t)$。这样 NURBS 曲线就是 $\mathbb{R}^{d+1}$ 中多项式 B-spline 在 $\mathbb{R}^d$ 上的投影。

---

## 权重的几何意义

权重 $w_i$ 控制曲线对控制点 $\mathbf{P}_i$ 的"引力"强度：

- $w_i \to \infty$：曲线趋近于 $\mathbf{P}_i$
- $w_i \to 0^+$：曲线远离 $\mathbf{P}_i$
- $w_i = 1$（所有 $i$）：退化为标准 B-spline

### 权重与圆锥曲线参数

对于二次 NURBS（$p=2$，3 个控制点 $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2$），令 $w_0 = w_2 = 1$，则 $w_1$ 决定了曲线类型：

| $w_1$ | 曲线类型 |
|-------|----------|
| $w_1 < 1$ | 椭圆弧 |
| $w_1 = 1$ | 抛物弧（即多项式 Bézier） |
| $w_1 > 1$ | 双曲弧 |

一般地，圆锥曲线等价于：

$$
w_1^2 \gtrless w_0 w_2 \quad \Longleftrightarrow \quad \text{双曲线 / 抛物线 / 椭圆}
$$

这是 NURBS 的**核心优势**：用有限个控制点**精确**表示圆、椭圆、双曲线，而多项式样条只能近似。

### 精确表示圆

单位圆的 90° 圆弧可由三次 NURBS 精确表示。例如，第一象限的四分之一圆：

控制点与权重（$p=2$，节点 $[0,0,0,1,1,1]$）：

$$
\begin{aligned}
\mathbf{P}_0 &= (1, 0), \quad w_0 = 1 \\
\mathbf{P}_1 &= (1, 1), \quad w_1 = \frac{\sqrt{2}}{2} \\
\mathbf{P}_2 &= (0, 1), \quad w_2 = 1
\end{aligned}
$$

对于任意角度的圆（$p=2$，节点 $[0,0,0,1,1,1]$），权重的一般公式为：

$$
w_1 = \cos\left(\frac{\theta}{2}\right)
$$

其中 $\theta$ 为弧所对的圆心角。

---

## NURBS 基函数的有理形式

定义有理基函数：

$$
R_{i,p}(t) = \frac{w_i N_{i,p}(t)}{\sum_{j=0}^{n-1} w_j N_{j,p}(t)}
$$

则 NURBS 曲线可写为标准线性形式：

$$
\gamma(t) = \sum_{i=0}^{n-1} \mathbf{P}_i R_{i,p}(t)
$$

$R_{i,p}$ 继承了 $N_{i,p}$ 的几乎所有性质：
- 非负性：$R_{i,p}(t) \geq 0$
- 单位分解：$\sum_i R_{i,p}(t) = 1$
- 局部支撑：$\operatorname{supp}(R_{i,p}) = \operatorname{supp}(N_{i,p})$
- 可微性与连续性同 $N_{i,p}$

---

## 导数

利用商规则和 B-spline 导数公式，NURBS 曲线的一阶导数：

$$
\gamma'(t) = \frac{\mathbf{A}'(t) - w'(t) \gamma(t)}{w(t)}
$$

其中：

$$
\mathbf{A}(t) = \sum_i w_i \mathbf{P}_i N_{i,p}(t), \quad w(t) = \sum_i w_i N_{i,p}(t)
$$

$\mathbf{A}'(t)$ 和 $w'(t)$ 可直接使用 B-spline 导数公式计算。高阶导数可递推，但表达式较复杂。在代码实现中通常使用齐次坐标求导再投影。

---

## 拟合方法

### 1. 固定权重的最小二乘

若权重 $w_i$ 已给定（如通过经验或从已知形状继承），退化为标准有理最小二乘：

$$
\min_{\mathbf{P}} \|\mathbf{R} \mathbf{P} - \mathbf{Q}\|_F^2
$$

其中 $R_{ji} = R_{i,p}(u_j)$。由于 $R_{i,p}$ 同样具有局部支撑，$\mathbf{R}$ 是稀疏的，可用稀疏求解器。

### 2. 联合优化控制点和权重

同时优化 $\mathbf{P}_i$ 和 $w_i$：

$$
\min_{\mathbf{P}, \mathbf{w}} \sum_{j=1}^m \left\|\frac{\sum_i w_i \mathbf{P}_i N_{i,p}(u_j)}{\sum_i w_i N_{i,p}(u_j)} - \mathbf{q}_j\right\|^2, \quad w_i > 0
$$

这是一个**非线性最小二乘**问题（权重出现在分子和分母中），可使用：
- **Gauss-Newton 法**：线性化有理函数
- **Levenberg-Marquardt**：带阻尼的 Gauss-Newton，更鲁棒
- **齐次坐标法**：在 $\mathbb{R}^{d+1}$ 中做线性拟合再投影

### 3. 齐次坐标线性化方法

将数据点 $\mathbf{q}_j$ 提升到齐次坐标 $\mathbf{q}_j^w = [\mathbf{q}_j, 1]^\top$（或带权重提升），然后在 $\mathbb{R}^{d+1}$ 中拟合多项式 B-spline 到 $\{h_j \mathbf{q}_j^w\}$，再经透视除法回到 $\mathbb{R}^d$。$h_j$ 的选择影响拟合质量。

### 4. 带约束的拟合

工程中常添加权重约束以防止病态：
- $w_i > 0$（严格正）
- $w_{\min} \leq w_i \leq w_{\max}$（有界权重）
- 部分权重固定（如端点 $w_0 = w_{n-1} = 1$）

---

## 工业标准：IGES 与 STEP

NURBS 是 CAD 数据交换标准的数学基础：

- **IGES**（Initial Graphics Exchange Specification）：实体类型 126 为 NURBS 曲线
- **STEP**（ISO 10303）：AP203/AP214 中所有曲线和曲面都是 NURBS

几乎所有主流 CAD 系统（CATIA、SolidWorks、Siemens NX、Rhino、Alias）都以 NURBS 为内部表示。

---

## 节点插入与细化

NURBS 继承了 B-spline 的节点插入算法（Boehm 算法），但需要在齐次坐标下进行：

1. 将 $\mathbf{P}_i$ 和 $w_i$ 提升为齐次控制点 $[w_i \mathbf{P}_i, w_i]^\top$
2. 在 $\mathbb{R}^{d+1}$ 中对多项式 B-spline 做标准节点插入
3. 通过透视除法回到 $\mathbb{R}^d$

这保证了插入节点不改变曲线几何形状（几何不变性）。

---

## 优缺点总结

| 优点 | 缺点 |
|------|------|
| 精确表示圆锥曲线（圆、椭圆、双曲线） | 权重优化是非线性的 |
| 继承 B-spline 全部优良性质 | 微积分（尤其是高阶导数）复杂 |
| 投影不变性（透视投影仍为 NURBS） | 理解和实现门槛高 |
| CAD 工业标准 | 反求权重可能病态 |

---

## 与 B-spline / Bézier 的关系

```
Bézier ⊂ B-spline ⊂ NURBS

Bézier: w_i = 1, 单个跨度的 clamped 节点
B-spline: w_i = 1（所有权重为 1）
NURBS: 最一般形式
```

---

## 数值注意事项

- 避免 $w_i$ 接近 0（导致分母接近 0，数值爆炸）
- 齐次坐标求导后投影比直接对有理函数求导更稳定
- 节点插入在齐次坐标下进行以保持正确性
- 大权重比（如 $\max w_i / \min w_i > 10^4$）会导致条件数恶化
