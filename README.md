# IOC_AGV

基于**信息几何优化（IGO）** 与**逆最优控制（IOC）** 的自动引导车（AGV）控制参数优化框架。

---

## 项目概述

本项目围绕两个核心方法展开：

| 方法 | 角色 | 说明 |
|------|------|------|
| **MGIGO** (Mixture of Gaussian IGO) | 优化引擎 | 无需梯度的黑盒随机搜索算法，用高斯混合模型在参数空间中搜索最优解 |
| **IOC** (Inverse Optimal Control) | 代价函数设计 | 从控制律和 CLF 逆向推导"有意义的"代价函数，为 IGO 提供优化目标 |

两者结合形成完整工作流：**IOC 设计代价函数 → IGO 搜索最优参数 → 得到理论有保证且数值最优的控制律**。

此外，项目还包含一个**多机器人系统（MRS）B-spline 轨迹规划与可视化工具**。

---

## 目录结构

```
IOC_AGV/
├── igo/                          # IGO 求解器核心
│   ├── blockwise_mgigo_jax.py    #   主求解器：分块 MGIGO（350行）
│   ├── MPC_G_MS.py               #   多智能体博弈 IGO 求解器（233行）
│   ├── MPCsolverM22.py           #   MPC 求解器接口（174行）
│   ├── utils.py                  #   协方差计算、采样、统计工具（111行）
│   └── plot_utils.py             #   可视化工具（213行）
│
├── ioc_utils.py                  # IOC 核心：CLF、控制律、代价函数构造（471行）
├── ioc_viz.py                    # 差速驱动机器人交互仿真（707行）
├── main_IOC.py                   # IOC 仿真入口
│
├── curves/                       # B-spline 曲线工具
│   ├── bspline_utils.py          #   NumPy 版 B-spline 工具
│   └── bspline_utils_jax.py      #   JAX 版 B-spline 工具
├── mrs_bs_viz.py                 # MRS 轨迹规划可视化
├── main_MRS_bs.py                # MRS 可视化入口
│
├── igo_test.py                   # IGO 求解器测试（6个测试函数）
├── output_igo_test/              # 测试输出：等高线图 + 优化过程视频
├── figures/                      # 文档插图
├── curve_theory/                 # 曲线理论参考文档（9篇）
│
├── presentation/                 # Reveal.js 学术汇报演示文稿
│   ├── README.md                 #   演示文稿制作说明
│   └── igo_presentation.html     #   演示文稿（浏览器打开即用）
│
├── README_IGO.md                 # IGO 求解器详解
├── README_IOC.md                 # 逆最优控制详解
├── README_RNE.md                 # 多智能体博弈 IGO 详解
├── b-sp-MRS.md                   # MRS 轨迹规划说明
├── pyproject.toml                # 项目配置与依赖
└── README.md                     # 本文件
```

---

## 快速开始

### 环境要求

- Python ≥ 3.14
- 支持 CUDA 的 GPU（可选，JAX 可回退到 CPU）
- 依赖：`numpy`, `matplotlib`, `jax[cuda13]`, `tqdm`, `pyqt6`

### 安装

建议用 `uv` 管理 Python 环境：

```bash
cd IOC_AGV
uv sync                          # 或 pip install -e .
```

### 运行

```bash
# 1. IGO 求解器测试（6种目标函数）
uv run igo_test.py

# 2. 差速驱动机器人 IOC 交互仿真
uv run main_IOC.py

# 3. 多机器人系统 B-spline 轨迹规划
uv run main_MRS_bs.py
```

---

## 核心模块

### 1. MGIGO 求解器（`igo/`）

**一句话：用多个"探针云"（高斯分布）在参数空间中搜索，每次保留表现最好的探针，沿自然梯度方向移动，逐步逼近最优解。**

- **算法类型**：无梯度黑盒随机搜索
- **核心机制**：高斯混合模型 + 自然梯度 + 精英选择 + 周期性重置
- **关键特性**：分块（Blockwise）机制缓解维度灾难、全 JAX 实现支持 GPU 加速
- **适用场景**：非凸、多模态、不可导的目标函数优化

详细文档：[README_IGO.md](./README_IGO.md)

#### 测试案例

| 案例 | 类型 | 关键挑战 |
|------|------|----------|
| 余弦乘积 | 多模态 | 多个等深全局最优 |
| 二次型 | 凸函数 | 收敛速度与精度 |
| 各向异性二次型 | 狭长山谷 | 不同方向尺度差异大 |
| 半平面约束 | 不连续 | 跨越不连续边界 |
| 线性走廊约束 | 狭长可行域 | 窄带内搜索 |
| 环形约束 | 非线性约束 | 圆环可行域 |

测试输出位于 `output_igo_test/`，包含等高线图和优化过程动画。

#### 多智能体扩展（MPC_G_MS）

支持多个智能体各自控制一部分决策变量，在不知彼此策略的情况下通过反复试探收敛到随机纳什均衡（RNE）。

详细文档：[README_RNE.md](./README_RNE.md)

---

### 2. 逆最优控制（IOC）

**一句话：先设计一个好的控制律，再逆向推导出使该控制律最优的代价函数——避免求解困难的 HJB 方程。**

- **理论基础**：控制 Lyapunov 函数（CLF）+ Freeman & Kokotović 逆最优性定理
- **应用案例**：双轮差速驱动机器人的镇定控制
- **核心产出**：4 种 CLF 设计方案，对应 4 种不同的 $\delta$-$\gamma$ 耦合策略

详细文档：[README_IOC.md](./README_IOC.md)

#### 交互仿真（`python main_IOC.py`）

| 控件 | 功能 |
|------|------|
| 拖动绿点 | 改变起始位置 |
| 拖动红点 | 改变目标位置 |
| 滑块 | 调节 $k_1, k_2, k_3$ 和初始朝向 $\theta_0$ |
| 单选按钮 | 切换控制律变体 1–4 |
| 复选框 | 切换 tanh 速度饱和 |

---

### 3. MRS B-spline 轨迹规划（`curves/`, `mrs_bs_viz.py`）

多机器人系统（MRS）中心轨迹的 B-spline 曲线拟合与交互可视化。

- 支持非均匀节点向量的 B-spline 插值
- 交互式调节控制点、阶数、时间点、机器人数量
- 实时显示中心轨迹及各机器人轨迹

详细文档：[b-sp-MRS.md](./b-sp-MRS.md)

---

## IGO + IOC 工作流

```
IOC 方法                           IGO 求解器
─────────                          ──────────
设计镇定控制律 u = k(x)            接收代价函数 J 作为目标
       ↓                                  ↓
构造 CLF V(x)，验证 V̇ < 0         参数化控制律 u=k(x;θ)
       ↓                                  ↓
逆推代价函数 J = ∫[l(x)+uᵀRu]dt   定义 f(θ) = J(x,k(x;θ))
       ↓                                  ↓
       └──────────→  cost 函数  ←──────────┘
                            ↓
                   MGIGO 搜索 θ*
                            ↓
              理论保证 + 数值最优的控制律
```

---

## 技术栈

| 技术 | 用途 |
|------|------|
| **JAX** | IGO 核心计算（jit 编译、vmap 并行、lax.scan 循环） |
| **NumPy/SciPy** | 数值计算 |
| **Matplotlib** | 可视化与交互动画 |
| **PyQt6** | IOC 仿真 GUI |
| **Reveal.js** | 学术汇报演示文稿 |
| **MathJax** | 数学公式渲染 |
| **uv** | Python 包管理 |

---

## 参考文献

- Ollivier, Y., Arnold, L., Auger, A., & Hansen, N. (2017). "Information-Geometric Optimization Algorithms: A Unifying Picture via Invariance Principles." *JMLR*, 18(18), 1–65.
- Freeman, R. A. & Kokotović, P. V. (1996). "Inverse optimality in robust stabilization." *SIAM Journal on Control and Optimization*, 34(4), 1365–1391.
- Sontag, E. D. (1989). "A 'universal' construction of Artstein's theorem on nonlinear stabilization." *Systems & Control Letters*, 13(2), 117–123.
- Kalman, R. E. (1964). "When is a linear control system optimal?" *Journal of Basic Engineering*, 86(1), 51–60.
- Sepulchre, R., Janković, M., & Kokotović, P. V. (1997). *Constructive Nonlinear Control*. Springer.
- Freeman, R. A. & Kokotović, P. V. (2008). *Robust Nonlinear Control Design: State-Space and Lyapunov Techniques*. Birkhäuser.

该项目的 IGO 求解器基于项目 [MGIGO](https://github.com/Konsteidinoeevich/MGIGO.git) 。

---

## 许可证

本项目仅用于学术研究与教育目的。

p.s. 大部分内容由 DeepSeek 协助生成，整体内容没有问题，细节上需要仔细甄别。
