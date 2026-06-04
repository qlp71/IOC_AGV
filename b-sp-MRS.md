# MRS中心轨迹和各个Robot的轨迹关系

可视化一个MRS （Multi-Robot System）中心轨迹和各个Robot的轨迹关系，中心轨迹用B-spline曲线拟合

可调节的参数有：

1. num_points: B-spline曲线拟合的点数 (通过一个输入框进行赋值)

2. degree: B-spline曲线的阶数  (通过一个输入框进行赋值)

3. T: 时间范围，单位为秒   (通过一个输入框进行赋值)

4. t_inp: [0, t1, t2, ..., t_{n-2}, T] 非均匀待插值的时间点，起始为0，结束为T，长度为num_points （通过一个数轴进行赋值，起点是0，终点是T，通过拖动中间的点来调整）

5. control_points: B-spline曲线的控制点，长度为num_points，维度为3，分别代表 x_c, y_c, theta_c: MRS中心的轨迹点，（在笛卡尔坐标系下可视化并且可交互调整，这些插值是用t_inp生成b-spline的knots进行插值）

6. N: Robot的数量 (通过一个输入框进行赋值)

7. x_i^r, y_i^r: robots关于MRS中心的相对位置（i=1,2,...,N, 在笛卡尔坐标系下可视化并且可交互调整）

可视化的内容包括：

1. MRS中心的轨迹（B-spline曲线）和robot的轨迹, 其中robot的轨迹由MRS中心的轨迹和相对位置计算得到：
x_i(t) = x_c(t) + cos(theta_c(t)) * x_i^r - sin(theta_c(t)) * y_i^r
y_i(t) = y_c(t) + sin(theta_c(t)) * x_i^r + cos(theta_c(t)) * y_i^r

2. 通过一个数轴来调整t_inp中的时间点，观察MRS中心轨迹和robot轨迹的变化

3. 通过输入框调整num_points, degree, T, N等参数，观察MRS中心轨迹和robot轨迹的变化

4. 在坐标系中可视化插值点和控制点（t-x_c, t-y_c, t-theta_c）

5. 在坐标系中可视化robot的轨迹点（t-x_i, t-y_i，t-theta_i）
其中theta_i(t) = arctan2(dy_i/dt, dx_i/dt) 代表robot的朝向

dx_i/dt = dx_c/dt + (cos(theta_c(t) + pi/2) * x_i^r - sin(theta_c(t) + pi/2) * y_i^r) * dtheta_c/dt
dy_i/dt = dy_c/dt + (sin(theta_c(t) + pi/2) * x_i^r + cos(theta_c(t) + pi/2) * y_i^r) * dtheta_c/dt

布局如下：
1. 左上角坐标系：可视化MRS中心轨迹和robot轨迹，插值点和控制点（x-y平面, 不需要theta的可视化，这个最大）
2. 左下角是一系列输入框和数轴：num_points, degree, T, N, t_inp的调整
3. 右边一系列坐标系：分别可视化t-x_c, t-y_c, t-theta_c, t-x_i, t-y_i，t-theta_i（每个坐标系只可视化一个变量，方便观察）
4. 在每个坐标系中可视化对应的轨迹点（插值点和控制点）和robot的轨迹点（t-x_i, t-y_i，t-theta_i）

