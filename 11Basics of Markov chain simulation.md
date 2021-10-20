# 吴毓志 2019302130058
# 11、马尔可夫链的基础知识
## 11.1 Gibbs 采样器
&emsp;&emsp;能够应用到许多多维问题中的Gibbs采样器是一种特殊的马尔可夫链算法，也称为交替条件采样，它是用θ的子向量定义的。假设参数向量θ被划分为d个分量或子向量，$θ=(θ_1，……，θ_d)$。Gibbs采样器的每次迭代都通过θ的子向量进行循环，绘制每个子集的条件是所有其他子集的值。因此，在迭代t中有d个步骤。在每次迭代t时，选择θ的d个子向量的顺序，然后依次从给定的所有θ的分量的条件分布中采样每个$θ^t_j$:
$
p(θ_j|θ_{-j}^{t-1},y)
$

其中$θ_{-j}^{t-1}$表示θ除$θ_j$外的所有分量，他们的当前值：

$
θ_{t−1}^ {−j} = (θ_{t1}, . . . , θ^t_{j−1}, θ^{t−1} _{j+1}, . . . , θ^{t−1}_d )
$

因此，每个子向量$θ_j$都以θ的其他分量的最新值为条件进行更新，这些值是已经更新的分量的迭代t值和其他组件的迭代t−1值。

&emsp;&emsp;对于许多涉及标准统计模型的问题，可以直接从参数的大部分或所有条件后验分布中采样。我们通常使用一组条件概率分布来构建模型，例如层次模型。这种模型中的条件分布通常是提供了简单模拟的共轭分布。

&emsp;&emsp;我们接下来用一个简单的例子来说明Gibbs采样器的工作原理。
### 11.1.1 样例：双变量正态分布
&emsp;&emsp;考虑一个具有未知均值$θ=(θ_1，θ_2)$和已知协方差矩阵$\begin{gathered}\begin{bmatrix}1&\rho\\\rho&1\end{bmatrix}\end{gathered}$的双变量正态分布群体的单一观察结果$(y_1，y_2)$，在θ上的均匀先验分布，后验分布为：
$
\begin{gathered}\begin{pmatrix}\theta_1\\\theta_2\end{pmatrix}\end{gathered}|y\sim N\begin{gathered}\begin{pmatrix}\begin{pmatrix}y_1\\y_2\end{pmatrix},\begin{gathered}\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}\end{gathered}\end{pmatrix}\end{gathered}
$

为了说明这个问题，我们在这里演示了Gibss采样器。我们需要的条件后验分布：
$
\theta_1|\theta_2,y\sim N(y_1+\rho(\theta_2-y_2),1-\rho^2)
$
$
\theta_2|\theta_1,y\sim N(y_2+\rho(\theta_1-y_1),1-\rho^2)
$
Gibbs采样器通过从以上这两个正态分布中交替采样来进行。一般来说，迭代开始的一种自然方法是从正态近似到后验分布的随机抽取。这样的随机抽样消除了在这个简单的例子中进行迭代模拟的需要。

代码：
```
def partialSampler(x,dim):
    xes = []
    for t in range(10): 
        xes.append(domain_random())
    tilde_ps = []
    for t in range(10): 
        tmpx = x[:]
        tmpx[dim] = xes[t]
        tilde_ps.append(get_tilde_p(tmpx))
    norm_tilde_ps = np.asarray(tilde_ps)/sum(tilde_ps)
    u = np.random.random()
    sums = 0.0
    for t in range(10):
        sums += norm_tilde_ps[t]
        if sums>=u:
            return xes[t]
def gibbs(x):
    rst = np.asarray(x)[:]
    path = [(x[0],x[1])]
    for dim in range(2): 
        new_value = partialSampler(rst,dim)
        rst[dim] = new_value
        path.append([rst[0],rst[1]])
    return rst,path

def testGibbs(counts = 100,drawPath = False):
    plotContour()

    x = (domain_random(),domain_random())
    xs = [x]
    paths = [x]
    for i in range(counts):
        xs.append([x[0],x[1]])
        x,path = gibbs(x)
        paths.extend(path) 
    if drawPath:
        plt.plot(map(lambda x:x[0],paths),map(lambda x:x[1],paths),'k-',linewidth=0.5)
    plt.scatter(map(lambda x:x[0],xs),map(lambda x:x[1],xs),c = 'g',marker='.')
    plt.show()
```
## 11.2 Metropolis算法、Metropolis-Hastings算法
&emsp;&emsp;Metropolis算法是一类马尔可夫链模拟方法的通称，这些方法有利于从贝叶斯后验分布中抽样。

&emsp;&emsp;上一节中介绍的Gibss采样器，它可以看作是Metropolis-Hastings算法的一种特殊情况。本节提出了基本Metropolis算法的及其推广到Metropolis-Hastings算法。
### 11.2.1 Metropolis算法
&emsp;&emsp;Metropolis算法：一种自适应的具有接受/拒绝规则的随机游走，以收敛到指定的目标分布。该算法的执行方法如下。

1、从起始分布$p_0（θ）$中抽取一个起点$θ^0$，其中$p(θ^0|y)>0$。

2、对于$t=1,2，·······：$

(a)从时间$t、J_t（θ^*|θ_{t−1}）$的跳跃分布（或proposal分布）中取样$\theta^*$。对于Metropolis算法，它的满足所有$θ_a、θ_b$和t的条件$J_t(θ_a|θ_b)=J_t(θ_b|θ_a)$的跳跃分布必须是对称的。

(b)计算密度之比:

$
r=\frac {p(\theta^*|y)}{p(\theta^{t-1}|y)}
$
(c)令$$ r=\begin{cases}
\theta^* & P_{min}(r,1) \\
\theta^{t-1} & 其它
\end{cases}$$

给定当前值$θ_{t−1}$，马尔可夫链的过渡分布$T^t(θ_t|θ^{t−1})$是$θ^t=θ^{t−1}$处的点的混合和根据接受率进行调整的加权的跳跃分布$J_t(θ^t|θ^{t−1})$。

该算法需要能够计算所有$(θ，θ^*)$的比率r（上述步骤c），并从所有θ和t的跳跃分布$J_t（θ^∗|θ）$中抽取θ。此外，上述步骤(c)需要生成一个均匀的随机数。当$θ^t=θ^{t−1}$—即如果跳跃不被接受时，这仍然算作算法中的迭代。
### 11.2.2 样例：具有正态跳跃核的双变量单位正态密度
&emsp;&emsp;我们用双变量单位正态分布的简单例子来说明Metropolis算法。目标密度为二元单位正态，$p(θ|y)=N(θ|0，I)$，其中$I$是2×2单位矩阵。跳跃分布也是双变量正态分布，以当前迭代为中心，并缩放到$\frac{1}{5}$的大小：
$
J_t(θ^∗|θ^{t−1}
) = N(θ^∗|θ^{t−1}, 0.2^2I).
$
在每一步中，都很容易计算出密度比$r=\frac{N(θ^∗|0，I)}{N(θ^{t−1}|0，I)}$。从正态分布的形式可以清楚地看出，跳跃规则是对称的。我们特别设计了很小的相对于目标分布的跳跃算法，这样该算法的运行效率就会低下，其随机游走就很明显了。
* 为什么Metropolis算法有效？

证明迭代序列$θ_1，θ_2，······$收敛于目标分布有两个步骤：

&emsp;&emsp;首先，证明模拟序列是一个具有唯一平稳分布的马尔可夫链。如果马尔可夫链是不可约的、非周期的而不是瞬态的，则证明的第一步成立。除了一些无价值的例外外，后两个条件适用于任何适当分布上的随机游动，只要随机游动有最终从任何其他状态到达任何状态的正概率，不可约性就成立。也就是说，跳跃分布$J_t$最终必须能够以正概率跳转到所有状态。

&emsp;&emsp;其次，证明平稳分布等于该目标分布。

&emsp;&emsp;为了了解目标分布是由Metropolis算法生成的马尔可夫链的平稳分布，考虑在时间t−1从目标分布$p(θ|y)$中抽取$θ^{t−1}$来启动算法。现在考虑任意两个从$p(θ|y)$中提取的点$θ_a$和$θ_b$，并标记为$p(θ_b|y)≥p(θ_a|y)$。从θa过渡到θb的无条件概率密度为:
$
p(θ^{t−1}=θ_a, θ^t=θ_b) = p(θ_a|y)J_t(θ_b|θ_a),
$
上式的接受概率为1，是因为我们对a和b的标记，从θb到θa过渡的无条件概率密度为：
$
p(θ^t =θ_a, θ^{t−1} =θ_b) = p(θ_b|y)J_t(θ_a|θ_b)\begin{gathered}\begin{pmatrix} \frac{p(θ_a|y)}{p(θ_b|y)}\end{pmatrix} \end{gathered}
$
$
=p(θa|y)Jt(θa|θb),
$
&emsp;&emsp;这与从$θ_a$过渡到$θ_b$的概率相同，因为我们要求$J_t（·|·）$是对称的。由于它们的联合分布是对称的，$θ_t$和$θ_{t−1}$具有相同的边际分布，因此$p(θ|y)$是θ的马尔可夫链的平稳分布。
### 11.2.3 Metropolis-Hastings算法
&emsp;&emsp;Metropolis-Hastings算法用两种方法概括了上述基本的Metropolis算法。首先，跳转规则$J_t$不再需要对称；也就是说，不要求$J_t(θ_a|θ_b)≡J_t(θ_b|θ_a)$。其次，为了纠正跳跃规则中的不对称性，比率r替换为比率：
$
r = \frac{p(θ^∗|y)/J_t(θ^∗|θ^{t−1})}{p(θ^{t−1}|y)/J_t(θ^{t−1}|θ^∗)}
$
(比率r总是被定义的，因为只有当$p(θ^{t−1}|y)$和$J_t(θ^∗|θ^{t−1})$都为非零时，才能发生从$θ^{t−1}$到$θ^∗$的跳转。)

&emsp;&emsp;允许非对称跳跃规则有助于提高随机游走的速度。与Metropolis算法相同，证明了其与目标分布的收敛性。收敛到唯一平稳分布的证明是相同的。为了证明平稳分布是目标分布，$p(θ|y)$，考虑了标记有后验密度的任意两点$θ_a$和$θ_b$，从而得到$p(θ_b|y)J_t(θ_a|θ_b)≥p(θ_a|y)J_t(θ_b|θ_a)$。如果$θ_{t−1}$遵循目标分布，那么很容易看出从$θ_a$到$θ_b$的转变的无条件概率密度与反向转变相同。

代码：
```
def get_p(x):
    return 1/(2*PI)*np.exp(- x[0]**2 - x[1]**2)
def get_tilde_p(x):
    return get_p(x)*20
#每轮采样的函数
def domain_random(): 
    return np.random.random()*3.8-1.9
def metropolis(x):
    new_x = (domain_random(),domain_random()) 
    acc = min(1,get_tilde_p((new_x[0],new_x[1]))/get_tilde_p((x[0],x[1])))
    u = np.random.random()
    if u<acc:
        return new_x
    return x

    def testMetropolis(counts = 100,drawPath = False):
    plotContour()
    x = (domain_random(),domain_random()) #x0
    xs = [x]
    for i in range(counts):
        xs.append(x)
        x = metropolis(x) #采样并判断是否接受
    if drawPath: 
        plt.plot(map(lambda x:x[0],xs),map(lambda x:x[1],xs),'k-',linewidth=0.5)
    plt.scatter(map(lambda x:x[0],xs),map(lambda x:x[1],xs),c = 'g',marker='.')
    plt.show()
    pass
```
## 11.3 使用Gibss和Metropolis为构件
&emsp;&emsp;Gibbs采样器和Metropolis算法可以用于各种组合，从复杂的分布中采样。Gibbs采样器是马尔可夫链模拟算法中最简单的一种，它是条件共轭模型的首选，我们可以直接从每个条件后验分布中采样。例如，我们可以对常态层次模型使用Gibbs采样器。

&emsp;&emsp;Gibbs采样可以看作是Metropolis-hastings算法的一个特例。我们首先将迭代t定义为由一系列d步骤组成，其中迭代t的步骤j对应于θ所有其他元素条件的子向量$θ_j$的更新。然后在迭代步骤j的分布$J_{j，t}（·|·）$只沿第j个子向量跳跃，并且根据给定$θ^{t−1}_{−j}$的$θ_j$的条件后验密度j：
$$ J^{Gibbs}_{j,t}(\theta^*|\theta^{t-1})=\begin{cases}
p(\theta^*_j|\theta^{t-1}_{-j},y) & if \theta^*_{-j}=\theta^{t-1}_{-j} \\
0 & 其它
\end{cases}$$
&emsp;&emsp;唯一可能的跳转是与除第j个以外的所有分量上的$θ_{t−1}$相匹配的参数向量$θ^∗$。在此跳跃分布下，迭代t第j步的比率为:
$
r=\frac{p(θ^∗|y)/J^{Gibbs}_{j,t}(\theta^*|\theta^{t-1})}{p(θ^{t-1}|y)/J^{Gibbs}_{j,t}(\theta^{t-1}|\theta^*)}
$
$
=\frac{p(θ^∗|y)/p(\theta^*_j|\theta^{t-1}_{-j},y)}{p(θ^{t-1}|y)/p(\theta^{t-1}_{j}|\theta^{t-1}_{-j},y)}
$
$
=\frac{p(\theta^{t-1}_{-j}|y)}{p(\theta^{t-1}_{-j}|y)}
$
$
=1
$

&emsp;&emsp;通常，Gibbs采样器的一次迭代被定义为，包含θ的d组件对应的所有d步骤，从而在每次迭代中更新所有θ。然而，只要每个组件定期更新，就可以说每个组件在每次迭代中都被更新。
## 11.4 推理和评估收敛性
&emsp;&emsp;迭代模拟推理的基本方法与一般的贝叶斯模拟相同：使用从$p(θ|y)$中所有模拟绘图的集合来总结后验密度，并根据需要计算分位数、矩和其他感兴趣的摘要。根据θ的抽样值，可以得到未观测结果的后验预测模拟。
* 迭代模拟给仿真推理的困难

1、首先，如果迭代进行的时间不够长，模拟可能完全不能代表目标分布。即使模拟达到了近似收敛，早期迭代仍然反映的是起始近似，而不是目标分布。

2、其次，它们有序列内相关性；除了任何收敛问题外，相关抽取的模拟推断通常不如来自相同数量的独立抽取的模拟推断精确。

* 解决方法

1、我们试图设计模拟运行，以允许有效地监测收敛性，特别是通过模拟起点分散在整个参数空间中的多个序列。

2、其次，我们通过比较模拟序列之间和模拟序列内部的变化来监测所有感兴趣的量的收敛性，直到“内部”变异大致等于“序列间”变异，只有当每个模拟序列的分布接近于所有混合在一起的序列的分布时，它们才能全部近似于目标分布。

3、如果模拟效率低得难以接受（在计算机上需要太多的实时时间来获得感兴趣量的后验推断的近似收敛），算法可以改变。
## 11.5 有效的模拟抽样
&emsp;&emsp;一旦模拟序列混合，我们就可以计算出任何感兴趣的$ψ$估计的近似“独立模拟抽样的有效数量”。我们首先观察到，如果每个序列内的n个模拟抽样是真正独立的，那么序列间的方差B将是后验方差$var(ψ|y)$的无偏估计，我们将从m个序列中得到总共$m*n$个独立的模拟。然而，一般来说，每个序列内的$ψ$模拟是自相关的，预期B将大于$var(ψ|y)$。

&emsp;&emsp;定义相关模拟抽样的有效样本量的一种方法是将模拟的$\overlineψ...$，平均值的统计效率作为后验均值E(ψ|y)的估计。
继续这个定义，通常使用以下相关序列平均值方差的渐近公式来计算有效样本量：
$
\lim\limits_{x\rightarrow\infty } mnvar(\overlineψ..)=(1+2\Sigma_{t=1}^\infty \rho t)var(ψ|y)
$
其中$ρt$是序列ψ在滞后t时的自相关。如果从每个m条链中提取的n个模拟是独立的，那么$var（\overlineψ..）$仅为$\frac{1}{mn}var(ψ|y)$，样本量为mn。在存在相关性的情况下，我们将有效样本量定义为:
$
n_{eff} =\frac{mn}{1 + 2\Sigma_{t=1}^\infty ρt}
$
### 11.5.1 样例：分层法态模型
&emsp;&emsp;我们用普通模型来演示，因为它足够简单，关键的计算思想不会在细节中丢失。
* 模型

&emsp;&emsp;在层次正态模型下，数据$y_{ij},i=1,···,n_j,j=1,···,J,$在每个J组内独立正态分布，均值为$θ_j$和共同方差$σ^2$。观测总数为$n=\Sigma^J_{j=1}n_j$。假设组均值服从正态分布，均值µ和方差$τ^2$未知，假设µ，$log(µ，logσ，τ)$的先验分布均匀，σ>0和τ>0;并且，$p(µ，logσ，logτ)∝τ$。如果我们将均匀分配给先验τ，后验分布将是不恰当的.

&emsp;&emsp;所有参数的联合后验密度均为：
$
p(θ, µ, log σ, log τ |y) ∝ τ \prod^J_{j=1}N(θ_j |µ, τ^2) \prod^J_{j=1}\prod^{n_J}_{j=1}N(y_{ij} |θ_j , σ^2).
$
