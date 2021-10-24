# MCMC
## 一、简介
MCMC，全称Markov chain Monte Carlo，马尔可夫链蒙特卡洛，是一种使用随机抽样来近似复杂积分的方法，在机器学习，深度学习以及自然语言处理等领域都有广泛的应用，是很多复杂算法求解的基础。MCMC由马尔可夫链和蒙特卡洛积分两部分组成。核心思想是通过构造马尔可夫链从想做积分的分布中抽样，然后利用蒙特卡洛积分来近似求解积分。  
* 解决问题：使用随机抽样来近似复杂积分  
* 贝叶斯中可解决问题：可用于求解参数的后验分布  
* 核心思想：构造马尔可夫链从目标积分的分布中抽样，然后利用蒙特卡洛积分来近似求解   
* 组成部分：蒙特卡洛积分（Monte Carlo integration）、马尔可夫链（Markov chain）  
本文将按照下面的思路介绍MCMC：
1. 蒙特卡洛积分——对应第二小节
2. 马尔可夫链——对于第三小节
3. MCMC采样框架——对应第四小节
4. MCMC采样方法——对应第五小节

## 二、蒙特卡洛积分（Monte Carlo integration）
最早的蒙特卡洛方法都是为了求解一些不太好求解的求和或者积分问题。比如积分：  
$$\theta =\int_a^bf(x)dx$$
如果不知道$f(x)$的原函数，那么这个积分就比较难求解。  
常用的思路是通过模拟求解近似值。假设函数图像如下:  
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/figure1.png)

则一个简单的近似求解方法是在$[a,b]$之间随机的采样一个点。比如$x_0$，然后用$f(x_0)$代表在$[a,b]$区间上所有的$f(x)$的值。那么定积分的近似求解为:  
$$(b-a)f(x_0)$$
然而选择一个数来近似太过粗糙。我们可以采样$[a,b]$区间的n个值：$x_0,x_1,...x_{n-1}$,用它们的均值来代表$[a,b]$区间上所有的$f(x)$的值。这样上面定积分的近似求解为:  
$$\frac{b-a}n\sum_{i=0}^{n-1}f(x_i)$$
虽然选取多个点的方法可以更好地求解出近似解，但是它隐含了一个假定，即$x$在$[a,b]$之间是均匀分布的。然而绝大部分时候，$x$在$[a,b]$之间不是均匀分布的。如果仍然采用上面的方法，模拟求出的结果很可能和真实值相差甚远。　
为了解决这个问题，即引入蒙特卡洛方法。蒙特卡洛利用了概率分布函数这一特征。假设$x$在$[a,b]$的概率分布函数$p(x)$，那么我们的定积分求和可以这样进行：  
$$\theta = \int_a^bf(x)dx=\int_a^b\frac{f(x)}{p(x)}p(x)dx\approx \frac{1}{n}\sum_{i=0}^{n-1}\frac{f(x_i)}{p(x_i)}$$
上式最右边的这个形式就是蒙特卡洛方法的一般形式。（这里是连续函数形式，离散情况一样成立)  
比如，假设$x$在$[a,b]$上是均匀分布，即$p(x_i)=\frac{1}{(b-a)}$。那么利用蒙特卡洛积分求解  
$$\int_a^bf(x)dx=\frac{1}{n}\sum_{i=0}^{n-1}\frac{f(x_i)}{p(x_i)}=\frac{(b-a)}{n}\sum_{i=0}^{n-1}f(x_i)$$
然而，虽然蒙特卡洛积分能够较好地求解积分问题，它在应用上仍然存在两个缺点：  
* 面对一些不常见的分布的时候，在很多时候很难获取它的概率分布函数，也很难得到相应概率分布的样本集。
* 同时，对于一些高维的复杂非常见分布，其先验和后验分布均是高维的。这种情况通常被描述为维度诅咒（curse of dimensionality）。这就意味着当维度增加时，空间$x_j$的体积增加得非常之快（以指数增加），可用的数据变得稀疏，数据的统计意义下降。仅仅基于蒙特卡洛的概率分布采样方法很难获取所需的样本集。  

因此，要想将蒙特卡罗方法作为一个通用的采样模拟求积分的方法，就需要从分布中抽样，得到各种复杂概率分布的对应的采样样本集$x^{(1)},x^{(2)},...,x^{(n)}$。这时MCMC引入了马尔可夫链来解决这个需求。  
 
## 三、马尔可夫链（Markov chain）
蒙特卡洛积分是利用有关分布的随机抽样来近似积分的一种有利方法，但从有关概率分布中抽样很困难或者无法直接做到，因此MCMC加入了马尔可夫链。马尔可夫链是一种序贯模型，它以概率的方式从一种状态转移到另一种状态，其中链所采取的下一个状态取决于以前的状态（即马尔可夫性）。如果马尔可夫链构造得当并运行很长时间，那么它也将从目标概率分布中提取样本的状态这也就是马氏链能够以较小的样本得到好的近似的根本。  
 
MCMC使用马尔可夫链机制生成样本$x^{(i)}$，这种构造是为了使链花费更多时间在最重要的地方。特别的是，它使样本$x^{(i)}$模拟从目标分布$p(x)$中提取样本，当然，用MCMC不能直接从$p(x)$中提取样本，但可以估计$p(x)$到一个标准化常数的程度。  

**马尔可夫链的定义如下：**  
假设我们在有限空间$\chi = {x_1,x_2,...,x_s}$上引入马尔可夫链，令$x^{(i)}$为马尔可夫链的第$i$个状态，这里$x^{(i)}\in \chi$只能由$s$个离散值。满足下面性质的随机过程$x^{(i)}$被称为马尔可夫链：  
$$p(x^{(i)}|x^{(i-1)},x^{(i-2)},...x^{(1)})=p(x^{(i)}|x^{(i-1)}), \forall i$$  
式中的$p(x^{(i)}|x^{(i-1)})$称为从状态x^{(i-1)到状态(x^{(i)的转移概率。上式意味着随机过程处于某状态的概率仅依赖于前一个状态，与之前的历史无关，这就是马尔可夫链的马尔可夫性。  

**马尔可夫链中的4个概念：**  
1. <b>时齐</b>  
如果对于任意两个状态$x_i,x_j \in \chi$，$s \times s$矩阵$\pmb P \equiv [p_{ij}]=[p(x^{(k)}|x^{(k-1)}=x_i)]$对所有$k$保持不变，而且$\sum_i P_{ij}=1$，则称马尔可夫链是时齐的。 

2. <b>转移概率和转移概率矩阵</b>  
转移概率是马尔可夫链中的重要概念，若马氏链分为m个状态组成。从任意一个状态出发，经过任意一次转移，必然出现状态$1,2,...m$中的一个，这种状态之间的转移称为转移概率。   
其表达式为  $$P_{ij}(m,m+n)=P(X_{m+n}=j|X_m=i)$$
称$p_{ij}(m,m+n)$为链在$m$时刻处于$i$状态,再经$n$步转移到$j$状态的转移概率,简称$n$步转移概率。  
如果以$p_{ij}(m,m+n)$ 作为矩阵$\pmb P(m,m+n)$的第$i$行第$j$列元素,则$\pmb P(m,m+n)$称为马氏链的$n$步转移阵。值得注意的是,当$n=1$时，一步转移概率为$p_{ij}(m,m+1)$。其组成的是一步转移概率矩阵，记为$\pmb P_{ij}(m)$。  

    1. 转移概率性质  
    * 对$\forall m,n,i,j$，有$p_{ij}(m,m+n)\geq 0$
    * 对$\forall m,n,i$，有$\sum_{j\in E}p_{ij}(m,m+n)=1$

    2. 一步转移概率矩阵性质
    * $0\leq \pmb P_{ij}(m) \geq 1,i\in I$
    * $\sum_{j\in I}\pmb P_{ij}(m)\leq 1$

    3. 转移矩阵条件  
    为了保证初始概率分布不影响后续稳定后的概率，转移矩阵需要满足以下两个条件：
    * 不可约性（irreducibility）  
    即非负性，从任何状态出发访问马尔可夫链的任何其他状态的概率都是正的。同时也意味着，转移矩阵不能再简化为更小的矩阵。  
    * 非周期性（aperiodicity）  
    马氏链不能被困在循环中，即需要马氏链是非周期性的。  
    假设有一个状态$x_i$，它经历$t$步之后回到自身状态的转移概率为$\pmb P_{ii}^t$，则周期为：  $$k=gcd(n|\pmb P_{ii}^n>0)$$
    式子中，$gcd$表示最大公约数。如果$k=1$，则状态$x_i$称为非周期性的。  
    如果每个状态都是非周期性的，那么马尔可夫链是非周期性的，一个不可约的马尔可夫链需要是非周期性的。  

 
3. 平稳概率(stationary distribution)  
对于任意初始分布$\pmb \pi_0 = (\pi_1^{(0)},\pi_2^{(0)},...\pi_s^{(0)})$，在$t$次转换之后在各个状态的概率为向量  $$\pmb \pi_0\overbrace{PP...P}^t = \pmb \pi_0P^t$$
很容易验证，当$t$很大时，概率向量$\pi_0P^t$收敛到一个稳定的值$\pi$，而且该值和初始概率$\pi_0$无关，即  $$\lim_{t\rightarrow \infty}\pmb \pi_0P^t = \pmb \pi$$
乘积矩阵$\pmb P^t$的元素$P_{ij}^t$为从状态$x_i$经过$x$步转移到状态$x_j$的概率，显然有如下关系式：  $$\pmb \pi = \pmb \pi \pmb P$$
概率$\pmb \pi$称为平稳概率（stationary distribution或者equilibrium distribution），求解平稳概率即是求解$\pmb P^\mathsf{T}$特征值问题，这也是转移矩阵必须满足两大条件的一个原因。  
R的伪代码如下：
    ```R 
    状态转移矩阵P = matrix(矩阵数据)
    初始概率p0 = c(数据)
    p1 = p0  # 进行到下一步
    for(i in 1:迭代次数)
    P1 = P1%*%P
    P1  # 返回结果
    ```
    Python的伪代码如下：  
    ```Python
    状态转移矩阵A=[数据]
    阶段n步=状态转移矩阵A
    阶段n1步=np.dot(状态转移矩阵A,状态转移矩阵A)
    while not (阶段n步 == 阶段n1步).all():
        阶段n步 = 阶段n1步
        阶段n1步 = np.dot(阶段n1步,状态转移矩阵A)
    print(阶段n步)
    ```
    【举例】有三种状态：牛市（Bull market）, 熊市（Bear market）和横盘（Stagnant market）。每一个状态都以一定的概率转化到下一个状态。如果我们定义矩阵$\pmb P$某一位置$P(i,j)$的值为$P(j|i)$,即从状态$i$转化到状态$j$的概率。定义牛市为状态0，熊市为状态1, 横盘为状态2. 可以得到了马尔科夫链模型的状态转移矩阵为：
    $$P=\left(
    \begin{matrix}
    0.9 & 0.075 & 0.025\\
    0.15 & 0.8 & 0.05 \\
    0.25 & 0.25 & 0.5
    \end{matrix}
    \right)$$
    假设当前股市的概率分布为：$[0.3,0.4,0.3]$，以Python代码实践马尔可夫链，代码如下：
    ```Python
    import numpy as np
    matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=float)
    vector1 = np.matrix([[0.3,0.4,0.3]], dtype=float)
    for i in range(100):
        vector1 = vector1*matrix
        print "迭代次数:" , i+1
        print vector1
    ```
    结果如下：
    ```bochs
    迭代次数: 1
    [[0.405  0.4175 0.1775]]
    迭代次数: 2
    [[0.4715  0.40875 0.11975]]
    迭代次数: 3
    [[0.5156 0.3923 0.0921]]
    迭代次数: 4
    [[0.54591  0.375535 0.078555]]
    迭代次数: 5
    [[0.567288 0.36101  0.071702]]
    ...
    迭代次数: 60
    [[0.625  0.3125 0.0625]]
    迭代次数: 61
    [[0.625  0.3125 0.0625]]
    ...
    迭代次数: 99
    [[0.625  0.3125 0.0625]]
    迭代次数: 100
    [[0.625  0.3125 0.0625]]
    ```
    可以发现，从第60轮开始，我们的状态概率分布就不变了，一直保持在[0.625   0.3125  0.0625]，那么它就是所求的平稳概率$\pmb \pi$。（更改初始概率发现结果也稳定在该矩阵）

4. 遍历原理（erigodic theorem）  
所谓遍历原理，即指如果马尔可夫链  $$x^{(1)},x^{(2)},...x^{(n)}\sim \pmb \pi$$
是非周期性和不可约的，而且$\pmb \pi$是平稳分布，那么，当$n\rightarrow \infty$时，有  $$\frac{1}{n} \sum_n^{t=1}h(x^{(t)}) \rightarrow E_{\pmb \pi}[h(x)]=\int h(\pmb x)\pmb\pi(\pmb x)d\pmb x$$  
  

**基于马尔可夫链的采样**  
基于以上概念，我们可以得到基于马尔可夫链的采样过程
根据某个平稳分布所对应的马尔科夫链状态转移矩阵，我们就很容易采用出这个平稳分布的样本集。
假设我们任意初始的概率分布是$\pi_0(x)$, 经过第一轮马尔科夫链状态转移后的概率分布是$\pi_1(x)$，...第i轮的概率分布是$\pi_i(x)$。假设经过$n$轮后马尔科夫链收敛到平稳分布$\pi(x)$，即：
$$\pi_n(x)=\pi_{n+1}(x)=\pi_{n+2}(x)=...\pi(x)$$
对于每个分布$\pi_i(x)$，我们有：
$$\pi(x)=\pi_{i-1}(x)\pmb P=\pi_{i-2}(x)\pmb P^2=\pi_0(x)\pmb P^i$$
基于此采样步骤如下：  
首先，基于初始任意简单概率分布比如高斯分布\pi_0(x)采样得到状态值$x_0$，基于条件概率分布$p(x|x_0)$采样状态值$x_1$，一直进行下去；当状态转移进行到一定的次数$n$时，可以认为此时的采样集$(x_n,x_{n+1},x_{n+2},...)$即是符合所求平稳分布并可以用来做蒙特卡罗模拟求和的对应样本集。

  
## 四、MCMC采样框架  
1. 输入目标平稳分布$\pi(x)$，设定需要的样本个数$n_2$
2. 从任意简单概率分布采样得到初始状态值$x_0$
3. 从$t=0$到$n_1+n_2−1$: 
    * 从条件概率分布$Q(x|x_t)$中采样得到样本$x_∗$
    * 从已有分布中采样
    * 设定规则，接受或拒绝候补样本

最终样本集$(x_{n1},x_{n1+1},...,x_{n1+n2−1})$为我们需要的平稳分布对应的样本集。  

  
## 五、MCMC采样方法
### 方法综述  
使用马尔可夫链从特定的目标分布中进行抽样，关键是必须设计合适的转移矩阵（算子），以便生成的链达到与目标分布相匹配的稳定分布，这也是所有MCMC方法的核心目标。 
下面简要介绍三个MCMC采样的算法：  
* Metropolis 算法
* M-H算法
* Gibbs Samper

### Metropolis算法
根据第2、3小节，可以知道如果假定可以得到目标采样样本的平稳分布所对应的马尔科夫链状态转移矩阵，那么就可以使用马尔科夫链采样得到需要的样本集，进而进行蒙特卡罗模拟。  

但是一个重要的问题是，面对任意的平稳分布$\pi$，如何能得到它所对应的马尔科夫链状态转移矩阵$P$？

这时我们需要考虑马尔可夫链的一大性质——细致平稳条件（detailed balance condition）细致平稳条件指，如果非周期马氏链的转移矩阵$\pmb P$和分布$\pi(x)$满足$$\pi(i)\pmb P_{ij}=\pi(j)\pmb P_{ij}\qquad for\;all\quad i,j$$
则$\pi(x)$是马氏链的平稳分布。  
由细致平稳条件可以得到：
$$\sum_{i=1}^{\infty}\pi(i)\pmb P(i,j)=\sum_{i=1}^{infty}\pi(j)\pmb P_{ij}=\pi(j)\sum_{i=1}^{infty}\pmb P_{ij}=\pi(j)$$
矩阵表示为：
$$\pi \pmb P=\pi$$
为了更方便地找到对应的转移矩阵，需要对上式做一个改造，引入一个$\alpha(i,j)$)，即：
$$\pi(i)Q(i,j)\alpha(i,j)=\pi(j)Q(i,j)\alpha(j,i)$$
$\alpha(i,j)$和$\alpha(j,i)$需要满足下两式：  
* $\alpha(i,j)=\pi(j)Q(j,i)$
* $\alpha(j,i)=\pi(i)Q(i,j)$
则分布$\pi(x)$对应的马尔科夫链状态转移矩阵$\pmb P$为
$$P(i,j)=Q(i,j)\alpha(i,j)$$
$\alpha(i,j)$一般称为接受率,取值在$[0,1]$之间，可以理解为一个概率值,即目标矩阵$\pmb P$可以通过任意一个马尔科夫链状态转移矩阵$\pmb Q$以一定的接受率获得。  

Metropolis算法就是使用简单的启发式方法来实现这样的过渡算子，使链的平稳分布于目标分布相匹配。  

Metropolis方法从一些随机初始状态$x^{(0)}\sim \pi^{(0)}$开始，该算法首先从类似于马尔可夫链转移概率的分布$q(x|x^{(t-1)})$中提取可能的候选样本$x^{*}$，该候选样本收到Metropolis方法的一个额外步骤的评估，看目标分布在其附件是否有足够大的密度，以确定是否接受其作为链的下一个状态，如果$p(x^*)$的密度低于建议的状态，则它可能被拒绝。接受或拒绝候补状态的标准由以下直观方法定义：  
1. 如果$p(x^*)\geq p(x^{(t-1)})$,则保留候补状态$x^*$作为链的下一个状态，也就是马氏链的$p(x)$不能减少  
2. 如果$p(x^*) < p(x^{(t-1)})$，这就说明在$x^*$附近密度$p(x)$较小，候选状态仅仅以概率$p(x^*)/p(x^{(t-1)})$保留

为了说明，设置接受概率  
$$\alpha = min(1,\frac{p(x^*)}{p(x^{(t-1)})})$$
有了接受概率，Metropolis算法的转移运算符则运行如下：如果均匀随机数$\mu$小于等于$\alpha$，则接受状态$x^*$，否则拒绝$x^*$并建议下一个候选状态。  
下面是收集M个样本的伪代码：  
```
set t = 0
从初始状态上的先验分布生成初始状态
repeat,until t = M
    set t = t + 1
    从转移概率生成候补状态
    计算接受概率
    从Unif(0,1)中抽取随机数μ
        if μ小于等于α，接受候补状态
        else 不接受
```

**Metropolis——Python案例实践**  
下面给出利用Metropolis采样算法对t分布进行采样的一个例子。t分布的概率密度函数为：  
$$f(x,n)=\frac{\Gamma(\frac{n+1}{2})}{\sqrt{n\pi}\Gamma(\frac{n}{2})}(1+\frac{x^2}{n})^{-\frac{n+1}{2}}$$  
其中$n$是$t$分布的自由度。下图是自由度为3的$t$分布的概率密度函数分布图。  
【图像】  
下面的例子中自由度取3，样本序列的初始值从均匀分布中随机选取，从正态分布中采集候选样本，迭代次数为10000次，取样本序列的第9001到第10000个样本作为从t分布中采集的样本的近似，并画出选取的1000个样本的分布直方图，与标准分布进行比较。   
```Python
import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

# 自由度为3的t分布
def t_distribution(x):    # n=3
    p = 2/(np.sqrt(3)*np.pi*np.square(1+np.square(x)/3))
    return p

T = 10000   # 迭代次数
sigma = 1.  # 正态分布标准差
sample_x = np.zeros(T+1)
sample_x[0] = uniform.rvs(size=1)[0]   # 初始化马尔科夫链初值
for i in range (1, T+1):
    hat_x = norm.rvs(loc = sample_x[i-1], scale=sigma, size=1, random_state=None)   # 从正态分布中生成候选值
    alpha = min(1, t_distribution(hat_x[0])/t_distribution(sample_x[i-1]))  # 计算接受概率
    alpha_t = uniform.rvs(size=1)[0]  # 生成接受概率判定值
    if alpha_t <= alpha :      # 若判定值小于接受概率则接受候选值，否则拒绝候选值
        sample_x[i] = hat_x[0]
    else:
        sample_x[i] = sample_x[i-1]

fig, ax = plt.subplots(1, 1)
df = 3   # t分布的自由度为3
mean, var, skew, kurt = t.stats(df, moments='mvsk')
x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)
p1 = ax.plot(x, t.pdf(x, df), 'k-', lw=5, alpha=0.6, label='t')     # pdf: Probability density function；画自由度为3的标准t分布曲线
p2 = plt.hist(sample_x[9001:], 100,density=True,alpha=0.5, histtype='stepfilled', facecolor='red', label='sample_t')   # 画生成的马尔科夫链的标准化柱状图
plt.legend()
plt.show()
```
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/Metorpolis_Python.png)  
由结果图可以看到，通过Metropolis采样算法采集到的样本序列可以近似地看做自由度为3的t分布的样本。  

**Metropolis——R案例实践**  
考虑形状参数为$\alpha$，尺度参数为$s$的Gamma分布
$$p(x)=\frac{1}{s^a\Gamma(a)}x^{a-1}e^{-\frac{x}{s}}$$
下面是Gamma分布（取$a=s=5$）的Metropolis抽样的R代码。其中转移概率取正态分布$N(0，4)$，前面一半的抽样值算是用于热身（burn-in）最终舍弃：  
```R
M=20000;
k=floor(M/2)
X=NULL
x=1
set.seed(1010)
for (i in 1:M){
  u = rnorm(1,0,4)
  alpha = dgamma(u,5,5)/dgamma(x,5,5)
  if (runif(1) < min(alpha,1)) x=u
  X[i]=x
}
layout(t(1:2))
hist(X[-(1:k)],20,prob=TRUE,xlim = c(0,8),xlab = 'X',ylab = '',main="")
curve(dgamma(x,5,5),from=0,to=8,add=TRUE,col=2,lwd=3)

plot(1:k,X[1:k],type = 'l',col=2,lty=2,ylab = 'X',xlab = "index",xlim = c(1,M),ylim = range(X))
lines((k+1):M,X[(k+1):M])
```
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/Metropolis_R.jpeg)  
左图为样本的直方图和真实分布，可以看出其符合较好。
右图为马尔可夫链（粉色为热身部分）。 


### M-H算法
Metropolis算法的一个约束是，建议转移概率分布$q(x|x^{(t-1)})$必须是对称的，为了能够使用非对称的转移概率分布，Metropolis-Hastings算法（简称M-H算法）增加一个基于建议的转移概率分布额外的校正因子$c$:
$$c=\frac{q(x^{(t-1)|x^*})}{q(x^*|x^{(t-1)})}$$
校正因子调整转移算子，以确保$x^{(t-1)}\rightarrow x^{(t)}$的转移概率等于$x^{(t)}\rightarrow x^{(t-1)}$的转移概率。  
M-H算法的实现过程于Metropolis算法基本相同，只是在接受概率的评估中使用了校正因子。  
具体收集$M$个样本的伪代码如下：  
```
set t = 0
从初始状态上的先验分布生成初始状态
repeat,until t = M
    set t = t + 1
    从转移概率生成候补状态
    计算接受概率[注意这里的与Metropolis的区别]
    从Unif(0,1)中抽取随机数μ
        if μ小于等于α，接受候补状态
        else 不接受
```
这里接受概率为：  
$$\alpha = min(1,\frac{p(x^*)}{p(x^{(t-1)})}*c)$$
显然，在对称分布时，$c=1$，就是Metropolis算法。  

**M-H算法——Python案例实践**  
目标平稳分布是一个均值3，标准差2的正态分布，而选择的马尔可夫链状态转移矩阵$Q(i,j)$的条件转移概率是以$i$为均值,方差1的正态分布在位置$j$的值。
```Python
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))

    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t - 1]


plt.scatter(pi, norm.pdf(pi, loc=3, scale=2),label='target')
num_bins = 50
plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.7,label='sample')
plt.legend()
plt.show()
```
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/M-H_Python.png)  
输出的图中可以看到采样值的分布与真实的分布之间的关系如下，采样集还是比较拟合对应分布的。  

**M-H算法——R案例实践**  
考虑形状参数为$\alpha$，尺度参数为$s$的Gamma分布
$$p(x)=\frac{1}{s^a\Gamma(a)}x^{a-1}e^{-\frac{x}{s}}$$  
下面是Gamma分布的M-H抽样的R代码：
```R
M=20000;
k=floor(M/2)
X=vector()
x=1
set.seed(1010)
for (i in 1:M){
  ch<-rchisq(1,3)
  alpha = dgamma(ch,5,5)/dgamma(x,5,5)*(dchisq(ch,3)/dchisq(x,3))
  if (runif(1) < min(alpha,1)) x=ch
  X[i]=x
}
hist(X[-(1:k)],15,prob=TRUE,xlim = c(0,8),ylim=c(0,1),xlab = 'X',ylab = '',main="")
curve(dgamma(x,5,5),from=0,to=8,add=TRUE,col=2,lwd=3)

plot(1:k,X[1:k],type = 'l',col=2,lty=2,ylab = 'X',xlab = "index",xlim = c(1,M),ylim = range(X))
lines((k+1):M,X[(k+1):M])
```
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/M-H_R.jpeg)  
左图为样本的直方图和真实分布，可以看出其符合较好。 
右图为马尔可夫链（粉色为热身部分）。 
 

  
### Gibbs抽样   
 **理论**  
已发现在许多多维问题中有用的特定马尔可夫链算法是Gibbs采样器，也称为altering conditional sampling。  
根据《Bayesian Data Analysis》，Gibbo抽样由$\theta$子向量定义。假设参数向量$θ$已被划分为$d$个分量或子向量，$\theta=(\theta_1,...\theta_d)$。Gibbs采样器的每次迭代循环都遍历$\theta$的子向量，根据所有其他子集的值绘制每个子集。
在每次迭代时，选择$θ$的$d$个子向量的顺序，然后，每个$\theta_j$从给定所有$\theta$的其他分量的条件分布$$p(\theta_j|\theta_{-j},y)$$中采样。  
其中$\theta_{-j}$表示$\theta$除了$\theta_j$外的所有分量，公式如下：  $$\theta_{-j}=(\theta_1,...,\theta_{j-1},\theta_{j+1},...,\theta_d)$$
因此，每个子向量$\theta_j$依据$θ$的其他分量的最新值更新。  
简单来说，Gibbs抽样是MCMC采样的一个特例，它交替的固定某一维度$x_i$，然后通过其他维度$x_{-i}$的值来抽样该维度的值。因此，Gibbs采样针对的是高维对象（2维以上）。其抽样工作方式与M-H方法大同小异，但需要从变量的相应条件部分的一维中抽样，并接受抽到的所有值。接着序贯地对每个变量进行抽样，同时保持所有其他变量固定。因此Gibbs适用于容易得到条件分布，而且都是熟悉的分布形式情况。  
Gibbs采样的伪代码如下：  
```
set t=0
生成初始状态矩阵
repeat,until t = M
    set t=t+1
    对于i=1,2,...,D的每一维
    从条件分布中抽样
```

**Gibbs——Python案例实践**  
【案例】假设我们要采样的是一个二维正态分布$Norm(\mu,\Sigma)$,其中：  
$$\mu=(\mu_1,\mu_2)=(5,−1)$$
$$\Sigma=
    \begin{pmatrix}
    \sigma_1^2 & \rho\sigma_1\sigma_2,\\
    \rho\sigma_1\sigma_2 & \sigma_2^2\\
    \end{pmatrix}
    $$
$$\Sigma=
    \begin{pmatrix}
    1 & 1,\\
    1 & 4\\ 
    \end{pmatrix}
    $$
    
而采样过程中的需要的状态转移条件分布为：  
$$P(x_1|x_2)=Norm(\mu_1+\rho\sigma_1 / \sigma_2(x_2-\mu_2),(1-\rho^2)\sigma_1^2)$$
$$P(x_2|x_1)=Norm(\mu_2+\rho\sigma_2 / \sigma_1(x_1-\mu_1),(1-\rho^2)\sigma_2^2)$$
具体实现代码如下：  
```Python
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
samplesource = multivariate_normal(mean=[5,-1], cov=[[1,1],[1,4]])

def p_ygivenx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt((1 - rho ** 2) * (s2**2))))

def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt((1 - rho ** 2) * (s1**2))))

N = 5000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

rho = 0.5
y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiveny(y, m1, m2, s1, s2)
        y = p_ygivenx(x, m1, m2, s1, s2)
        z = samplesource.pdf([x,y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.hist(x_res, num_bins, density=True, facecolor='green', alpha=0.5,label='feature1')
plt.hist(y_res, num_bins, density=True, facecolor='red', alpha=0.5,label='feature2')
plt.title('Histogram')
plt.legend()
plt.show()
```
输出的两个特征各自的分布如下：  
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/Gibbs_Python1.png)  
查看样本集生成的二维正态分布，代码如下：  
```Python
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o',color='pink')
plt.show()
```
输出的正态分布图如下：  
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/Gibbs_Python2.png)  
可以看出，Gibbs抽样得到的采样集还是比较拟合对应分布的。  

**Gibbs——R案例实践**  
考虑均值为$\mu$，精度为$\tau=1/\sigma^2$的正态分布$N(\mu,\tau)$。这里的维数$D=2$，假定数据$\pmb x=(x_1,x_2,...x_n)$的样本量为$n=70$，样本均值和样本标准差分别为$\bar{x}=8$，以及$s=2$  
重复抽样，在第$i$步，从$f(\mu|\tau^{(t-1)})$抽取$\mu^{(i)}$，从$f(\tau|\mu^{(i)})$抽取$\tau^{(i)}$，为此需要分别得到有关条件分布的解析表达式。假定先验分布为：  
$$p(\mu,\tau)=p(\mu)p(\tau),\;p(\mu)\propto 1 / \tau$$  
需要的条件分布为：  
$$(\mu|\tau,\pmb x)\sim N(\bar{x},\frac{1}{n\tau})$$
$$(\tau|\mu,\pmb x)\sim Gamma(\frac{n}{2},\frac{2}{(n-1)s^2+n(\mu-\bar{x})^2})$$  
下面是关于该情况的Gibbs抽样R代码。
```R
n = 70;
xbar = 8;
s2 = 4
N = 99999;
k=5000
mu = vector()->tau
tau[1] = 1
set.seed(1010)
for (i in 2:M){
  mu[i] = rnorm(n=1,mean = xbar,sd=sqrt(1/n*tau[i-1]))
  tau[i] = rgamma(n=1,shape = n/2,scale = 2/((n-1)*s2+n*(mu[i]-xbar)^2))
}
par(mfrow=c(1,2))
hist(mu[-(1:k)])
hist(tau[-(1:k)])
```
![avatar](https://github.com/hx-ling/2021BayesianCourse/blob/main/figure/Gibbs_R.jpeg)  
左右图分别是$\mu$和$\tau$后验分布去掉热身部分后的直方图  

## 六、MCMC案例应用
**前提知识：**
对于贝叶斯推断，MCMC模拟一个自选初始点$\theta^{(0)}$的离散时间马尔可夫链，产生一串相依的随机变量${\{\theta^{(i)}}\}_{i=1}^M$，有近似分布  $$p(\theta^{(i)})\approx p(\theta |x)$$
由于马尔可夫性，$\theta^{(i)}$的分布仅仅和$\theta^{(i-1)}$有关。MCMC在状态空间$\theta\in\Theta$产生了一个马尔可夫链${\{\theta^{(1)},\theta^{(2)},...\theta^{(M)}}\}$，其每个样本都假定来自稳定分布$p(\theta|x)$，即后验分布。因此通过MCMC，能够获取参数的后验分布。  

**MCMC案例实践**   
在上述基础上，下面展示一个MCMC的案例实践，在本案例中使用MCMC求参数的后验分布。  

【案例】本数据时关于对8所学校的短期训练效果的研究，来自Gelman et al.（2003）并且倍Sturtz et al.（2005）使用。学术能力检测（scholoastic aptitude test，SAT）测量高中生能力，以帮助大学做出录取决定。它分为两部分：口头（SAT-V）和数学（SAT-M）。这个数据来自8所不同高中的SAT-V（SAT-Verbal），源于20世纪70年代后期的一项实验。在8所不同的学校中，每所学校约有60个对象，他们都已经参加了PSAT（Preliminary SAT），结果被用作协变量。对于每所学校，给出了估计的短期训练效果（处理效应）和他们的标准误差。数据中的这些结果是通过适用于完全随机化实验的协方差调整的线性回归分析来计算的。  

下表为该案例数据：  
|   学校   | A | B | C | D | E | F | G | H |
|  ----  | ----  |----  |----  |----  |----  |----  |----  |----  |
| 处理效应$({\{y_i}\})$ | 28.39 | 7.94| -2.75| 6.82|-.064|0.63|18.01|12.16
| 标准误差$({\sigma_i})$  | 14.90|10.20|16.30|11.00|9.40|11.40|10.40|17.60|

利用R/Stan实现R
```R
# 输入list形式的数据
schools.data<- list(
  J = 8,
  y = c(28.39,7.94,-2.75,6.82,-.064,0.63,18.01,12.16),
  sigma = c(14.90,10.20,16.30,11.00,9.40,11.40,10.40,17.60)
)

# 建立Stan模型SNC_model
SNC_model =" 
data{
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters{
  real mu;
  real<lower=0> tau;
  real theta_tilde[J];
}

transformed parameters{
  real theta[J];
  for (j in 1:J)
    theta[j] = mu + tau * theta_tilde[j];
}
model{
  mu ~ normal(0,5);
  tau ~ cauchy(0,5);
  theta_tilde~normal(0,1);
  y~normal(theta,sigma);
}"

# 利用程序包rstan来运行Stan模型（需提前安装好rstan）
library(rstan)
library(StanHeaders)
library(ggplot2)
fit<- stan(
  model_code = SNC_model,  # Stan程序
  data = schools.data,     # 数据（变量名）
  chains = 2,              # 用2条马尔可夫链
  warmup = 1000,           # 每条链的热身次数
  iter = 2000,             # 每条链的迭代总次数
  refresh = 1000           # 每1000次迭代显示过程
)

# 输出结果
print(fit)
```
得到各个参数后验分布的汇总：一共19个参数，包括模型18个以及一“lp__”，它是所有MCMC抽样得到的参数代入模型中变量y的后验分布所得到的对数似然向量。  
```bochs
> # 输出结果
> print(fit)
Inference for Stan model: 77ceca9ec6e72da9312fe4b4b96a677d.
2 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=2000.

                mean  se_mean sd    2.5%  25%   50%   75%  97.5%  n_eff  Rhat
mu              4.54    0.06 3.34  -1.89  2.31  4.56  6.79 11.16  2722    1
tau             3.54    0.08 3.10   0.15  1.28  2.82  4.89 11.42  1437    1
theta_tilde[1]  0.34    0.02 0.95  -1.58 -0.30  0.36  0.95  2.15  2336    1
theta_tilde[2]  0.09    0.02 0.96  -1.76 -0.53  0.08  0.74  2.01  2546    1
theta_tilde[3] -0.09    0.02 0.94  -1.88 -0.75 -0.10  0.54  1.66  2197    1
theta_tilde[4]  0.07    0.02 0.94  -1.87 -0.53  0.09  0.70  1.94  2305    1
theta_tilde[5] -0.11    0.02 0.93  -1.96 -0.71 -0.12  0.49  1.78  2838    1
theta_tilde[6] -0.08    0.02 0.98  -1.93 -0.77 -0.10  0.57  1.82  2745    1
theta_tilde[7]  0.33    0.02 0.99  -1.66 -0.33  0.32  1.03  2.23  2369    1
theta_tilde[8]  0.07    0.02 1.00  -1.87 -0.61  0.09  0.72  2.03  2412    1
theta[1]        6.31    0.12 5.45  -2.66  2.91  5.89  9.04 18.84  1923    1
theta[2]        5.00    0.10 4.74  -3.96  1.91  4.85  7.96 14.90  2290    1
theta[3]        3.98    0.12 5.48  -7.89  1.20  4.36  7.43 14.10  1934    1
theta[4]        4.97    0.11 4.91  -4.54  2.03  4.85  7.97 14.80  1926    1
theta[5]        4.14    0.09 4.63  -5.86  1.45  4.29  7.08 12.45  2399    1
theta[6]        4.09    0.11 4.87  -6.85  1.27  4.26  7.32 12.72  2032    1
theta[7]        6.19    0.11 5.02  -2.97  2.97  5.74  9.12 17.27  2039    1
theta[8]        5.00    0.12 5.35  -5.31  1.86  4.89  7.97 16.09  2049    1
lp__           -6.92    0.08 2.29 -12.26 -8.26 -6.65 -5.23 -3.45   787    1

Samples were drawn using NUTS(diag_e) at Tue Oct 19 19:30:30 2021.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

利用Python/PyMC3实现
```Python
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use('seaborn-darkgrid')

# 输入数据
J = 8
y = np.array([28.39,7.94,-2.75,6.82,-.064,0.63,18.01,12.16])
sigma = np.array([14.90,10.20,16.30,11.00,9.40,11.40,10.40,17.60])

# 定义模型
with pm.Model() as NC:
    mu = pm.Normal('mu',mu=0,sd=5)
    tau = pm.HalfCauchy('tau',beta=5)
    theta_tilde = pm.Normal('theta_t',mu=0,sd=1,shape=J)
    theta = pm.Deterministic('theta',mu+tau*theta_tilde)
    obs = pm.Normal('obs',mu=theta,sd=sigma,observed=y)

# MCMC抽样
with NC:
    fit = pm.sample(5000,chains=2,tune=1000,random_seed=[20190818,20191010],target_accept=.90)

# 查看后验分布的汇总
pm.summary(fit).round(2)
```
查看各个后验分布的汇总如下：
```bochs
	mean	sd	hdi_3%	hdi_97%	mcse_mean	mcse_sd	ess_bulk	ess_tail	r_hat
mu	4.46	3.33	-1.78	10.85	0.03	0.02	11373.0	6727.0	1.0
theta_t[0]	0.33	0.98	-1.42	2.22	0.01	0.01	11232.0	7815.0	1.0
theta_t[1]	0.09	0.92	-1.69	1.74	0.01	0.01	10052.0	6972.0	1.0
theta_t[2]	-0.09	0.99	-1.91	1.80	0.01	0.01	11300.0	7258.0	1.0
theta_t[3]	0.05	0.94	-1.71	1.83	0.01	0.01	16072.0	7414.0	1.0
theta_t[4]	-0.14	0.93	-1.89	1.58	0.01	0.01	12465.0	6921.0	1.0
theta_t[5]	-0.07	0.95	-1.84	1.73	0.01	0.01	11263.0	7355.0	1.0
theta_t[6]	0.33	0.97	-1.50	2.15	0.01	0.01	11102.0	7737.0	1.0
theta_t[7]	0.10	0.97	-1.74	1.90	0.01	0.01	13802.0	7902.0	1.0
tau	3.59	3.21	0.00	9.32	0.04	0.03	5308.0	3783.0	1.0
theta[0]	6.34	5.62	-3.56	17.05	0.06	0.05	9002.0	7987.0	1.0
theta[1]	4.93	4.70	-3.75	14.08	0.04	0.04	11273.0	8178.0	1.0
theta[2]	3.97	5.26	-5.40	14.66	0.05	0.04	10483.0	7841.0	1.0
theta[3]	4.76	4.87	-4.79	13.67	0.04	0.04	12917.0	8017.0	1.0
theta[4]	3.87	4.70	-5.11	12.77	0.04	0.04	11946.0	7867.0	1.0
theta[5]	4.04	4.91	-4.93	13.64	0.05	0.04	10794.0	8105.0	1.0
theta[6]	6.26	5.23	-3.20	16.45	0.05	0.04	11056.0	7647.0	1.0
theta[7]	5.01	5.25	-5.32	14.47	0.05	0.04	11439.0	7908.0	1.0
```
## 七、参考文献
1. 吴喜之(2020),《贝叶斯数据分析——基于R与Python的实现》,中国人民大学出版社
2. Anrew Geiman(2013),_Bayesian Data Analysis_,CRC Press
