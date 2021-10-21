# 马尔科夫计算的有效性
## 窦奇峰 2019302130051
# 一、有效的Gibbs采样器
## （1）转换和重新参数化
&emsp;&emsp;Gibbs采样器在参数化独立分量时是最有效的。重新参数化最简单的方式是通过参数的线性变换，但不能近似为正态分布的后验分布需要特殊的方法。
&emsp;&emsp;&emsp;同样的参数也适用于Metropolis跳跃。在正态或近似正态的设置下，跳跃核理想情况下应该具有与目标坟墓相同的协方差结构，这个可以根据模式下的正态近似进行估计。

## （2）辅助变量
&emsp;&emsp;Gibbs采样器的计算通常可以通过添加辅助变量来简化或加速收敛，例如混合分布的指标。添加变量的想法也被称为数据增强，无论是对于Gibbs采样器还是EM算法，这通常是一个很有用的概念和计算工具。

### 例子：将t分布表示为正态分布的混合
&emsp;&emsp;辅助变量的一个简单但重要的例子是t分布，它可以表示为正态分布的混合。我们用参数$\mu$，$\sigma^2$的例子来说明，给定来自$t_v(\mu,\sigma^2)$分布的n个独立数据点，为了简单起见，我们假设v已知。我们还假设在$\mu$，$\log\mu$上有一个先验分布。每个数据点上的t似然等价于模型，
$$y_i\sim N(\mu,V_i)$$
$$V_i\sim Inv-\chi^2(\upsilon,\sigma^2)$$
&emsp;&emsp;其中$V_i$是不能直接观察到的辅助变量。如果我们使用联合后验分布$p(\mu,\sigma^2,V|y)$进行推导，然后只考虑$\mu$，$\sigma$模拟，这些将代表原始t模型下的后验分布。
&emsp;&emsp;在t模型中没有直接对参数$\mu$，$\sigma^2$进行采样，而是在增广模型中在V、$\mu$、$\sigma^2$上执行Gibbs采样器很简单：
1. **每个$V_i$的条件后验分布**。根据数据y和模型的其他参数，每个$V_i$都是一个有尺度的反$\chi^2$先验分布的正态方差参数，因此其后验分布也是反$\chi^2$。
$$V_i|\mu,\sigma^2,v,y\sim Inv-\chi^2(v+1,\frac{\upsilon\sigma^2+(y_i-\mu)^2}{\upsilon+1})$$
n个参数$V_i$在其条件后验分布上是独立的，我们可以通过从它们的有标度逆$\chi^2$分布中采样来直接应用Gibbs采样器。
2. **$\mu$的条件后验分布。**在模型数据y和模型其他参数的条件下，$\mu$的信息由n个数据点$y_i$提供，每个数据点都有自己的方差。结合y的均匀先验分布，$$\mu|\sigma^2,V,\upsilon,y\sim N(\frac{\sum_{i=1}^n\frac{1}{V_i}y_i}{\sum_{i=1}^{n}\frac{1}{V_i}},\frac{1}{\sum_{i=1}^n\frac{1}{V_i}})$$
3. **$\sigma^2$的条件后验分布。**在数据y和模型其他参数的条件下，关于$\sigma$的所有信息都来自方差$V_i$。条件后验分布为：$$\begin{aligned} p(\sigma^2|\mu,V,\upsilon,y)&\propto\sigma^{-2}\prod_{i=1}^n\sigma^{\upsilon}e^{-\upsilon\sigma^2/(2V_i)}\\
&=(\sigma^2)^{n\upsilon/2-1}exp(-\frac{\upsilon}{2}\sum_{i=1}^{n}\frac{1}{V_i}\sigma^2)\\
&\propto Gamma(\sigma^2|\frac{n\upsilon}{2},\frac{\upsilon}{2}\sum_{i=1}^{n}\frac{1}{V_i})
\end{aligned}$$
我们可以直接从中采样。

## （3）参数扩展
&emsp;&emsp;对于一些问题，由于那些不能简单地用线性变换来解决的参数之间的后验依赖性，Gibbs采样器可以缓慢收敛。矛盾的是，添加一个额外的参数——从而在更大的空间中执行随机游走，可以提高马尔科夫链模拟的收敛性。我们用上面的t例子来说明。
### 例子：拟合t模型（续）
&emsp;&emsp;在t模型的潜在参数形式中，如果$\sigma$的模拟接近0，收敛速度会很慢，因为条件分布将导致$V_i$的值在接近0时采样，然后$\sigma$的条件分布将接近0，以此类推。最终，模拟将会卡住，对于一些问题可能还会很慢。我们可以通过添加一个新的参数来修复问题，其唯一作用是允许Gibbs采样器向更多的方向移动，从而避免被卡住。扩展的模型是，
$$y_i\sim N(\mu,\alpha^2U_i)$$
$$U_i\sim Inv-\chi^2(\upsilon,\tau^2)$$
其中，$\alpha$>0可以看作是一个额外的标度参数。在这个新模型中，$\alpha^2U_i$起$V_i$的作用，$\alpha\tau$起$\sigma$的作用。参数$\alpha$本身没有意义，我们可以在对数尺度上分配一个非信息均匀的先验分布。

&emsp;&emsp;在扩展模型上的Gibbs采样器现在有四个步骤：
1. 对于每个$i,U_i$和之前的$V_i$一样更新：$$U_i|\alpha,\mu,\tau^2,\upsilon,y\sim Inv-\chi^2(\upsilon+1,\frac{\upsilon\tau^2+((y_i-\mu)/\alpha)^2}{\upsilon+1})$$
2. 均值$\mu$和之前一样更新：$$\mu|\alpha,\tau^2,U,\upsilon,y\sim N(\frac{\sum_{i=1}^{n}\frac{1}{\alpha^2U_i}y_i}{\sum_{i=1}^{n}\frac{1}{\alpha^2U_i}},\frac{1}{\sum_{i=1}^{n}\frac{1}{\alpha^2U_i}})$$
3. 方差参数$\tau^2$和之前的$\sigma^2$一样更新：$$\tau^2|\alpha,\mu,U,\upsilon,y\sim Gamma(\frac{n\upsilon}{2},\frac{\upsilon}{2}\sum_{i=1}^{n}\frac{1}{U_i})$$
4. 最后，我们必须更新$\alpha^2$，因为基于模型中的所有其他参数，它只是一个正态方差参数，所以这很容易：$$\alpha^2|\mu,\tau^2,U,\upsilon,y\sim Inv-\chi^2(n,\frac1n\sum_{i=1}^n\frac{(y_i-\mu)^2}{U_i})$$

&emsp;&emsp;该扩展模型中的参数$\alpha^2,U,\tau$没有被识别，因为数据没有提供足够的信息来估计它们。然后，只要我们查看$\mu,\sigma=\alpha\tau$和$V_i=\alpha^2U_i,i=1,...,n$的收敛性，整个模型就会被识别。或者，如果唯一的目标是对原始t模型进行推断，我们可以简单地从模拟中保存$\mu$和$\sigma$。
&emsp;&emsp;扩展参数化下的Gibbs采样器收敛更可靠，因为新的参数$\alpha$打破了$\tau$和$V_i$之间依赖性。

### Gibbs抽样器python实现
```py
import random
import math
import matplotlib.pyplot as plt
 
def xrange(x):
    return iter(range(x))
 
def p_ygivenx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))
 
def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))
 
N = 5000
K = 20
x_res = []
y_res = []
m1 = 10
m2 = -5
s1 = 5
s2 = 2
 
rho = 0.5
y = m2
 
for i in xrange(N):
    for j in xrange(K):
        x = p_xgiveny(y, m1, m2, s1, s2)
        y = p_ygivenx(x, m1, m2, s1, s2)
        x_res.append(x)
        y_res.append(y)
 
num_bins = 50
plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()
```

# 二、有效的Metropolis跳跃规则
&emsp;&emsp;对于任何给定的后验分布，Metropolis-Hastings算法都可以用无限多的方式实现。即使在重新参数化之后，在跳跃规则$J_t$中仍然有无数的选择。在许多具有共轭族的情况下，后验模拟可以完全地或部分地使用GIbbs采样器来执行，虽然编程很容易，但不总是有效的。对于非共轭模型，我们必须依赖于Metropolis-Hastings算法（要么在Gibbs采样器内，要么直接依赖多元后验分布）。然后就出现了跳跃规则的选择。

&emsp;&emsp;简单的跳跃规则主要有两类。第一个基本上是围绕参数空间的随机游走。这些跳跃规则通常是正态的跳跃核，其均值等于当前参数和方差的值。第二种方法构造近似于目标分布的建议分布（Gibbs采样器中子集的条件分布或联合后验分布）。在第二种情况下，目标是接收尽可能多的抽签，而Metropolis-Hastings的接受步骤主要用于纠正近似。每次改变一个参数没有什么自然的优势，只是在每一步评估部分后验密度的潜在的计算节省。

&emsp;&emsp;对于有效的跳跃规则，很难给出一般的建议，但对于随机行走的跳跃分布，在许多问题中已经得到了一些结果。假设有d个参数，并且$\theta=(\theta_1,...,\theta_d)$的后验分布，经过适当的变换，是有已知协方差矩阵$\Sigma$的多元正态分布。进一步假设我们将使用Metropolis算法进行绘制，其正态跳跃核以当前点为中心，与目标分布具有相同的形状，即$J(\theta^*|\theta^{t-1})=N(\theta^*|\theta^{t-1},c^2\Sigma)$。在这类跳跃规则中，最有效的是$c\approx2.4/\sqrt{d}$，其中有效性的定义是相对于后验分布中得到的独立样本。d维正态分布的最优Metropolis跳跃规则的有效性约为$0.3/d$（相比之下，如果d的后验分布独立，Gibbs采样器的效率将为1/d）。哪种算法对任何特定问题最好取决于每次迭代的计算时间，而每次迭代又取决于后验密度的条件独立性和共轭性质。

&emsp;&emsp;Metropolis算法也可以用被接受的跳跃的比例来描述。对于跳跃核与目标分布形状相同的多元正态随机游走分布，最优游走规则的一维接受率约为0.44，高维接受率下降到约0.23（约为d>5）。该结果提出了一种自适应仿真算法：
1. 用一个固定的算法开始并行模拟，如Gibbs采样器的一个版本，或采用正常随机游走跳跃规则形状的目标分布 的估计规则的Metropolis算法。
2. 经过多次模拟后，更新Metropolis规则如下：
(a) 调整跳跃分布的协方差，使其与模拟估计的后验协方差矩阵成正比。
(b) 如果模拟的接受率过高或过低，则增加或减少跳跃分布的尺度。目标是将跳跃规则引入近似最优值0.44（一维）或0.23（使用向量跳跃同时更新许多参数时）。

该算法可以通过各种方式进行改进，但即使是在它的简单形式中，我们也发现它对于d从1到50的一些问题的后验模拟很有用。当一个迭代模拟算法被“调优”时，即它运行时进行修改，必须注意避免收敛到错误的分布。如果更新规则依赖之前的模拟，那么转移概率比Metropolis-Hastings算法复杂得多，且一般不会收敛到目标分布。

&emsp;&emsp;为了安全，通常在两个阶段运行任意的自适应算法：第一，自适应阶段，其中算法的参数可以经常调整以提高效率；第二，固定阶段，适应算法运行足够长以近似收敛。在最终推断中，只使用了来自固定阶段的模拟。

# 三、Gibbs和Metropolis的进一步扩展
## （一）切片抽样
&emsp;&emsp;来自d维目标分布的$\theta$随机样本$p(\theta|y)$相当于来自分布下区域的随机样本。形式上，从$(\theta,u)$的d+1维分布中进行采样，其中对于任意的$\theta$，$p(\theta,u|y)\propto1$,$u\in[0,p(\theta|y)]$，否则为0。切片采样指迭代算法在这种均匀分布上的应用。实现一个有效的切片采样过程可能很复杂，但具有很大的普遍性，对于Gibbs采样结构中的一维条件分布的抽样特别有用。
## （二）在不同维度的空间中移动的可逆跳跃抽样
&emsp;&emsp;在多数情况下，希望进行跨维马尔科夫链模拟，其中参数空间的维数可以从一个迭代变化到下一个迭代。发生这种情况的一个例子是模型均匀，其中构建了单个马尔科夫链模拟，包括许多可信模拟之间的移动。这种马尔科夫链模拟的“参数空间”包括传统的参数和当前模型的标志。

&emsp;&emsp;在这种设置下，仍然可以使用可逆跳跃采样的方法使用Metropolis算法。我们使用对应于马尔科夫链在许多候选模型之间移动的情况的符号。设$M_k,k=1,...,K$，表示候选模型，$\theta_k$为维数为$d_k$的模型k的参数向量。可逆跳跃方法的一个关键方面是引入了额外的随机变量，是跨模型的参数空间维度能够匹配。具体来说，如果考虑从k到$k^*$的移动，则生成一个具有跳跃分布$J(u|k,k^*,\theta_k)$的随机变量u。定义一系列一一对应的确定函数，它们与$(\theta_{k^*},u^*)=g_{k,k^*}(\theta_k,u)$和$d_k+dim(u)=d_{k^*}+dim(u^*)$进行维数匹配。

&emsp;&emsp;我们给出了可逆跳跃算法，然后举一个粒子。对于一般描述，设$\pi_k$表示模型k，$p(\theta_k|M_k)$上的先验概率，模型k中参数的先验分布，$p(y|\theta_k,M_k)$表示模型k下参数的先验分布。可逆跳跃马尔科夫链在每次迭代中使用以下三个步骤从$p(k,\theta_k|y)$中生成样本：
1. 从状态$(k,\theta_k)$（即带有参数向量$\theta_k$的模型$M_k$）开始，提出了一种新的概率为$J_{k,k^*}$的模型$M_{k^*}$，并从建议密度$J(u|k,k^*,\theta_k)$生成一个增强的随机变量u。
2. 确定该模型的参数，$(\theta_{k^*},u^*)=g_{k,k^*}(\theta_k,u).$
3. 定义比率$$r=\frac{p(y|\theta_{k^*},M_{k^*})p(\theta_{k^*}|M_{k^*})\pi_{k^*}}{p(y|\theta_{k},M_k)p(\theta_k|M_k)\pi_k}\frac{J_{k^*,k}J(u^*|k^*,k,\theta_{k^*})}{J_{k,k^*}J(u|k,k^*,\theta_k)}|\frac{\nabla_{k,k^*}(\theta_k,u)}{\nabla(\theta_k,u)}|$$并且以最小概率(r,1)接受新模型。

所得到的后验图提供了关于每个模型的后验概率以及该模型下的参数的推断。

## （三）模拟回火和并行回火
&emsp;&emsp;多模态分布是马尔科夫链的特殊问题。目标是从整个后验分布中取样，这需要从每个具有显著后验概率的模态中采样。但马尔科夫链很容易长时间保持单一模式，主要发生在两个或更多模式被极低的后验密度区域隔开时，很难从一种模式移动到另一种模式。

&emsp;&emsp;模拟回火是提高该情况下马尔科夫链性能的一种策略。取$p(\theta|y)$作为目标密度。该算法适用于一组K+1维分布$p_k(\theta|y)$，k=0、1...其中$p_0(\theta|y)=p(\theta|y)$，$p_1、...、p_K$有相同形状的分布，但它提供了跨模式混合的机会，并且每个版本都有自己的采样器。通常，分布$p_k$不需要完全指定；用户只需要计算非归一化的密度函数$q_k$，其中$q_k(\theta)=p_k(\theta|y)$乘以一个常数，它可以依赖于y和k，但不依赖参数$\theta$。

&emsp;&emsp;非归一化密度$q_k$阶梯的一个选择是$$q_k(\theta)=p(\theta)^{1/T_k}$$
对于一组“温度”参数$T_k>0$。设置$T_k=1$降到原始密度，较大的$T_k$值产生较低的峰值。然后发展出一个单一的复合马尔科夫链，它在K+1维分布上随机移动，$t_0$设为1，从而使$q_0(\theta)\propto p(\theta|y)$。迭代t时复合马尔科夫链的状态用$(\theta_t,s_t)$表示，其中$s_t$是标识t时使用的分布的整数。复合马尔科夫链的每次迭代包括两个步骤：
1. 利用具有平稳分布$q_{s^t}$的马尔科夫链，选择新值$\theta^{t+1}$。
2. 利用概率$J_{s^t,j}$提出了一个从当前采样器$s^t$到备选采样器j的跳转。我们接受具有最小概率(r,1)的移动，其中$$r=\frac{c_jq_j(\theta^{t+1})J_{j,s^t}}{c_{s^t}q_{s^t}(\theta^{t+1})J_{s^t,j}}$$
常数$c_k,k=0,1,..,K$是自适应设置的，以近似由非归一化密度$q_k$定义的分布的归一化常数的倒数。然后，该链将在每个采样器中花费大约相同的时间。

&emsp;&emsp;在马尔科夫链结束时，仅适用从目标分布$q_0$模拟的$\theta$值来获得后验推断。

&emsp;&emsp;并行回火是上述算法的一种变体。每个链都可以自己移动，偶尔会有链之间的状态翻转，具有类似于模拟回火的Metropolis接受拒绝规则。收敛时，链0的模拟代表了来自目标分布的提取。

&emsp;&emsp;其他辅助变量方法已经开发出来，专门针对特定的多变量分布结构。

## （四）粒子过滤、加权和遗传算法
&emsp;&emsp;粒子过滤描述了一类涉及并行链的模拟算法，其中现有链被定期测试并允许死亡、存活或分裂，规则设置使得后验分布低概率区域的链更有可能死亡，而高概率区域的链更有可能分裂。

&emsp;&emsp;一个相关的想法是加权，在其中执行模拟，收敛到一个指定但错误的分布$g(\theta)$，然后由$p(\theta|y)/g(\theta)$进行加权。在更复杂的实现中，这种可以在整个模拟过程进行重新加权。有时从$p(\theta|y)$中采样可能很困难或昂贵，如果有可能，使用良好的近似g也会更快。加权可以通过使用死亡、存活、分裂概率中的权重与粒子滤波相结合。

&emsp;&emsp;遗传算法类似于粒子滤波，有多个可以存活、死亡的链，但随着更新算法本身可以改变和结合。

# 四、哈密顿蒙特卡洛
&emsp;&emsp;哈密顿蒙特卡洛（HMC）借鉴了物理学中的一个思想来抑制Metropolis-Hastings算法中的局部随机游走行为，从而允许它在目标分布中更快速地移动。对于目标空间中的每个分量$\theta_j$，HMC增加了一个“动量”变量$\varphi_j$。然后$\theta$和$\varphi$在一个新的Metropolis算法中一起更新，其中$\theta$的跳跃分布主要由$varphi$决定。HMC的每次迭代都经过几个步骤，在此期间，位置和动量基于模拟位置行为的规则演化，这些步骤可以通过$\theta$空间快速移动，甚至可以在参数空间中转弯，以保持轨迹的总“能量”。HMC也被成为混合蒙特卡洛，因为它结合了MCMC和确定性模拟的方法。

&emsp;&emsp;在HMC中，后验密度$p(\theta|y)$通过动量的独立分布$p(\phi)$增强，从而定义一个联合分布$p(\theta,\phi|y)=p(\phi)p(\theta|y)$。我们从联合分布进行模拟，但我们只关注$\theta$。

&emsp;&emsp;除了后验密度，HMC还需要对数-后验密度的梯度。实践中，梯度必须进行解析计算；数值微分需要大量函数计算才能有效。如果$\theta$有d个维度，这个梯度是$\frac{d\log{p(\theta|y)}}{d\theta}=(\frac{d\log{p(\theta|y)}}{d\theta_1},...,\frac{d\log{p(\theta|y)}}{d\theta_d})$。对于多数模型，向量很容易解析，然后编程。

## （一）动量分布$p(\phi)$
&emsp;&emsp;通常给定$\phi$一个多元正态分布，均值为0，协方差设为一个指定的“质量矩阵”M。为了简单起见，通常使用对角质量矩阵m。如此，$\phi$的分量是独立的，其中$\phi_j\sim N(0,M_{jj})$，j=1,...,d。对于M来说，利用后验分布的逆协方差矩阵$(var(\theta|y))^{-1}$是有用的，该算法在任何情况下都有效。

## （二）HMC迭代的三个步骤
&emsp;&emsp;HMC通过一系列的迭代进行，每个迭代有三部分：
1. 迭代首先通过从$\phi$的后验分布中随机抽取来更新$\phi$，与它的先验分布$\phi\sim N(0,M)$相同。
2. HMC迭代的主要部分是同时更新$(\theta,\phi)$，通过离散模拟以复杂但有效的方式进行。更新涉及L“跳跃步骤”，每个步骤都按因子$\epsilon$缩放。在跳跃步骤中，$\theta$和$\phi$都发生变化。L跳跃步骤如下：
重复以下步骤L次：
(a) 使用$\theta$的对数后验密度梯度（向量导数）进行$\phi$的半步：$$\phi\leftarrow\phi+\frac12\epsilon\frac{d\log{p(\theta|y)}}{d\theta}$$。
(b) 使用“动量”向量$\phi$更新“位置”向量$\theta$：$$\theta\leftarrow\theta+\epsilon M^{-1}\phi$$
同样，M是质量矩阵，是动量分布$p(\phi)$的协方差。如果M是对角阵，则上述步骤相当于缩放$\theta$更新每个维度。
(c) 再次使用$\theta$的梯度来半更新$\phi$：
$$\phi\leftarrow+\frac12\epsilon\frac{d\log{p(\theta|y)}}{d\theta}$$
除了第一步和最后一步，可以一起更新(c)和(a)。该步进从$phi$的半步开始，然后交替选择参数向量$\theta$和动量向量$\phi$的完整步骤，然后以$\phi$的半步结束。
3. 标记$\theta^{t-1}$、$\phi^{t-1}$为跳跃过程开始时的参数和动量向量的值，$\theta^*$、$\phi^*$为L步后的值。在接受-拒绝步骤中，计算$$r=\frac{p(\theta^*|y)p(\phi^*)}{p(\theta^{t-1}|y)p(\phi^{t-1})}$$
4. 令$$\theta^t=\begin{cases}\theta^*&最小概率(r,1)\\\theta^{t-1}&其他\end{cases}$$
严格地说，设置$\phi^t$是必要的，但我们不关心$\phi$，并且它在下一次迭代开始时立即更新，因此在接受-拒绝步骤之后不需要追踪它。

重复迭代，直到近似收敛，通过$\hat{R}$接近1来评估。

## （三）限制参数和后验密度为0的区域
&emsp;&emsp;HMC适用于所有正目标密度。如果在迭代的任何点，算法到达0后验密度点，停止步进并放弃，在当前$\theta$值上用另一个迭代。所得算法保持了平衡，并保持正面积。

&emsp;&emsp;另一种选择是“弹跳”，算法在每一步后再次检查密度是否为正，如果不是，则改变动量的符号，返回到它出现的方向。这再次保持了平衡，并比简单拒绝迭代更有效。

&emsp;&emsp;另一种处理有界参数的方法是通过转换，如将参数约束为正的对数或参数，或更复杂的参数约束集的联合转换。然后，计算出变换的雅可比矩阵，用它来确定新空间中的对数后验密度和梯度。

## （四）设置调优参数
&emsp;&emsp;HMC可以在三个地方进行调整：（i）动量变量$\phi$的概率分布；（ii）跳跃步骤的缩放因子$\epsilon$；（iii）每次迭代的跳跃步骤数L。

&emsp;&emsp;与Metropolis算法一样，这些调优参数可以提前设置，或随机改变，但在改变之前迭代给定的信息时必须小心。除某些特殊情况外，调优参数的自适应更新会改变算法，使其不再收敛与目标分布。所以设置参数时，从初始设置开始，运行HMC一段时间，然后后根据迭代重置参数，放弃预热的迭代。有必要时，可重复只保存最有一次设置的参数。

&emsp;&emsp;如何控制HMC的参数？首先将动量变量的尺度参数设为目标分布尺度的粗略估计。默认，使用单位矩阵。

&emsp;&emsp;将$\epsilon L$设为1。这大致将HMC算法校准为目标分布的“半径”。

&emsp;&emsp;最后，当HMC的接受率约为65%时是最优的。目前，如果平均接受概率不接近65%，则进行调整。如果较低，则跳跃。

## (五)  代码实现
```py
import numpy as np
import random
import scipy.stats as st
import matplotlib.pyplot as plt
 
 
def normal(x,mu,sigma):
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator
 
 
def neg_log_prob(x,mu,sigma):
    return -1*np.log(normal(x=x,mu=mu,sigma=sigma))
 
 
def HMC(mu=0.0,sigma=1.0,path_len=1,step_size=0.25,initial_position=0.0,epochs=1_000):
    # setup
    steps = int(path_len/step_size) # path_len and step_size are tricky parameters to tune...
    samples = [initial_position]
    momentum_dist = st.norm(0, 1) 
    # generate samples
    for e in range(epochs):
        q0 = np.copy(samples[-1])
        q1 = np.copy(q0)
        p0 = momentum_dist.rvs()        
        p1 = np.copy(p0) 
        dVdQ = -1*(q0-mu)/(sigma**2) # gradient of PDF wrt position (q0) aka potential energy wrt position
 
 
        # leapfrog integration begin
        for s in range(steps): 
            p1 += step_size*dVdQ/2 # as potential energy increases, kinetic energy decreases, half-step
            q1 += step_size*p1 # position increases as function of momentum 
            p1 += step_size*dVdQ/2 # second half-step "leapfrog" update to momentum    
        # leapfrog integration end        
        p1 = -1*p1 #flip momentum for reversibility     
 
 
        
        #metropolis acceptance
        q0_nlp = neg_log_prob(x=q0,mu=mu,sigma=sigma)
        q1_nlp = neg_log_prob(x=q1,mu=mu,sigma=sigma)        
 
 
        p0_nlp = neg_log_prob(x=p0,mu=0,sigma=1)
        p1_nlp = neg_log_prob(x=p1,mu=0,sigma=1)
        
        # Account for negatives AND log(probabiltiies)...
        target = q0_nlp - q1_nlp # P(q1)/P(q0)
        adjustment = p1_nlp - p0_nlp # P(p1)/P(p0)
        acceptance = target + adjustment 
        
        event = np.log(random.uniform(0,1))
        if event <= acceptance:
            samples.append(q1)
        else:
            samples.append(q0)
    
    return samples
        
import matplotlib.pyplot as plt
mu = 0
sigma = 1
trial = HMC(mu=mu,sigma=sigma,path_len=1.5,step_size=0.25)
 
 
lines = np.linspace(-6,6,10_000)
normal_curve = [normal(x=l,mu=mu,sigma=sigma) for l in lines]
 
 
plt.plot(lines,normal_curve)
plt.hist(trial,density=True,bins=20)
plt.show()
```
