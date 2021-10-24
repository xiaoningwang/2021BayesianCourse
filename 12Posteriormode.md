# 后验分布的模与边缘后验分布的模

## 后验分布的模(posterior mode)

### 引入

 #### 1. 后验分布的模
$$
\mathop{\arg\max}\limits_{\theta}p(\theta\vert x)
$$


在贝叶斯统计学中，“最大后验概率估计”是后验概率分布的众数。   

 #### 2. 我们为什么关注后验分布的模？

​		在贝叶斯计算中，我们寻找模不是为了它们本身，而是作为一种映射后验密度的方法。它在统计中经常被用于构造后验分布的近似分布以及被用作点估计,有时为罚似然估计的形式（其中先验密度的对数被视为一个惩罚函数)。



在后续内容中将介绍如何寻找后验分布的模和边缘后验分布的模、如果用后验分布的模来总结整个分布应该如何选择合适的先验分布以及基于模的正态近似和混合近似等问题，并辅以实例帮助大家进行理解。


​    

### 寻找后验分布的模-最大后验概率估计（MAP）

#### 1. 定义

​		我们需要根据观察数据$x$估计没有观察到的总体参数$\theta$，让$f$作为$x$的采样分布，这样$f(x\vert\theta)$就是总体参数为$\theta$时$x$的概率，函数$f(x\vert\theta)$为似然函数。假设$\theta$存在一个先验分布$f(\theta)$  ，$\theta$的后验分布就是：
$$
f(\theta\vert x)=\frac {f(x\vert\theta)f(\theta)}{f(x)}\tag{2.1}
$$
​		最大后验估计方法于是估计$\theta$为这个随机变量的后验分布的模：
$$
\hat \theta_{MAP}(x)=\mathop{\arg\max}\limits_{\theta}\frac {f(x\vert\theta)f(\theta)}{f(x)}\tag{2.1}=\mathop{\arg\max}\limits_{\theta}f(x\vert\theta)f(\theta)
$$
​		因为后验分布的分母与$\theta$无关，所以在优化过程中不起作用。

#### 2. 方法

​	    原则上，解决优化问题的数值方法都可以应用于寻找后验分布密度的模。如果存在多个模 ，应该尝试找到所有的模。通常先搜索一个模，如果它在实质意义上看起来不合理，则继续在参数空间中搜索其他模。有时为了找到所有的局部模，或者确保已经找到的模是唯一重要的模，必须从不同的起点运行寻找模的算法。在此节，我们将介绍在统计问题中最常用的两种简单方法。



​	**2.1 条件最大化**

   ​    最简单的方法是条件最大化（也称为逐步上升）。

   ​    此方法简单地从目标分布的某处开始,例如,将参数设置为粗略的估计,然后改变θ的一组向量,其他向量和之前的值保持一致,每一步增加对数后验密度。假设后验密度是有界的，这些步骤最终将收敛到一个局部的模。对于许多标准统计模型，参数的条件分布都有一个简单的分析形式，并且易于最大化。在这种情况下,应用条件最大化算法很简单:一次只对一组参数的密度最大化,迭代直到步长足够小,从而达到近似收敛。对数线性模型的迭代比例拟合方法就是是条件最大化的一个例子。

   ​	若要寻找多个模，需要从整个参数空间的各点开始运行条件最大化的。基于对参数的粗略估计和关于参数的合理边界特定问题的知识，应该能得到一系列合理的起点。

   

​	**2. 2 牛顿法**

- 牛顿法，也被称为牛顿-拉夫森算法，是一种基于对数后验密度的二次泰勒级数近似的迭代方法。
$$
L(\theta) =logp(\theta\vert y)\tag{2.1}
$$

​		该式中的后验密度可以是非归一化的。因为牛顿法只使用$L(\theta)$的导数,并且p中的任何乘法常数都是L中的一个加法常数。当数据点的数量大于参数的数量时，二次近似通常是相当准确的。从确定函数$L'(\theta)$和$L''(\theta)$开始,分别为后验密度对数的导数向量和二阶导数矩阵开始。它们可以用解析或数值的方法来确定。



- 导数的数值计算

​		如果对数后验密度的一阶和二阶导数难以确定解析解，则可以用有限差分法在数值上近似他们。$L'$的每一个分量在任何给定值$\theta=(\theta_1,...,\theta_d)$,都可以被式（2.3）数值地估计。
$$
L'_i(\theta)=\frac {dL}{d\theta_i}\approx\frac {L(\theta+\delta_ie_i\vert y)-L(\theta-\delta_ie_i\vert y)}{2\delta_i}\tag{2.3}
$$
​		其中$\delta_i$是一个较小的值,$e_i$是$\theta$的第i个分量对应的单位向量。$\delta_i$的值是根据问题的规模来选择的,通常,像0.0001对近似导数值已经足够低,对避免计算机上的圆角误差也足够高。通过再次差分,对$\theta$处的二阶导数矩阵进行数值估计,对于每个i，j：
$$
\begin{align}
L''_{ij}(\theta)=\frac {d^2L}{d\theta_id\theta_j}&=\frac d{d\theta_j}(\frac {dL}{d\theta^i})\\
      &\approx \frac {L'_i(L(\theta+\delta_ie_i\vert y)-L(\theta-\delta_ie_i\vert y))}{2\delta_j}\\
      &\approx \frac{[L(\theta+\delta_ie_i+ \delta_je_j)-L(\theta-\delta_ie_i+ \delta_je_j)-L(\theta+\delta_ie_i-\delta_je_j)-L(\theta-\delta_ie_i-\delta_je_j)]}{(4\delta_i\delta_j)}
\end{align}\tag{2.4}
$$




- 寻找模的算法： 

​		(1) 选择一个起值, $\theta^0$。

​		(2) 对于 t = 1、2、3、…

​	 	 (a)计算$L'(\theta^{t-1})$和$L''(\theta^{t-1})$。牛顿法在t次的步长是基于以$\theta^{t-1}$为中心的$L(\theta)$的二次近似。

​		  (b)设置新的迭代,$\theta^t$,以使二次近似最大化；
$$
\theta^t = \theta^{t-1}-[L''(\theta^{t-1})]^{-1}L'(\theta^{t-1})\tag{2.5}
$$
​		初始值$\theta^0$很重要。该算法不能保证从所有的起始值都收敛, 特别是在$-L''$不是正定的区域。起始值可以从粗略的参数估计中获得，或使用条件最大化方法生成牛顿法的起始值。



- 牛顿法的优点

   在二次近似准确处，一旦迭代接近解，收敛速度很快。如果迭代不收敛，它们通常会快速地向参数空间的边缘移动，下一步可能会再次尝试一个新的起点。
   
   
   
   **2.3 拟牛顿和共轭梯度方法**
   
    $-L''$的计算和存储在牛顿法中开销很大。拟牛顿方法,如Broyden-Fletcher-Goldfarb-Shanno(BFGS)方法,迭代使用梯度信息形成$-L''$的近似。


​     	共轭梯度方法只使用梯度信息，但是，其使用共轭方向公式确定优化方向，而不是最速下降法。共轭梯度可能比牛顿和准牛顿方法需要更多的迭代，但每次迭代使用的计算量和存储空间更少。



### 合适的先验函数

#### 1. 为什么要选择不同的先验分布？

- 后验的模是对称后验分布的一个很好的点总结。然而，如果后验是不对称的，则该模可能是一个较差的点估计。如果我们计划通过后验的模来总结后验分布，需选择适当的先验分布。

  

​		下面我们一起来看两个例子：

- 例子1

​		如图3.1所示,考虑8所学校例子中的组级尺度参数的后验分布。这个(边缘)后验分布的模是$\tau=0$,对应所有8所学校对大学招生考试指导的效用模型都相同。这一结论与数据一致，但从实际考虑，我们并不认为真正的变化完全是零。八所学校的辅导项目不同，所以效果应该变化 ，并且最好是少量变化。如图1所示，如果我们选择用它的模来总结这个分布，我们将处于$\hat{\tau}=0$的不理想位置，因为这是一个在参数空间边界的估计。

![图3.1](https://github.com/Dodongxi/2021BayesianCourse/blob/33605b6c638c32e29de3b261e8c588c468ae9865/figure/wjq%20%E5%9B%BE1.png)



- 例子2

​		边际似然的模为0的问题不仅出现在8所学校的例子中。
$$
y_i\sim N(\theta_j,1) ,\quad \quad j=(1,...,J) \tag{3.1}
$$
​		其中$J$=10

​		为了简单起见,认为$\theta_j$服从以零为中心的正态分布:
$$
\theta_j \sim N(0,r^2)\tag{3.2}
$$
​		在模拟中，假定$\tau=0.5$。

​		对这样一个简单的一维层次模型,我们创建了1000个模拟数据集$y$，进行1000次模拟,对均匀先验分布下的边缘后验的模进行采样分布。对于每个模型,我们确定边际似然和它获得其最大值的位置。

​		由图3.2和3.3可知，在几乎一半的模拟中， 边际似然在$\tau=0$时最大,但此处噪音多到数据无法得到精确估计，似然函数对$\tau$不能提供很多信息。

![图3.2](https://github.com/Dodongxi/2021BayesianCourse/blob/33605b6c638c32e29de3b261e8c588c468ae9865/figure/wjq%20%E5%9B%BE2.png)

![图3.3](https://github.com/Dodongxi/2021BayesianCourse/blob/33605b6c638c32e29de3b261e8c588c468ae9865/figure/wjq%20%E5%9B%BE3.png)

​		由此，就引出一个问题，选择什么合非信息先验分布,$p(\tau)$,可以避免8个学校中类似的问题?



#### 2. 组级方差参数避免为0的先验分布

​		针对上一小节提出的问题，我们可以得出以下答案：

​		首先,在$\tau=0$处,$p(\tau)$必须为零。

​		正随机变量的概率模型具有这一特性,包括对数正态($\log\tau \sim N(\mu_t,\sigma_t^2)$),以及逆伽马($\tau^2 \sim Inv-gamma(\alpha_\tau,\beta_\tau)$)。不幸的是，这两类先验分布在接近0时都关闭得太快。对数正态和逆伽马都有有效的下界，低于这个下界先验密度下降,迅速地关闭一些接近于零的$\tau$范围。如果这些模型上的尺度参数被设置为足够模糊，这个下界可以变得非常低，但然后先验就达到峰值。因此,这些模型没有合理的默认设置;我们必须要么选择一个模糊的先验来排除值接近0的$\tau$,要么选择高信息的分布。

​		相反,我们更喜欢一个先验的模型,如$\tau \sim Gamma(2,\frac 2A)$。当$\tau=0$时,密度从0开始,然后线性增加,最终在$\tau$的最大值下轻轻弯曲到零。在$\tau=0$的线性行为确保,后验分布将与数据保持一致,这一属性在对数正态或逆伽马先验分布不成立。

​		同样,这个先验分布的目的是在$\tau$的后验分布被其模总结的结果较好,就像在使用分层模型的统计计算中通常出现的情况那样。如果我们计划使用后验模拟 ，我们通常不会看到伽马先验分布的任何优势，而是使用均匀或半柯西作为默认选择。



#### 3.相关参数避免位于边界的先验分布

​		我们在估计相关参数时会遇到很多困难，下面是一个简单的例子。

​		在每一组$j=1,...,J$,我们假设有一个线性模型:
$$
y_{ij} \sim N(\theta_{j1}+\theta_{j2}x_i)\qquad i=1,...,n_j\tag{3.3}
$$
​		我们从标准正态分布中抽取独立的$x_i$，并令$n_j=5$，$J=10$。

​		j中每组的两个回归参数被建模为一个正态分布的随机抽取:
$$
\begin{pmatrix} \theta_{j1} \\ \theta_{j2} \end{pmatrix} \sim N\begin{pmatrix}\begin{pmatrix}0\\0\end{pmatrix},\begin{pmatrix}\tau_1^2,\rho\tau_1\tau_2\\\rho\tau_1\tau_2,\tau^2\end{pmatrix}\end{pmatrix}\tag{3.4}
$$
​		在前面的例子中,我们对线性参数θ取平均值,并使用边际似然,为
$$
p(y\vert \tau_1,\tau_2,\rho)=\prod_{j=1}^JN(\hat {\theta_j}\vert0,V_j+T)
$$
​		其中，$\hat {\theta_j}$和$V_j$由最小二乘估计得出,相应的协方差矩阵是从对第j组的数据，
$$
T=\begin{pmatrix}\tau_1^2,\rho\tau_1\tau_2\\\rho\tau_1\tau_2,\tau_2^2\end{pmatrix}\tag{3.5}
$$
​		对于本例,我们假设方差参数的真实值为$\tau_1=\tau_2=0.5$且$\rho=0$。为了获得稳定且远离边界的$\rho$估计值,将真实值设置为0，但这样，我们仍然有很多问题。

​		与前一样，我们模拟数据1000次并计算边际似然。因为我们关注$\rho$,所以关注$\rho$的最大边际似然估计$(\tau_1,\tau_2,\rho)$,我们也关注$\rho$的轮廓似然,也就是函数$L_{profile}(p\vert y)=max_{\tau_1\,tau_2}p(y\vert \tau_1\tau_2,\rho)$。对于每个模拟,我们都计算作为$\rho$的一个函数的轮廓似然，并使用数值优化。

​		优化很简单，因为边际似然函数可以写成封闭的形式。$\rho$的边缘后验密度，将需要更多的精力去计算,但会产生类似的结果。

​		图3.4和图3.5显示，在1000次模拟中,群级相关性的最大边际似然估计超过10%的情况是在边界上$(\hat \rho=\pm1)$上,而$\rho$的边缘轮廓似然通常不提供很大的信息。在全贝叶斯的情况下,我们将对ρ进行平均;而在一个罚似然框架中,我们想要一个更稳定的点估计。

![图3.4](https://github.com/Dodongxi/2021BayesianCourse/blob/4de340b2b97591bcf8b4e271dda08b29f128a269/figure/wjq_%E5%9B%BE4.png)

![图3.5](https://github.com/Dodongxi/2021BayesianCourse/blob/4de340b2b97591bcf8b4e271dda08b29f128a269/figure/wjq_%E5%9B%BE5.png)

​		如果计划是通过ρ的后验的模来总结推断,我们将用$p(\rho)\propto( (1-\rho)(1+\rho))$替换$U(-1,1)$，它等价于一个$Beta(2,2)$。由此后验分布的模不会是-1或者1。$\rho$的先验密度在边界附近是线性的,因此不会与任何可能性相矛盾。

#### 4. 协方差矩阵避免退化的先验分布

​			我们希望基于模的协方差矩阵的点估计或计算近似都是非退化的，即具有正的方差参数和正定的相关矩阵。

​		可以通过选择一个当协方差矩阵退化时先验密度趋于零的先验分布来确保后验的模有这一性质。对于一般的$d \times d$协方差矩阵,我们选择$wishart(d+3,AI)$先验密度,它为零,但在边界上有一个正的常数导数。与前面一样,我们可以根据问题背景将A设置为一个很大的值。估计的协方差矩阵是正定的，并且没有排除边界附近的估计，只要有似然支持的估计。在一个的大样本极限下,协方差矩阵上的$wishart(d+3,AI)$先验分布对应于每个$d$上独立的$Gamma(2,\sigma)$先验分布。使用$\sigma\to 0$的特征值,因此可以看作是我们对上述方差参数的默认模型的泛化。在二维空间中,极限一个多元模型在$A\to \infty$时对应于先验分布$p(\rho)\propto( (1-\rho)(1+\rho))$。

​		同样,如果要通过后验的模对协方差矩阵进行总结或近似推断,我们认为这个默认的Wishart先验分布是一个非信息的选择。对于全贝叶斯推断,不需要选择在边界上达到零的先验分布。

### 基于模的正态和相关混合近似

#### 1. 基于模对后验的简单正态近似

- 如果后验分布$p(\theta \vert y)$是单峰的,近似对称,则可以方便地用正态分布来近似它;也就是说,后验密度的对数用$\theta$的二次函数来近似。在这里，我们考虑了以后验的模为中心的对数后验密度的二次近似（通常容易使用现成的优化例程来计算）。

​		以后验的模为中心的$\log p(\theta\vert y)$的泰勒级数展开式,$\hat \theta$($\theta$是一个向量且在参数空间的内部),给出
$$
\log p(\theta\vert y)=\log p(\hat \theta\vert y)+\frac 12 (\theta-\hat\theta)^T[\frac {d^2}{d\theta^2}\log p(\theta\vert y)]_{\theta=\hat\theta}(\theta-\hat\theta)\tag{4.1}
$$
​		展开的线性项为零，因为对数后验密度在其模处的导数为零。该式(4.1)作为θ的函数,它的第一项是一个常数,而第二项与一个正态密度的对数成正比,得到近似,
$$
p(\theta\vert y)\sim N(\hat\theta,[I(\hat\theta)]^{-1})\tag{4.2}
$$
​		其中,$I(\theta)$是被观测的信息,如果模$\hat\theta$，在参数空间内部，则$I(\hat\theta)$的矩阵是正定的。



- 似然函数无界的问题

​		如果似然函数是无界的，那么在参数空间内可能没有后验的模 ，这时正态近似无效。一般来说，这个问题在实践中很少出现，因为无界似然的极点对应于模型中 的不现实的条件。这个问题可以通过限制在一组合理的分布来解决。当方差分量出现在零附近时，可以通过使用在边界处下降到零的先验分布解决这个问题。



- 正态近似的一个简单例子

​		$y_1,...,y_n$是$N(\mu,\sigma^2)$分布的独立观测，并且我们假定$(\mu,\log\sigma)$有一个均匀先验密度。我们建立了$(\mu,\log\sigma)$后验分布的正态近似，它可以将$\sigma$限制为正值。为了构造近似，我们需要对数后验密度的二阶导数，
$$
\log p(\mu,\log\sigma\vert y)=constant-n\log\sigma-\frac1{2\sigma^2}((n-1)s^2+n(\bar y-\mu))\tag{4.3}
$$
​		一阶导数是：
$$
\frac d{d\mu}\log p(\mu,\log\sigma\vert y)=\frac{n(\bar y-\mu)}{\sigma^2}\tag{4.4}
$$

$$
\frac d{d\log\sigma}\log p(\mu,log\sigma\vert y)=-n+\frac{(n-1)s^2+n(\bar y-\mu)}{\sigma^2}\tag{4.5}
$$



​		后验分布的模已经被求出：
$$
(\hat \mu,\log\hat\sigma)=(\bar y,log(\sqrt{\frac{n-1}n}s))\tag{4.6}
$$
​		对数后验密度的二阶导数是：
$$
\frac {d^2}{d\mu^2}\log p(\mu,\log\sigma\vert y)=-\frac{n}{\sigma^2}\tag{4.7}
$$

$$
\frac d{d\log\sigma d\mu}\log p(\mu,\log\sigma\vert y)=-2n+\frac{(\bar y-\mu)}{\sigma^2}\tag{4.8}
$$

$$
\frac {d^2}{d\log\sigma^2}logp(\mu,\log\sigma\vert y)=-\frac 2{\sigma^2}((n-1)s^2+n(\bar y-\mu)^2)\tag{4.9}
$$

​		模处的二阶导数矩阵为$\begin{pmatrix}{\frac {-n}{\hat\sigma^2 },0\\0,-2n}\end{pmatrix}$。后验分布可以被近似为：
$$
\log p(\mu,\log\sigma\vert y) \approx N\begin{pmatrix}\begin{pmatrix}\mu\\\log\sigma \end{pmatrix}\begin{pmatrix}\bar y\\\log\hat\sigma\end{pmatrix}\begin{pmatrix}{\frac {\hat\sigma^2 }n,0\\0,\frac 1{2n}}\end{pmatrix}\end{pmatrix}\tag{4.10}
$$
- 渐近正态性和一致性

​		我们可以在（4.1）式的基础上对大样本下后验分布及其模的性质进行一个简单探索。

 		在某些正则性条件下(特别是似然是$\theta$的连续函数并且和$\theta_0$不在参数空间的边界上),如$n\to\infty$，$\theta$的后验分布接近正态性并有均值$\theta_0$和方差$J(\theta_0)^{-1}$．这个结果可以用以后验的模为中心的对数后验密度的泰勒级数展开(4.1)来理解。初步结果表明,后验的模和$\theta_0$是一致的,因此随着$n\to\infty$,后验分布$p(\theta \vert y)$的质量集中在$\theta_0$越来越小的邻域,并且$\vert \hat\theta-\theta_0 \vert$距离接近零。

​		此外,我们还可以在(4.1)中改写二次项的系数:
$$
[\frac {d^2}{d\theta^2} \log p(\theta \vert y)]_{\theta=\hat\theta}=[\frac {d^2}{d\theta^2}\log p(\theta)]_{\theta=\hat\theta}+\sum_{i=1}^n[\frac {d^2}{d\theta^2}\log p( y_i\vert\theta)]_{\theta=\hat\theta}\tag{4.11}
$$
​		作为$\theta$的函数,只要是$\hat\theta$接近$\theta_0$,(我们现在假设对于一些$\theta_0$,$f(y)=p(y\vert\theta_0)$，是一个常数加上n项的和,每个项的期望值都在$y_i$，$p(y\vert\theta_0)$的真实采样分布下,约为$-J(\theta_0)$。因此,对于大样本情况，对数后验密度的曲率可以用费舍尔信息来近似。

​		在大样本的极限下,在一个特定的模型族的背景下,后验的模$\hat\theta$接近$\theta_0$,而曲率(观测信息或泰勒展开中第二项系数的负值)接近$nJ(\hat\theta)$或$J(\theta_0)$。此外,由于$n\to\infty$,似然主导了先验分布,所以我们可以单独使用似然来得到正态近似的模和曲率。



#### 2. 基于模曲率的多元正态密度拟合

​		一旦找到了模，我们可以构造一个基于多元正态分布的近似。

​		我们首先考虑单个模$\hat\theta$，我们用一个正态函数去拟合对数后验密度的在$\hat\theta$处的二阶导数：
$$
p_{approx} (\theta)= N(\theta \vert \hat\theta,V_\theta)\tag{4.12}
$$


​		方差矩阵是对数后验密度的在模处曲率的倒数,$V=[-\frac {d^2\log p(\theta\vert y)}{d\theta^2}\vert_{ \theta=\hat\theta_0}]^{-1}$。

​		这个二阶导数可以求解。在拟合正态密度之前,需要适当地变换参数,通常使用对数和逻辑变换，使它们能有一个大致对称分布。



#### 3. 多峰密度的混合近似

​		现在假设我们在后验密度中找到了K个模。后验分布可以用K个多元正态分布的混合分布来近似,每个分布都有自己的模$\hat\theta_k$,方差矩阵$V_{\theta k}$,和相对质量$\omega_k$。即,目标密度$p(\theta \vert y)$可以近似为：
$$
p_{approx} (\theta) \propto\sum_{k=1}^K\omega_kV(\theta\vert \hat\theta_k,V_{\theta k})\tag{4.13}
$$
​		多元正态混合分布的$\omega_k$的第$k$个向量都可以被后验密度估计$p(\hat\theta_k \vert y)$，或者未标准化的后验密度$q(\hat\theta_k \vert y)$，如果模分布较广，并且正态近似适用于每个模，那么我们可以得到：
$$
\omega_k=q(\hat\theta_k \vert y)[V_{\theta k}]^{1/2}\tag{4.14}
$$
​		导出混合正态近似:
$$
p_{approx} (\theta) \propto\sum_{k=1}^Kq(\hat\theta_k \vert y)exp(-\frac 12(\theta-\hat \theta_K)^TV_{\theta k}^{-1}(\theta-\hat \theta_K))\tag{4.15}
$$



#### 4. 使用多元t分布近似

​		对于一个广泛的分布,我们可以用自由度较小$\nu$的一个多元t分布来替换每个正态密度。相应的近似是一个具有泛函形式的混合密度函数，
$$
p_{approx} (\theta) \propto\sum_{k=1}^Kq(\hat\theta_k \vert y)(\nu +2(\theta-\hat \theta_K)^TV_{\theta k}^{-1}(\theta-\hat \theta_K))^{(d+\nu)/2}\tag{4.16}
$$




​		其是中$d$是$\theta$的维数。

​		我们还可以使用很多策略来进一步改进近似分布,比如变分贝叶斯等。



## 后验边缘分布的模(marginal posterior mode)

### 引入

#### 1. 为什么要寻找后验边缘分布的模？

​		在参数数量较多的问题中，联合分布的正态近似通常失效，联合的模通常也没有帮助 。然而,在参数子集的边缘后验的模上进行近似通常是有用的。我们可以到一个参数子集的后验边缘分布密度的模。然后在第一个子集的条件下，分析剩余的参数。



### EM算法寻找后验边缘分布的模

#### 1. EM算法

​		EM是一种解决存在隐含变量优化问题的有效方法,是一种确定边缘后验密度$p(\varphi \vert y)$的模的迭代方法。

​		“EM”主要包括两个交替的步骤:确定缺失值所需函数(充分统计)的期望,以及最大化得到的后验密度来估计参数。

​		EM具有广泛的应用性.一些模型很难直接最大化$p(\varphi \vert y)$,但却易于计算$p(\gamma\vert \varphi,y)$和$p(\varphi\vert\gamma ,y)$。此外许多模型,包括混合模型和一些层次模型,可以重新表示为增强参数空间上的分布,在增强参数空间中添加的参数γ可以被认为是缺失的数据。



#### 2. EM算法和广义EM算法

​		EM确定了边缘后验分布的模,$p(\varphi \vert y)$,并对参数$\gamma$进行平均。每次迭代的EM算法增加对数后验密度的值，直到收敛。
$$
\log p(\varphi \vert y)=\log p(\gamma,\varphi\vert y)-\log p(\gamma\vert \varphi,y)\tag{5.1}
$$
​		取双边的期望,将$gamma$作为一个随机变量,服从分布$p(\gamma\vert \varphi^{old},y)$,其中$\varphi^{old}$是当前的猜测。上述方程的左侧不依赖于$\varphi$,因此对$\varphi$求平均得：
$$
\log p(\varphi \vert y)=E_{old}(\log p(\gamma,\varphi\vert y))-E_{old}(\log p(\gamma\vert \varphi,y))\tag{5.1}
$$
​		$E_{old}$是$gamma$在分布$p(\gamma\vert \varphi^{old},y)$的平均值。（5.1）右侧最后一项的$E_{old}(\log p(\gamma\vert \varphi,y))$,在$\varphi=\varphi^{old}$时最大化。另一项,期望的对数联合后验密度,$E_{old}(\ logp(\gamma,\varphi\vert y))$,在计算中被重复使用。

$$
E_{old}(\log p(\gamma,\varphi\vert y))=\int (\log p(\gamma,\varphi\vert y))p(\gamma \vert \varphi^{old},y)d\gamma\tag{5.2}
$$
​		这个表达式为$Q(\varphi\vert\varphi^{old})$。

​		现在考虑任一$E_{old}(\ logp(\gamma,\varphi^{new}\vert y))\gt E_{old}(\log p(\gamma\vert \varphi^{old},y)$的$\varphi^{new}$的值。如果我们用$\varphi^{new}$替换了$\varphi^{old}$,我们增加了式(5.1)右边的第一项,总量也增加:$\log p(\varphi^{new}\vert y)\gt \log p(\varphi^{old}\vert y)$。

​		这一想法引发了广义EM (GEM)算法:在每次迭代中,确定$E_{old}(\ logp(\gamma,\varphi\vert y))$,被视为$\varphi$的一个函数,并更新$\varphi$为一个新的值以增加整个函数的值。EM算法是选择$\varphi$的新值来最大化$E_{old}(\ logp(\gamma,\varphi\vert y))$,而不仅仅是增加它。EM和GEM都在每次迭代时增加边缘后验密度$p(\varphi \vert y)$。

​		由于边缘后验密度$p(\varphi \vert y)$在EM算法的每一步都会增加,而且由于$Q$函数在每一步都最大化,所以除了在某些特殊情况下,他们收敛到后验密度的局部模。因为GEM算法不是在每一步都最大化,所以它不一定收敛到局部模。EM算法收敛到局部模的速率取决于$\varphi$在联合密度$p(\gamma ,\varphi \vert y)$中的比例,这是边缘密度$p(\varphi\vert y)$中缺失的。如果缺失信息的比例很大，收敛速度可能很慢。



#### 3. EM算法的实现

- 步骤

​		(1)从一个粗略参数估计,$\varphi^0$开始。

​		(2)对于$t=1,2,...$:

​			(a)E步:确定对数后验密度函数的期望
$$
E_{old}(\ logp(\gamma,\varphi\vert y))=\int (\log p(\gamma,\varphi\vert y))p(\gamma \vert \varphi^{old},y)d\gamma\tag{5.3}
$$
​			其中$\varphi^{old}=\varphi^{t-1}$。

​			(b) M步:$\varphi$的值是最大化$E_{old}(\log p(\varphi,\gamma\vert y)) $的值的$\varphi^{t}$。对于GEM算法,它只要求$E_{old}(\log p(\varphi,\gamma\vert y)) $会增加,但不一定会最大化。

​		正如我们所看到的,边缘后验密度$p(\varphi\vert y)$在EM算法的每一步都会增加,因此,除了在某些特殊情况外,该算法收敛到后验密度的局部模。

- 找多个模

​		用EM搜索多个模的一种简单方法是在整个参数空间选择许多点开始迭代。如果我们有几种模态，我们可以使用正态近似来大致比较它们的相对质量（如前所示）。



#### 4. EM算法的一些扩展

​		如SEM、SECM等，增加了算法可以应用的问题范围,还可以加快收敛速度，可以用来寻找边缘后验分布的模。



 #### 5. 用EM寻找后验边缘分布模的实例

​		假设我们在秤上称一个物体$n$次，称重 $y_1,...,y_n$,服从$N(\mu,\sigma^2)$且相互独立，其中 $\mu$ 是物体的真实重量。 为简单起见，我们假设 $N(\mu_0,\tau_0^2)$ 是$\mu$上的先验分布（ $\mu_0$和
$\tau_0$已知）和 $\log \sigma$有标准非信息的均匀先验分布； 它们部分形成了共轭联合先验分布。

​		因为模型不是完全共轭的，所以的$(\mu,\sigma)$联合后验分布没有标准形式，边缘后验分布密度没有封闭形式的表达式。 然而，我们可以使用 EM 算法找到 µ$\mu$的后验边缘分布的模，对 $\sigma$求平均值； 即，$(\mu,\sigma)$对应于$(\varphi,\gamma)$。

​		联合对数后验密度。 联合对数后验密度是：
$$
\log p(\mu,\sigma \vert y)=-\frac 1{2\tau_0^2}(\mu-\mu_0)^2-(n+1)\log \sigma-\frac 1{2\sigma^2}\sum_{i=1}^n(y_i-\mu)^2+constant\tag{5.4}
$$
​		忽略与$\mu$ 或 $\sigma^2$无关的项。

​		E-步骤。 对于 EM 算法的 E 步，我们必须确定(5.4）的期望，对 $\sigma$求平均值并以当前的猜测为条件，$\mu^{old}$，$y$：
$$
\begin{align}
\log p(\mu,\sigma \vert y)=&-\frac 1{2\tau_0^2}(\mu-\mu_0)^2-(n+1)E_{old}(\log \sigma)\\
&-\frac12E_{old}(\frac 1{\sigma^{2}})\sum_{i=1}^n(y_i-\mu)^2+constant
\end{align}\tag{5.5}
$$
​		我们现在必须评估 $E_{old}(\log \sigma)$ 和 $E_{old}(\frac 1{\sigma^2})$。 实际上，我们只需要评估后一个表达式，因为前一个表达式与 (5.5) 中的 $\mu$无关，因此不会影响 M 步。 表达式 $E_{old}(\frac 1{\sigma^2})$可以通过观察估计，给定 $\mu$，和$\sigma^2$ 的后验分布是对于已知的正态分布均值和未知方差，按$Inv-\chi^2$缩放：
$$
\sigma^2\vert\mu,y \sim Inv-\chi^2(n,\frac1n\sum_{i=1}n(y_i-\mu)^2)\tag{5.6}
$$
​		那么 $\sigma^2$ 的条件后验分布是一个缩放的$\chi^2$，并且
$$
E_{old}(\frac 1{\sigma^2})=E(\frac1{\sigma^2}\vert u^{old},y)=(\frac1n\sum _{i=1}^n(y_i-\mu^{old})^2)^{-1}\sum_{i=1}^n(y_i-\mu)^2=const\tag{5.7}
$$
​		我们可以把（5.5）重新写为：
$$
E_old\log p (\mu,\sigma\vert y)=-\frac1{2\tau_0^2}(\mu-\mu_0)^2-\frac12(\frac1n\sum_{i=1}^n(y_i-\mu^{old})^2)^{-1}\sum _{i=1}n(y_i-\mu)^2+const\tag{5.8}
$$
​		M步。 对于 M 步，我们必须找到使上述表达式最大化的 $\mu$。对于这个问题，任务很简单，因为（5.8）具有正规的形式对数后验密度，具有先验分布 $\mu \sim N(\mu_0,\tau_0^2)$ 和 $n$个数据点$y_i$,每个方差为 $\frac1n\sum _{i=1}^n(h_i-\mu^{old})^2$
M步是通过等效后验密度的模来实现的，即
$$
\mu^{new}=\frac{ \frac {1}{\tau_0^2}\mu_0+\frac n{\frac1n\sum _{i=1}^n(y_i-\mu^{old})^2}\bar y}{\frac1{\tau_0^2}+\frac n{\frac1n\sum _{i=1}^n}(y_i-\mu^{old})^2}\tag{5.8}
$$
​		如果我们迭代这个计算，$\mu$ 会收敛到 $p(\mu\vert y)$ 的边缘的模。



## 参考文献

[1]Andrew Gelman, John B. Carlin, Hal S. Stern, Donald B. Rubin, A. Gelman.Bayesian Data Analysis[M].Chapman & Hall,1995:06-01.
