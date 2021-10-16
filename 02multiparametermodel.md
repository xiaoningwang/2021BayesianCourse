# 贝叶斯多参数模型

贺淼蓉  2019302130037

## 1. 引入

顾名思义，在多参数模型中我们有超过一个的参数，这使得我们的模型有更广泛的应用。一个常见的例子就是均值和方差（或均值向量和协方差矩阵）均未知的（多元）正态分布。另一个常见的例子就是多项分布——如果对话题模型[topic model](https://en.wikipedia.org/wiki/Topic_model)的隐狄利克雷模型[latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)有所了解的话，对此应该不会陌生。

事实上，有些时候虽然我们使用多参数模型建模，但我们只关心其中某个参数的分布。此时我们将其他不感兴趣的参数成为“妨碍参数”(Nuisance parameter)。在我们得到参数的后验分布后，我们可能需要通过积分或其他手段求出感兴趣的参数的分布，也就是边缘分布。

## 2. 一元正态分布

一元正态分布$y_i\lvert \mu,\sigma^2$的概率密度函数为
$$
f(y_i\lvert\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}exp\{-\frac{1}{\sqrt{2\pi\sigma^2}}\}
$$

多样本下有
$$
f(y_i\lvert\mu,\sigma^2)\propto\sigma^{-n}exp\{-\frac{1}{\sqrt{2\sigma^2}}\sum_{i=1}^{n}({y_i-\mu})^2\}
$$
我们知道，先验选取的不同会导致我们的后验分布的结果不同。因此这里我们分为无信息先验和共轭先验两小节来讨论这一问题。

### 2.1 无信息先验

我们选取Jeffrey先验$p(\mu,\sigma^2)\propto(\sigma^2)^{-1}$，或者说$p(\mu,\log\sigma)\propto1$。注意，此时的先验也是和位置族、尺度族参数的结论是一致的。

于是我们立刻可以得到后验分布
$$
p(\mu,\sigma^2\lvert y)\propto\sigma^{-n-2}exp\{-\frac{1}{2\sigma^2}[(n-1)s^2+n({y_i-\mu})^2]\}
$$
其中$s^2=\frac{1}{n-1}\sum_{i=1}^{n}({y_i-\bar{y}})^2$为样本方差。

#### 2.1.1 $\mu$的条件后验分布

有了后验分布的形式，我们可以很轻松得写出$\mu$的条件后验分布
$$
p(\mu\lvert\sigma^2,y)\propto exp\{-\frac{n({\bar{y}-\mu})^2}{2\sigma^2}\}
$$

#### 2.1.2 $\sigma^2$的后验分布

即$\mu\lvert\sigma^2,y\sim\mathcal{N}(\bar{y},\sigma^2/n)$现在我们来计算$\sigma^2$的后验分布，直接积分有
$$
\begin{align}p(\sigma^2\lvert y) &\propto\int \sigma^{-n-2}\exp\left\{-\frac{1}{2\sigma^2}\left[(n-1)s^2+n(\bar{y}-\mu)^2\right]\right\}\,\mathrm{d}\mu\\ &\propto \sigma^{-n-2}\exp\left\{-\frac{(n-1)s^2}{2\sigma^2}\right\}\sqrt{2\pi\sigma^2/n}\\ &\propto (\sigma^2)^{-(n+1)/2}\exp\left\{-\frac{(n-1)s^2}{2\sigma^2}\right\} \end{align}
$$
即$\sigma^2\lvert y\sim \mathrm{Inv-}\chi^2(n-1,s^2)$，或写为${Inv-Gamma}((n-1)/2,(n-1)s^2/2)$。

当然，另一种做法是使用下面的公式
$$
p(\mu,\sigma^2)=p(\sigma^2)p(\mu\lvert\sigma^2)
$$
在给定$y$下上式依然成立，即
$$
p(\mu,\sigma^2\lvert y)=p(\sigma^2\lvert y)p(\mu\lvert\sigma^2,y)
$$
于是可以省去积分的操作，直接
$$
p(\sigma^2\lvert y)=\frac{p(\mu,\sigma^2\lvert y)}{p(\mu\lvert\sigma^2,y)}\
$$


得到完全相同的结果。

#### 2.1.3 $\mu$的后验分布

使用直接积分，先做变换
$$
\begin{align}z&=\frac{1}{2\sigma^2}\left[(n-1)s^2+n(\bar{y}-\mu)^2\right]=A/\sigma^2,\,\frac{\mathrm{d}\sigma^2}{\mathrm{d}z}=-\frac{A}{z^2}\\ p(\mu\lvert y) &\propto\int \sigma^{-n-2}\exp\left\{-\frac{1}{2\sigma^2}\left[(n-1)s^2+n(\bar{y}-\mu)^2\right]\right\}\,\mathrm{d}\mu\\ &\propto \int (A/z)^{-(n+2)/2}\exp\{-z\}\frac{A}{z^2}\,\mathrm{d}z\\ &\propto A^{-n/2}\int z^{(n-2)/2}\exp\{-z\}\,\mathrm{d}z\\ &\propto A^{-n/2}\\ &\propto \left(1+\frac{n(\mu-\bar{y})^2}{(n-1)s^2}\right)^{-n/2} \end{align}
$$


事实上这是$t$分布的形式，即$\mu\sim t_{n-1}(\bar{y},s^2/n)$，或等价的$\frac{\mu-\bar{y}}{s/\sqrt{n}}\sim t_{n-1}$，这和传统方法给出的结论也是相同的。

建议读者尝试使用相除的方法求取$\mu$的后验分布，因为许多初学者第一次并不能准确地求得。其陷阱在于归一化常数里是可能含有不能丢掉的信息的——本题为例的话就是分母的归一化因子中含有$\mu$。因此如果不小心的一路使用$\propto$可能会导致出现错误的结果。

### 2.2 共轭先验

我们可以求得共轭先验为$\mathrm{N-Inv-}\chi^2(\mu_0,\sigma_0^2/\kappa_0; \nu_0, \sigma_0^2)$。该分布的表达式为
$$
p(\mu,\sigma^2)\propto p(\sigma^2)p(\mu\lvert\sigma^2)
$$
其中$\sigma^2\sim\mathrm{Inv-}\chi^2(\nu_0,\sigma_0^2),\,\mu\lvert\sigma^2\sim\mathcal{N}(\mu_0,\sigma^2/\kappa_0)$，即
$$
p(\mu,\sigma^2)\propto\sigma^{-1}(\sigma^2)^{-(\nu_0/2+1)}\exp\left\{-\frac{1}{2\sigma^2}(\nu_0\sigma_0^2+\kappa_0(\mu_0-\mu)^2)\right\}
$$
在此先验下，可以算得其后验分布为
$$
p(\mu,\sigma^2\lvert y)\propto \sigma^{-1}(\sigma^2)^{-((\nu_0+n)/2+1)}\exp\left\{-\frac{1}{2\sigma^2}(\nu_0\sigma_0^2+\kappa_0(\mu_0-\mu)^2+(n-1)s^2+n(\bar{y}-\mu)^2)\right\}
$$
整理为先验的形式为$\mathrm{N-Inv-}\chi^2(\mu_n,\sigma_n^2/\kappa_n; \nu_n, \sigma_n^2)$，其中
$$
\begin{align} \mu_n&=\frac{\kappa_0}{\kappa_0+n}\mu_0+\frac{n}{\kappa_0+n}\bar{y}\\ \kappa_n&=\kappa_0+n\\ \nu_n&=\nu_0+n\\ \nu_n\sigma^2&=\nu_0\sigma_0^2+(n-1)s^2+\frac{\kappa_0n}{\kappa_0+n}(\bar{y}-\mu_0)^2\\ \end{align}
$$
由此我们可以立刻得到$\sigma^2\lvert y\sim\mathrm{Inv-}\chi^2(\nu_n,\sigma_n^2),\,\mu\lvert\sigma^2,y\sim\mathcal{N}(\mu_n,\sigma^2/\kappa_n)$

使用类似的算法，我们可以计算出$\mu$的后验分布
$$
p(\mu\lvert y)\propto\left(1+\frac{\kappa_n(\mu-\mu_n)^2}{\nu_n\sigma_n^2}\right)^{-(\nu_n+1)/2}=t_{\nu_n}(\mu\lvert\mu_n,\sigma_n^2/\kappa_n)
$$

## 3. 多项分布

多项分布的概率密度函数为
$$
p(y\lvert\theta)\propto\prod_{i=1}^{k}\theta_i^{y_i},\,\sum_{i=1}^{k}\theta_i=1
$$
其共轭先验为狄利克雷分布[Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)。其形式为
$$
p(\theta\lvert\alpha)\propto\prod_{i=1}^{k}\theta_i^{\alpha_i-1}
$$
其后验分布为
$$
p(\theta\lvert y)\propto\prod_{i=1}^{k}\theta_i^{\alpha_i+y_i-1}
$$
也就是说，参数从$\alpha$更新为$\alpha+y$。

## 4. 多元正态分布

多元正态分布的概率密度函数为
$$
p(y\lvert \mu,\Sigma)\propto \lvert\Sigma\rvert^{-n/2}\exp\left\{-\frac{1}{2}\sum_{i=1}^{n}(y_i-\mu)^\intercal\Sigma^{-1}(y_i-\mu)\right\}=\lvert\Sigma\rvert^{-n/2}\exp\left\{-\frac{1}{2}\mathrm{tr}(\Sigma^{-1}S_0)\right\}
$$
其中$S_0=\sum_{i=1}^{n}(y_i-\mu)^\intercal(y_i-\mu)$，这里用到了迹[trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra))的性质$\mathrm{tr}(ABC)=\mathrm{tr}(BCA)$，而且且$1\times1$的矩阵的迹就是这个值。

### 4.1 共轭先验

#### 4.1.1 给定$\Sigma$的情形

此时，我们可以使用共轭先验$\mu\lvert\Sigma\sim\mathcal{N}(\mu_0,\Lambda_0)$以得到后验分布
$$
p(\mu\lvert y,\Sigma)\propto\exp\left\{-\frac{1}{2}\left( (\mu-\mu_0)^\intercal\Lambda_0^{-1}(\mu-\mu_0)+\sum_{i=1}^{n}(y_i-\mu)^\intercal\Sigma^{-1}(y_i-\mu) \right) \right\}\propto \exp\left\{-\frac{1}{2}\left( (\mu-\mu_n)^\intercal\Lambda_n^{-1}(\mu-\mu_n)\right)\right\}
$$
其中
$$
\begin{align} \Lambda_n^{-1}&=\Lambda_0^{-1}+n\Sigma^{-1}\\ \mu_n&=(\Lambda_0^{-1}+n\Sigma^{-1})^{-1}(\Lambda_0^{-1}\mu_0+n\Sigma^{-1}\bar{y})\\ \end{align}
$$
即$\mu\lvert y,\Sigma\sim\mathcal{N}(\mu_n,\Lambda_n)$，从这个形式也能看出“加权平均”的影子。

#### 4.1.2 $\Sigma$的先验分布

回忆一元模型下我们使用卡方分布来刻画方差。若$z_1,\ldots,z_\nu\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\tau^2)$，我们有
$$
S=\sum_{i=1}^{\nu}z_i^2\sim\tau^2\chi_\nu^2
$$
这里我们使用方差的推广——威沙尔分布[Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution)来刻画协方差阵。若$z_1,\ldots,z_\nu\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\Lambda)$，我们有
$$
\Sigma=\sum_{i=1}^{\nu}z_iz_i^\intercal\sim\mathrm{Wishart}_\nu(\Lambda)
$$
仿照一元正态的模型，我们使用$\mathrm{N-Inv-Wishart}(\mu_0,\Lambda_0^2/\kappa_0; \nu_0, \Lambda_0)$作为先验，同样是分两步走：$\Lambda\sim\mathrm{Inv-Wishart}_{\nu_0}(\Lambda_0^{-1}),\,\mu\lvert\Lambda\sim\mathcal{N}(\mu_0,\Lambda/\kappa_0)$。

我们可以求得其后验分布为$\mathrm{N-Inv-Wishart}(\mu_n,\Lambda_n^2/\kappa_n; \nu_n, \Lambda_n)$，其中
$$
\begin{align} \mu_n&=\frac{\kappa_0}{\kappa_0+n}\mu_0+\frac{n}{\kappa_0+n}\bar{y}\\ \kappa_n&=\kappa_0+n\\ \nu_n&=\nu_0+n\\ \Lambda_n&=\Lambda_0+\sum_{i=1}^{n}(y_i-\bar{y})(y_i-\bar{y})^\intercal+\frac{\kappa_0n}{\kappa_0+n}(\bar{y}-\mu_0)(\bar{y}-\mu_0)^\intercal\\ \end{align}
$$
此时计算得条件后验、边缘后验如下
$$
\begin{align} \mu\lvert y&\sim t_{\nu_n-d+1}(\mu_n,\Lambda_n/(\kappa_n(\nu_n-d+1)))\\ \Sigma\lvert y&\sim \mathrm{Inv-Wishart}_{\nu_n}(\Lambda_n^{-1})\\ \mu\lvert \Sigma,y&\sim\mathcal{N}(\mu_n,\Sigma/\kappa_n)\\ \end{align}
$$
这里展示了最难算的$\Lambda_n$​的计算过程
$$
p(\mu,\Sigma\lvert y)\propto \lvert\Sigma\rvert^{-(n+\nu_0+d)/2+1}\exp\left\{-\frac{1}{2}\mathrm{tr}(\Sigma^{-1}S_0)-\frac{\kappa_0}{2}(\mu-\mu_0)^\intercal\Sigma^{-1}(\mu-\mu_0)-\frac{1}{2}\mathrm{tr}(\Lambda_0\Sigma^{-1})\right\}
$$
我们依次考察指数中的三个项。
$$
\begin{align} \mathrm{tr}(\Sigma^{-1}S_0) &=\sum_{i=1}^{n}(y_i-\mu)^\intercal\Sigma^{-1}(y_i-\mu)\\ &=\sum_{i=1}^{n}(y_i-\bar{y}+\bar{y}-\mu)^\intercal\Sigma^{-1}(y_i-\bar{y}+\bar{y}-\mu)\\ &=\sum_{i=1}^{n}(y_i-\bar{y})^\intercal\Sigma^{-1}(y_i-\bar{y})-2\sum_{i=1}^{n}(y_i-\bar{y})^\intercal\Sigma^{-1}(\bar{y}-\mu)+\sum_{i=1}^{n}(\bar{y}-\mu)^\intercal\Sigma^{-1}(\bar{y}-\mu)\\ &=\sum_{i=1}^{n}(y_i-\bar{y})^\intercal\Sigma^{-1}(y_i-\bar{y})+n(\bar{y}-\mu)^\intercal\Sigma^{-1}(\bar{y}-\mu)\\ &=S+n(\bar{y}^\intercal\Sigma^{-1}\bar{y}-2\bar{y}^{\intercal}\Sigma^{-1}(\bar{y}-\mu)+\mu^{\intercal}\Sigma^{-1}\mu)\\ \end{align}
$$

$$
\kappa_0(\mu-\mu_0)^\intercal\Sigma^{-1}(\mu-\mu_0)=\kappa_0(\mu_0^\intercal\Sigma^{-1}\mu_0-2\mu_0\Sigma^{-1}(\bar{y}-\mu)+\mu^{\intercal}\Sigma^{-1}\mu+\mu^\intercal\Sigma^{-1}\mu)
$$



于是乎，指数项(的-2倍)为
$$
\begin{align} &\mathrm{tr}(\Sigma^{-1}S_0)+\kappa_0(\mu-\mu_0)^\intercal\Sigma^{-1}(\mu-\mu_0)+\mathrm{tr}(\Lambda_0\Sigma^{-1})\\ =&(n+\kappa_0)\mu^\intercal\Sigma^{-1}\mu-2(n\bar{y}+\kappa_0\mu_0)^{\intercal}\Sigma^{-1}\mu+(n\bar{y}^\intercal\Sigma^{-1}\bar{y}+\kappa_0\mu_0^\intercal\Sigma^{-1}\mu_0)+S+\mathrm{tr}(\Lambda_0\Sigma^{-1})\\ =&(n+\kappa_0)(\mu-\mu_n)^{\intercal}\Sigma^{-1}(\mu-\mu_n)-(n+\kappa_0)\mu_n^\intercal\Sigma^{-1}\mu_n+(n\bar{y}^\intercal\Sigma^{-1}\bar{y}+\kappa_0\mu_0^\intercal\Sigma^{-1}\mu_0)+S+\mathrm{tr}(\Lambda_0\Sigma^{-1})\\ =&(n+\kappa_0)(\mu-\mu_n)^{\intercal}\Sigma^{-1}(\mu-\mu_n)+\mathrm{tr}\left(\left[-\frac{(n\bar{y}+\kappa_0\mu_0)(n\bar{y}+\kappa_0\mu_0)^\intercal}{n+\kappa_0}+n\bar{y}\bar{y}\intercal+\kappa_0\mu_0\mu_0^\intercal+S+\Lambda_0\right]\Sigma^{-1}\right)\\ \end{align}
$$


其中第二项即为$\Lambda_n\Sigma^{-1}$。

### 4.2 无信息先验

使用$\Sigma\sim\mathrm{Inv-Wishart}_{d+1}(I)$，此时其Jeffreys无信息先验为
$$
p(\mu,\Sigma)\propto\lvert\Sigma\rvert^{-(d+1)/2}
$$


转载自：[http://blog.vicayang.cc/Note-Multi-Parameter-Model/](https://blog.vicayang.cc/Note-Multi-Parameter-Model/)

