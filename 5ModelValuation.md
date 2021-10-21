# 第七章 模型评估 **Chapter 7 model Valuation*



## 目录 Content

[TOC]

 

##   基本概念 **Basic Conceptions*

### **前言*

​        *在前一章中，作者讨论了检查模型拟合的差异度量的到数据。在本学期所学的机器学习课程中，清华大学出版社出版的由周志华编写的机器学习中的第二章也涉及到大量的模型评价的内容。其实模型评价的本质是评价模型的“泛化”能力，即学习新数据时出现的误差、方差、偏差的大小。随着模型复杂度的增加，模型的泛化误差、方差、偏差分别占据测试误差的主导地位。下图是误差“窘境”的泛化误差、方差、偏差的关系示意图。由下图可知公式：*
$$
{E}_{D}\left[(f(\boldsymbol{x} ; D)-\bar{f}(\boldsymbol{x}))^{2}\right]+E_{D}\left[(\bar{f}(\boldsymbol{x})-y)^{2}\right]+{E}_{D}\left[\left(y_{D}-y\right)^{2}\right]
$$
<img src="C:\Users\len\Pictures\Saved Pictures\7d1f1f282aa47261eda7cedae38df70e (2).jpg" alt="center"  />

<center/>图7.1模型评估指标函数图像<center>


> 图像来源于[(40条消息) 模型评估_温染的笔记-CSDN博客_模型评估](https://blog.csdn.net/weixin_43378396/article/details/90707493?ops_request_misc=%7B%22request%5Fid%22%3A%22163452653716780264025861%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=163452653716780264025861&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-90707493.first_rank_v2_pc_rank_v29&utm_term=模型评估&spm=1018.2226.3001.4187)

```python
# 进行预测
x=testX_min_max
y=testY_min_max
result=predict=a*np.power(x,7)+b*np.power(x,6)+c*np.power(x,5)+d*np.power(x,4)+e*np.power(x,3)+f*np.power(x,2)+g*x+h
# 计算预测误差
err=np.mean((result-y)*(result-y))
# 打印误差
print('err=%.3f' % err)
# 绘制拟合结果
plt.scatter(x,y)
plt.scatter(testX_min_max,result)
```

> 一个通过计算模型误差对机器学习模型进行评价的方法

*在本章中，我们寻求的不是检查模型（model checking），而是在模型之间作比较然后探索改进方向。即使所有能被我们选择的模型都不匹配，有了在比较模型过程中的这些数据，我们就可以评估其预测准确性并考虑下一步如何提升。本章关注的是预测模型的估计准确性，纠正评估模型对数据预测的固有偏差用来适应它。*





### **一个实例：一个预测总统选举的例子*

​        我们将使用简单的线性回归作为示例。 图 7.2 显示了一个快速总结过去近几十年的总统选举季的近几年经济情况与选举情况。 它基于政治科学家创建的“面包与和平”模型道格拉斯希布斯仅根据经济增长预测选举。

<img src="C:\Users\len\Pictures\Saved Pictures\微信截图_20211018123038.png" style="zoom: 50%;" />

<center/>图7.2 Douglas Hibbs’s ‘bread and peace’ model of voting and the economy.

> 图片来源于Bayesian Data Analysis Third Edition Chapter 7

​      

​        在上图所示的实例中，作者用了一种简单的线性回归模型（A simple linear regression model)作为模型。一个更好的预测模型是需要更多的先验信息的，在这个模型中，仅用自身所有的数据就获得相当高的预测准确度。它基于政治科学家创建的“面包与和平”模型 道格拉斯希布斯仅根据经济增长预测选举。



​      在这个模型里仅用 $$x$$(社会经济状况) 预测 $$y$$ (投票份额)，用了一个线性模型：$y~\sim \mathrm{N}\left(a+b x, \sigma^{2}\right)$$ 和一个无先验信息分布$$p(a, b, \log \sigma) \propto 1$    尽管这两个数据集是时间序列数据，在这里我们还是把他们视为一个简单的线性回归问题。

具有共轭先验的线性回归的后验分布是正态逆$$\chi^{2}$$，然后我们可以将后验分布分解为：
$$
\\p\left(a, b, \sigma^{2} \mid y\right)=p\left(\sigma^{2} \mid y\right) p\left(a, b \mid \sigma^{2}, y\right)\\ \\
方差参数的边际后验分布为:\sigma^{2} \mid y \sim \operatorname{Inv}-\chi^{2}\left(n-J, s^{2}\right)
$$
​    

​     其中$$X$$是预测变量的 $$n \times j$$ 矩阵，在这种情况下是 $$15 \times 2$$ 矩阵，其第一列是一列 1，第二列是经济绩效数字的向量 $$x$$。数学表达式为
$$
\\s^{2}=\frac{1}{n-J}(y-X \hat{\beta})^{T}(y-X \hat{\beta})\\
$$
​     系数向量的条件后验分布, $$\beta=(a, b)$$，可得：
$$
\\\beta \mid \sigma^{2}, y \sim \mathrm{N}\left(\hat{\beta}, V_{\beta} \sigma^{2}\right)\\\begin{aligned}
\hat{\beta} &=\left(X^{T} X\right)^{-1} X^{T} y \\V_{\beta} &=\left(X^{T} X\right)^{-1}
\end{aligned}\\
$$
​      

​    最后可得：$$s=3.6, \hat{\beta}=(45.9,3.2), \text  V_{\beta}=\left(\begin{array}{cc}
0.21 & -0.07 \\
-0.07 & 0.04
\end{array}\right)$$ 我们最后可知模型的准确率。



## 7.1用于预测准确率的方法**Measures of predictive accuracy*

​        评估模型的一种方法是通过其预测的准确性，有时我们在评估预测时也关心这种准确性。 预测准确性的价值在于比较不同的模型，而不是模型本身。我们首先考虑定义模型预测的准确性或误差的不同方法，然后讨论从数据中估计预测准确度或误差的方法。

​        预测准确度的度量是专门为应用定制的，且尽可能正确地衡量预测未来的收益（或成本）数据与模型。 特定于应用程序的措施的示例是分类准确性和成本。这其中大概包含了三种方法：

***1、预测点估计或点预测（predictive point estimation or point forecasting）***

​          在点预测（预测点估计或点预测）中，报告单个值作为对未知未来观察的预测。 点预测的预测准确性的度量称为评分函数。 我们以平方误差为例，因为它是预测文献中最常见的评分函数。模型在点预测中对新数据的拟合可以通过均方误差总结为：
$$
\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mathrm{E}\left(y_{i} \mid \theta\right)\right)^{2}

$$


***2、概率预测（probabilistic prediction）***

​        关于 $$\tilde{y}$$的推论，考虑到 $$\tilde{y}$$的不确定性。 概率预测的准确性的度量称为评分规则。示例包括二项、对数和0-1分数。 好的评分规则可分为特定背景下的和局部的：特定背景下的评分规则下的分数能够让决策者坚持或改变自己的研究方法或研究思路，而局部性包含了对某些 $$y$$ 的预测可能增加误判的可能性。 可以证明对数分数是唯一的（直到仿射变换）局部和适当的评分规则，它通常用于评估概率预测。



***3、对数预测密度或对数似然（Log predictive density or log-likelihood）***

​        预测的对数分数是对数预测密度 $$\log p(y|\theta)$$，如果模型是具有恒定方差的正态分布。 对数预测密度有时也称为对数似然。 对数预测密度在统计模型比较中有着重要的作用，因为它与 Kullback-Leibler 信息度量有关。 在大样本量的限制下，具有最低 Kullback-Leibler 信息的模型将具有最高的后验概率。 因此，使用预期对数预测密度作为整体模型拟合的度量似乎是合理的。 因其通用性，我们在本章中使用对数预测密度来衡量预测精度。

​       先验与估计参数有关，但与评估模型的准确性无关。先验密度与计算预测精度无关。 在评估模型时，预测准确性不是唯一的问题，即使在预测准确性的范围内，先验也是相关的，因为它会影响关于$$\theta$$的推论，从而影响任何涉及 $$p(y|\theta)$$ 的计算。





### 7.1.1单个数据点的预测准确性 **Predictive accuracy for a single data point*

​        模型拟合的理想度量是它的样本外预测性能从真正的数据生成过程（外部验证）产生的新数据。 我们标记 f作为真实模型，y 作为观察数据（因此，数据集 y 的单一实现来自分布 f(y))，以及作为未来数据或替代数据集的$$\tilde{y_i}$$看到了。 使用对数分数对新数据点$\tilde{y_i}$的样本外预测拟合是 然后:

$$
\log p_{\text {post }}\left(\tilde{y}_{i}\right)=\log \mathrm{E}_{\text {post }}\left(p\left(\tilde{y}_{i} \mid \theta\right)\right)=\log \int p\left(\tilde{y}_{i} \mid \theta\right) p_{\text {post }}(\theta) d \theta\ \ \ \ \ \ (7.1.1)
$$

​        在上面的表达式中，$$p_{post}(\tilde{y_i})$$是由后验诱导的$$\tilde y$$的预测密度分布 $$p_{post}(\theta)$$。 我们在这里引入了符号 $$p_{post}$$ 来表示后验分布，因为我们的表达式很快就会变得更加复杂方便避免明确显示我们对观察到的推断的条件数据y。 更一般地，我们使用 $$P_{post}$$和 $$E_{post}$$来表示任何概率或期望对 θ 的后验分布求平均值。







### 7.1.2平均未来数据的分布 * *Averaging over the distribution of future data*

​        然后我们必须采取进一步的措施。 未来数据 $$\tilde y_{i}$$本身是未知的，因此我们定义了预期的样本外对数预测密度:
$$
elpd=新数据点的预期对数预测密度
=\mathrm{E}_{f}\left(\log p_{\text {post }}\left(\tilde{y}_{i}\right)\right)=\int\left(\log p_{\text {post }}\left(\tilde{y}_{i}\right)\right) f\left(\tilde{y}_{i}\right) d \tilde{y}\ \ \ \ (7.1.2)
$$
​        在任何应用程序中，我们都会有一些$$p_{post}$$，但我们通常不知道数据分布 f. 一种估计预期样本外对数预测密度的自然方法
将插入 f 的估计值，但这往往意味着过拟合。 现在我们考虑贝叶斯模型中预测准确度的估计语境。 为了与给定的数据集保持可比性，可以定义一种预测的度量一次取一个数据点的准确度，公示如下：
$$
elppd = 新数据集的预期对数逐点预测密度=
\sum_{i=1}^{n} \mathrm{E}_{f}\left(\log p_{\text {post }}\left(\tilde{y}_{i}\right)\right) \ \ \ \ (7.1.3)
$$
​        上式必须基于某种商定的将数据 y 划分为单独的数据来定义点 $$y_i$$ 。 使用逐点测量而不是使用关节的优点后验预测分布，$$P_{post}(y_i)$$ 与逐点计算有关到交叉验证，它允许一些相当通用的方法来逼近 样本外数据进行样本拟合。这在给定点估计的情况下考虑预测准确性有时很有用$$\tilde{\theta}(y)$$ 因此我们可以得出预期的$$\log$$函数曲线，给定$$\hat{\theta}$$：$$ \quad E_{f}(\log p(\tilde{y} \mid \hat{\theta}))$$​,对于给定参数的具有独立数据的模型，给定点估计的联合或逐点预测之间没有区别，比如说：
$$
p(\tilde{y} \mid \hat{\theta})=\prod_{i=1}^{n} p\left(\tilde{y}_{i} \mid \hat{\theta}\right)\ \ \ (7.1.4)
$$





### *7.1.3评估拟合模型的预测准确性* **Evaluating predictive accuracy for a fitted model*

​       实际上参数 θ 是未知的，所以我们无法知道对数预测密度$$\log p(y \mid \theta)$$。 综上，我们用后验分布：$$p_{\text {post }}(\theta)=p(\theta \mid y)$$​，并将拟合模型的预测精度总结为：
$$
\begin{aligned}
\mathrm{lppd} &=\log \text { 逐点预测密度  } \\
&=\log \prod_{i=1}^{n} p_{\text {post }}\left(y_{i}\right)=\sum_{i=1}^{n} \log \int p\left(y_{i} \mid \theta\right) p_{\text {post }}(\theta) d \theta
\end{aligned}\ \ \ \ \ \ (7.1.5)
$$

为了在实践中预测密度，我们用如下这一函数：
$$
\begin{aligned}
\text { computed lppd } &=\text { computed log pointwise predictive density } \\
&=\sum_{i=1}^{n} \log \left(\frac{1}{S} \sum_{s=1}^{S} p\left(y_{i} \mid \theta^{s}\right)\right)
\end{aligned} \ \ \ \ (7.1.6)
$$




### 7.1.4定义似然量和预测量的选择 **Choices in defining the likelihood and predictive quantities*

​          正如在分层建模中众所周知的那样，将先验分布与似然分开的线有些随意，并且与假设复制中数据在哪些方面将发生变化的问题有关。 在具有直接参数 $$\alpha_{1}, \ldots, \alpha_{J}$$ 和超参数 φ 的分层模型中，分解为 :
$$
p(\alpha, \phi \mid y) \propto p(\phi) \prod_{j=1}^{J} p\left(\alpha_{j} \mid \phi\right) p\left(y_{j} \mid \alpha_{j}\right) \ \ \ \ (7.1.7)
$$
​          我们可以想象在现有组中复制新数据的可能分布为$$\left.p(y \mid \phi)=\int p\left(\alpha_{J+1} \mid \phi\right) p\left(y \mid \alpha_{J+1}\right) d \alpha_{J+1}\right)$$ 无论哪种情况，我们都可以轻松计算 观测数据y的后验预测密度： 

​          **1、当预测$$ \tilde{y}|\alpha_{j}$$（即来自现有组的新数据）时，我们为每个后验模拟$$\alpha_{j}^{s}$$计算 $$p\left(y \mid \alpha_{j}^{s}\right)$$，然后取平均值，如（7.1.6）所示。 **

​          **2、当我们预测$$\tilde{y}|\alpha_{J+1}$$时（这是一个新数据集中的数据）我们从$$p\left(\alpha_{J+1} \mid \phi^{s}\right)$$抽$$\alpha_{J+1}^{s}$$去计算$$p\left(y \mid \alpha_{J+1}^{s}\right)$$**

​      

   类似地，在混合模型中，我们可以考虑以混合指标为条件的复制，或重绘混合指标的复制。即使在最简单的实验中也会出现类似的选择。 例如，在模型$$y_{1}, \ldots, y_{n} \sim \mathrm{N}\left(\mu, \sigma^{2}\right)$$中我们可以选择假设样本量是设计固定的 （让 $$n$$ 未建模）或将其视为随机变量并在假设复制 $$\tilde{n}$$。 

​         我们不会被预测分布的非唯一性所困扰。 就像后验预测检查一样，不同的分布对应于后验推理的不同潜在用途。 给定某些特定数据，模型可能会在某些情况下准确预测新数据，但在其他情况下则不然。

## 



## 7.2信息标准和交叉验证 (Information criteria and cross-validation)

​          由于历史原因，预测准确性的度量被称为信息标准，通常根据偏差。 点估计 $$\tilde{\theta}$$ 和后验分布 $$p_{post}(y)$$ 拟合数据 y，样本外预测的准确度通常低于样本内预测准确度所暗示的准确度。 换句话说，在预期中，拟合模型对未来数据预测的准确度通常会低于同一模型对观察数据预测的准确度,即使拟合的模型族恰好包括真实的数据生成过程，模型中的参数恰好从指定的先验分布中采样。 我们对预测准确性感兴趣有两个原因：第一，衡量我们正在使用的模型的性能； 第二，比较模型。 我们在模型比较中的目标不一定是选择具有最低估计预测误差的模型，甚至不需要对候选模型进行平均。

​        当不同模型具有相同数量且以相同方式估计的参数时，人们可能会简单地直接比较它们的最佳拟合对数预测密度，但是当比较不同大小或不同有效大小的模型时（例如，比较使用uniform、 样条或高斯过程先验），重要的是对较大模型的自然能力进行一些调整以更好地拟合数据，虽然有些时候这种情况只是偶然。

### 



### 7.2.1 使用可用数据估计样本外预测准确性 (Estimating out-of-sample predictive accuracy using available data)

​         有几种方法可用于估计预期的预测准确度，而无需等待样本外数据。我们不知道真实的分布 。 相反，我们可以考虑各种近似。我们知道一般情况下没有近似值，但预测准确性非常重要。 我们在这里列出了几个看似合理的近似值。 这些方法中的每一种都有缺陷。



**1、样本内预测准确度**

当我们要用
$$
\begin{aligned}
\text { computed lppd } &=\text { computed log pointwise predictive density } \\
&=\sum_{i=1}^{n} \log \left(\frac{1}{S} \sum_{s=1}^{S} p\left(y_{i} \mid \theta^{s}\right)\right)
\end{aligned} \ \ \ \ (7.1.6)
$$
进行计算时，即使用贝叶斯逐点公式进行计算，我们往往会高估$$\sum_{i=1}^{n} \mathrm{E}_{f}\left(\log p_{\text {post }}\left(\tilde{y}_{i}\right)\right) \ \ \ \ (7.1.3)$$ 因为它是根据拟合模型的数据进行评估的。 



**2、调整后的样本内预测准确度** 

​         鉴于 lppd 是 elppd 的有偏估计，下一个合乎逻辑的步骤是纠正该偏差。 AIC、DIC 和 WAIC 等公式通过从 lppd 之类的东西开始，然后减去对参数数量或有效参数数量进行拟合的修正，给出了 elppd 的近似无偏估计。 在许多情况下，这些调整可以给出合理的答案，但存在最多只能在预期中正确的一般问题。



**3、交叉验证**

​          可以尝试通过拟合来捕获样本外预测误差模型训练数据，然后在坚持集上评估这种预测准确性。 交叉验证避免了过度拟合的问题，但仍与手头的数据相关联，因此最多只能在预期中正确。 







### 7.2.2 用于正态线性模型的渐近对数预测密度(Log predictive density asymptotically, or for normal linear models)

​        在第 4 章指定的条件下，后验分布 $$p(y|\theta)$$在增加样本量的限制下接近正态分布。 在这个渐近极限中，后验由似然支配——先验只贡献一个因素，而like-hood贡献了 n 个因子，每个数据点一个——因此似然函数也接近相同的正态分布。 

随着样本大小不断增大趋于正无穷 $$n \rightarrow \infty$$，后验分布服从$$\mathrm{N}\left(\theta_{0}, V_{0} / n\right)$$正态分布，log预测函数密度是：
$$
\log p(y \mid \theta)=c(y)-\frac{1}{2}\left(k \log (2 \pi)+\log \left|V_{0} / n\right|+\left(\theta-\theta_{0}\right)^{T}\left(V_{0} / n\right)^{-1}\left(\theta-\theta_{0}\right)\right) \ \ \ (7.2.1)
$$
​     其中$$c(y)$$是常数，其只取决于$$y$$和模型类别，(7.2.1)虽然只是一个近似值，但是它对解释对数预测的基准很有用。下图为对数预测密度分布图。



<img src="C:\Users\len\AppData\Roaming\Typora\typora-user-images\image-20211019172136335.png" alt="image-20211019172136335"  />

<center/>图7.3对数预测密度分布图<center>


> 图片来源于Bayesian Data Analysis Third Edition Chapter 7





### 7.2.3 赤池信息准则（AIC） (Akaike information criterion (AIC))

​        在很多统计文献中，对$$\theta$$的推断往往不是通过后验分布而是通过点估计$$\tilde{\theta}$$来进行的，通常采用的方法是最大似然估计法。鉴于渐进正态分布的后验分布，在具有已知方差和均匀先验分布的正态线性模型的特殊情况下，且在给定最大似然估计的情况下，从对数预测密度中减去 k 是对 k 参数的拟合将提高预测准确度的修正，只是偶然
$$
\widehat{\mathrm{elpd}}_{\mathrm{AIC}}=\log p\left(y \mid \hat{\theta}_{\mathrm{mle}}\right)-k \ \ \ (7.2.2)
$$
​      与简单最小二乘法或最大似然估计下发生的情况相比，信息丰富的先验分布和层次结构倾向于减少过度拟合的数量。





### 7.2.4 偏差信息准则 (DIC) 和有效参数数量 (Deviance information criterion (DIC) and effective number of parameters)

​      DIC实际上是贝叶斯视角下的AIC，很多性质与AIC相近，其表达式为：
$$
\widehat{\mathrm{elpd}}_{\mathrm{DIC}}=\log p\left(y \mid \hat{\theta}_{\mathrm{Bayes}}\right)-p_{\mathrm{DIC}} \ \ \ (7.2.3)
$$
​     其中$$p_{DIC}$$是参数的有效数量，定义的公式为：
$$
p_{\mathrm{DIC}}=2\left(\log p\left(y \mid \hat{\theta}_{\text {Bayes }}\right)-\mathrm{E}_{\text {post }}(\log p(y \mid \theta))\right) \ \ \ (7.2.4)
$$
​     其中第二项中的期望是 θ 在其后验分布上的平均值。表达式 (7.2.4) 是使用模拟计算 $\theta^{s}$ 当$$s=1,2,3...S$$时：
$$
\text { computed } p_{\mathrm{DIC}}=2\left(\log p\left(y \mid \hat{\theta}_{\text {Bayes }}\right)-\frac{1}{S} \sum_{s=1}^{S} \log p\left(y \mid \theta^{s}\right)\right) \text  \ \ \ (7.2.5)
$$
​     还有， DIC 的实际数量是根据偏差而不是对数预测密度来定义的，见下式：
$$
\mathrm{DIC}=-2 \log p\left(y \mid \hat{\theta}_{\text {Bayes }}\right)+2 p_{\mathrm{DIC}} \ \ \ (7.2.6)
$$




### 7.2.5 Watanabe-Akaike 广泛可用的信息准则 (WAIC)

​        可以简称这个方法为WAIC方法，这个是一种相对完整的贝叶斯方法，用于计算样本外期望，从计算的对数逐点后验预测密度开始，然后添加对有效参数数量的校正以调整过度拟合。这种方法有两种调整，可以视为交叉验证的近似值。

​      **调整1的公式：**
$$
p_{\mathrm{WAIC} 1}=2 \sum_{i=1}^{n}\left(\log \left(\mathrm{E}_{\mathrm{post}} p\left(y_{i} \mid \theta\right)\right)-\mathrm{E}_{\mathrm{post}}\left(\log p\left(y_{i} \mid \theta\right)\right)\right) \ \ \ (7.2.7)
$$
​       **调整2的公式(使用在 n 个数据点上求和的对数预测密度中单个项的方差)：**
$$
p_{\text {WAIC } 2}=\sum_{i=1}^{n} \operatorname{var}_{\text {post }}\left(\log p\left(y_{i} \mid \theta\right)\right) \ \ \ (7.2.8)
$$
​         我们推荐 $$p_{\mathrm{WAIC} 2}$$因为它的级数扩展与 LOO-CV 的级数扩展更相似，而且在实践中似乎给出更接近 LOO-CV 的结果。

​         对于样本量大、方差已知且系数先验分布均匀的正态线性模型，$$p_{\mathrm{WAIC} 1}$$和$$p_{\mathrm{WAIC} 2}$$大约等于模型中的参数数量。更一般地，调整可以被认为是模型中“无约束”参数数量的近似值，其中如果参数是在没有约束或先验信息的情况下估计的，则计为 1，如果是完全约束或所有参数则计为 0，有关参数的信息来自先验分布，如果数据和先验分布都提供信息，则为中间值。

​         与 AIC 和 DIC 相比，WAIC 具有对后验分布求平均而不是对点估计进行调节的理想特性。这在预测环境中尤其重要，因为 WAIC 正在评估贝叶斯环境中实际用于新数据的预测。 AIC 和 DIC 估计插件预测密度的性能，但这些度量的贝叶斯用户仍将使用后验预测密度进行预测。





### 7.2.6 留一法交叉验证 (Leave-one-out cross-validation)

​         贝叶斯交叉验证中，将数据重复划分为训练集 $$y_{train}$$和保持集 $$y_{holdout}$$，然后将模型拟合到 $$y_{train}$$（从而产生后验分布 $$p_{train}(\theta) = p(\theta|y_{train})$$   )使用对数估计值评估此拟合保持数据的预测密度：$$\log p_{\text {train }}\left(y_{\text {holdout }}\right)=\log \int p_{\text {pred }}\left(y_{\text {holdout }} \mid \theta\right) p_{\text {train }}(\theta) d \theta$$    

为简单起见，我们将注意力限制在留一法交叉验证 (LOO-CV) 上，这有 $$n$$个特殊情况，其中每个保留集代表一个数据点。对 $$n$$ 个数据点中的每一个进行分析（或者，如果 $$n$$ 很大，则可能是有效计算的随机子集）产生 n 个不同的推论 $$p_{post}(-i)$$，每个推论由 S 个后验模拟 $$\theta^s$$总结。

​         样本外预测拟合的贝叶斯 LOO-CV 估计为：
$$
\operatorname{lppd}_{\text {loo-cv }}=\sum_{i=1}^{n} \log p_{\text {post }(-i)}\left(y_{i}\right), \text { calculated as } \sum_{i=1}^{n} \log \left(\frac{1}{S} \sum_{s=1}^{S} p\left(y_{i} \mid \theta^{i s}\right)\right) \ \ \ (7.2.9)
$$
每个预测都包括$$n-1$$个数据点，这会导致对预测拟合的低估。对于大的数据量，差异可以忽略不计，但对于小 $n$（或使用 k 折交叉验证时），我们可以使用一阶偏差校正 $b$，通过估计以 $n$ 个数据点为条件会获得更好的预测。
$$
b=\operatorname{lppd}-\overline{\operatorname{lppd}}_{-i} \ \ \ (7.2.10)
$$
其中
$$
\overline{\mathrm{lppd}}_{-i}=\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{n} \log p_{\text {post }(-i)}\left(y_{j}\right), \text { calculated as } \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{n} \log \left(\frac{1}{S} \sum_{s=1}^{S} p\left(y_{j} \mid \theta^{i s}\right)\right) \ \ \ (7.2.11)
$$


​        偏差校正后的贝叶斯 LOO-CV 是：
$$
\mathrm{lppd}_{\mathrm{cloo}-\mathrm{cv}}=\operatorname{lppd}_{\mathrm{loo}-\mathrm{cv}}+b \ \ \ (7.2.12)
$$
​        偏差校正 b 很少使用，因为它通常很小，与其他方法进行比较，我们计算了有效参数数量的估计为：
$$
p_{\text {loo-cv }}=l p p d-l p p d_{\operatorname{loo}-c v} \ \ \ (7.2.13)
$$
​        交叉验证类似于 WAIC，因为它需要将数据分成不相交的、理想情况下条件独立的部分。当应用于结构化模型时，这代表了该方法的局限性,同时其也不便于计算，除非我们能快速的得到$$p_{post}$$的分布







### 7.2.7 比较样本外预测准确度的不同估计 (Comparing different estimates of out-of-sample prediction accuracy)

​       上面讨论的所有不同措施都基于通过减去近似偏差校正来调整观测数据的对数预测密度。这些措施的出发点和调整都不同。比如$$AIC$$以数据的对数预测密度开始，$$DIC$$以后验均值 $$E(\theta|y)$$为条件，$$WAIC$$ 以对数预测密度开始，这三种方法中，只有$$WAIC$$ 是完全贝叶斯的，因此在使用偏差校正公式时我们更喜欢使用它。



**一个例子: 选举预测模型中的预测误差**（文章最开始的预测选举回归模型）

​        这是对上文我所总结的几种方法的一种实现，与文章基本概念中的实例做到首尾呼应，帮助我在一定程度上理解了模型评估的几种方法的使用和含义。 

**$$AIC$$方法：**合所有 15 个数据点，向量$$（\tilde{a},\tilde{b},\tilde{\sigma})$$的最大似然估计为$$ (45.9, 3.2, 3.6)$$。由于估计了 3 个参数，因此
$$
\widehat{\operatorname{elpd}}_{\mathrm{AIC}}=\sum_{i=1}^{15} \log \mathrm{N}\left(y_{i} \mid 45.9+3.2 x_{i}, 3.6^{2}\right)-3=-43.3
$$
**$$DIC$$方法：**
$$
\mathrm{E}_{\text {post }}(y \mid \theta)=\frac{1}{S} \sum_{s=1}^{S} \sum_{i=1}^{15} \log \mathrm{N}\left(y_{i} \mid a^{s}+b^{s} x_{i},\left(\sigma^{s}\right)^{2}\right)=-42.0,
$$
**基于 S比较大的条件进行模拟:**
$$
\log p\left(y \mid \mathrm{E}_{\text {post }}(\theta)\right)=\sum_{i=1}^{15} \log \mathrm{N}\left(y_{i} \mid \mathrm{E}(a \mid y)+\mathrm{E}(b \mid y) x_{i},(\mathrm{E}(\sigma \mid y))^{2}\right)=-40.5
$$
*经过计算，我们能得出$$p_{DIC}$$=3，$$\widehat{\operatorname{elpd}}_{\mathrm{DIC}}$$ =-43.5，$$DIC$$=87*



**$$WAIC$$**方法：

拟合模型下观测数据的对数逐点预测概率为：
$$
\operatorname{lppd}=\sum_{i=1}^{15} \log \left(\frac{1}{S} \sum_{s=1}^{S} \mathrm{~N}\left(y_{i} \mid a^{s}+b^{s} x_{i},\left(\sigma^{s}\right)^{2}\right)\right)=-40.9
$$
参数有效量：
$$
p_{\text {WAIC } 2}=\sum_{i=1}^{15} V_{s=1}^{S} \log \mathrm{N}\left(y_{i} \mid a^{s}+b^{s} x_{i},\left(\sigma^{s}\right)^{2}\right)=2.7
$$






## 参考文献

【1】机器学习 清华大学出版社 周志华
【2】Bayesian Data Analysis, Third Edition (C - Gelman, Andrew
【3】The_Impact_of_Emotion_A_Blend  Wei-Lun Chang

