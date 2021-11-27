# 列联表的贝叶斯分析

### 胡琦文2019302130044

## 引言
&emsp;&emsp;对于列联表数据形式的数据，可以使用带参数的先验分布，并将结果以带参数的后验分布或其他一些形式进行数据分析。所用的分析必须依赖于先验分布，这里描述的形式只适用于某种类型的先验知识，贝叶斯分析能够使某些二项式问题得到应用，结合一些一般性结论能够应用于列联表。这里使用的方法与方差分析有紧密联系，在简化涉及三个或更高维的列联表分析时需要进行检验。

## （一）二项分布
&emsp;&emsp;二项分布作为一般多项情况的结果的特殊情况，在基于二项分布进行推广时具有启发意义。以概率$\theta$进行N次独立的试验，导致$n$次成功实验和$(N-n)$次失败实验的概率为：

**$$
(1.1)		\theta^{n}(1-\theta)^{N-n}
$$**

&emsp;&emsp;将一个密度与以下数据成正比的先验分布作为先验分布是很方便的

**$$
(1.2)		\theta^{a}(1-\theta)^{b}
$$**

&emsp;&emsp;其中 $a, b>-1$ 当$a$ 和 $b$ 都趋向于-1时，后验密度与以下数据成正比

**$$
(1.3)		\theta^{a+n}(1-\theta)^{b+N-n}
$$**

&emsp;&emsp;由$β$分布和$F$分布之间的已知关系得到下式

**$$
(1.4)		F=\left(\frac{b+N-n+1}{a+n+1}\right)\left(\frac{\theta}{1-\theta}\right)
$$**

&emsp;&emsp;$F$分布的自由度为$[2(a+n+1), 2(b+N-n+1)]$

&emsp;&emsp;由于两个独立的服从卡方分布的变量相比服从$F$分布，所以$F$的对数变换及其近似正态性的使用与卡方或伽马变量的相同变换及其近似正态性密切相关。Bartlett和Kendall[2]讨论了后一种变换，认为可以安全地用于$n=10$及以上的情况，并且试探性地用于$n=5$至$n=9$的情况，但在$n<5$以下的情况下可能根本不使用。一般来说，自由度应不少于成功（或失败）次数的两倍，因此，只要这两个数字中的较小者为5或以上，就肯定可以使用这个转换。
Fisher在$F$中取$z=\frac{1}{2}$ : 在这种情况下，用F的自然对数更为方便。那么，Fisher的近似方法相当于说，如果$F$有$\nu_{1}$和$\nu_{2}$的自由度，那么$\ln F$是近似正态的，其均值为 $\ln \left\{\left(\nu_{1}-1\right) /\left(\nu_{2}-1\right)\right\}$，方差为$2\left(\nu_{1}^{-1}+\nu_{2}^{-1}\right)$对于$\ln \{\theta /(1-\theta)\}$，均值为：

**$$
(1.5)		\ln \left\{\left(a+n+\frac{1}{2}\right) /\left(N-n+b+\frac{1}{2}\right)\right\}
$$**

&emsp;&emsp;方差为：

**$$
(1.6)		(a+n+1)^{-1}+(b+N-n+1)^{-1}
$$**

&emsp;&emsp;因此，我们要考虑有利于成功的几率的自然对数，即 $\theta /(1-\theta)$。如果$\theta$的值是已知的，并且小于$\frac{1}{2}$，通常会引用它的值到固定的小数位。在这种情况下，$\theta$的对数可以被引用到一个固定的位数，它的特定变化对所有的$\theta$都具有同等意义。使用对数的另一个更重要的论据是，在下面要讨论的情况下，它们存在某些加法属性，基于独立的重要属性是以乘法方式表达的，而对对数而言是加法方式。本文的结果将完全用对数和它们的多项式概括来表达。

&emsp;&emsp;接下来考虑先验分布（1.2）。考虑极限情况下的$a=b=-1$，并通过以下两个主要论据来证明这一选择。在许多情况下，0的先验知识是很小的，考虑对标准先验的分析有一些好处，它可以用于大多数没有明显先验知识的应用中。如果要使用对数，那么最初的贝叶斯假设将建议把它的所有值都视为同等可能。注意到试验的结果不能减少$\beta$分布中$\theta$和$(1-\theta)$的权值，并且推测不能减少关于$\theta$的信息量，就可以看出这是合理的。由此产生的分布是不恰当的，但这并不影响我们，因为我们将使用$F$的自由度很大且后验分布是恰当的近似值。

&emsp;&emsp;关注$a=b=-1$的第二个原因是，这种特殊情况可以被视为属于$\beta$族的所有先验分布的典型形式（1.2）也就是任何贝塔分布都可以简化为$a=b=-1$的典型形式。为了说明这一点，我们注意到，如果先验分布是由（1.2）给出的，它可以被视为由$(a+1)$成功和$(b+1)$失败组成的数据的后验分布，这些数据的先验分布为$a=b=-1$。因此，如果实际数据产生了$n$个成功和$(N-n)$个失败，我们可以把$\theta$的总知识看作是由$(n+a+1)$个成功和$(N-n+b+1)$个失败组成的，对于这些数据，先验具有典型形式。因此，通过在实际观察到的成功和失败的数量上分别加上$(a+1)$和$(b+1)$，$\beta$分布可以被经典形式所取代。当然，该方法不适用于非$\beta$分布的先验分布：见下文第8节的讨论。

&emsp;&emsp;在许多应用中，先验分布的$a$和$b$的值可能确实很小。如果样本值$n$和$(N-n)$成功和失败的数量都很大，从（1.5）和（1.6）中我们可以看出，$a$和$b$的实际小值并不重要。因此，采取$a=b=-1$是合理的。在$a=b=-1$的情况下，$\ln \{\theta /(1-\theta)\}$的后验分布近似于正态分布，其均值为：

**$$
(1.7)		\ln \left\{\left(n-\frac{1}{2}\right) /\left(N-n-\frac{1}{2}\right)\right\}
$$**

&emsp;&emsp;方差为$n^{-1}+(N-n)^{-1}$，简化后记作:

**$$
(1.8)		\quad \ln \{\theta /(1-\theta)\} \sim N\left[\ln \{n /(N-n)\}, n^{-1}+(N-n)^{-1}\right]
$$**

&emsp;&emsp;在本文中，通过在计算后验平均值时减少$\frac{1}{2}$的观测值，可以对$a=b=-1$的先验近似值进行一些改进。换句话说，(1.8)说真实的对数是关于样本对数的近似正态分布，方差等于成功和失败两类数字的倒数之和。

## （二）多项式分布

&emsp;&emsp;让$k$表示类的数量，在二项式的情况下，$k=2$；令$\theta_{1}, \theta_{2}, \cdots, \theta_{k}$表示$k$类的概率：对于所有的$i$，必然有$\sum \theta_{i}=1$。在$N=\sum n_{i}$次独立试验中，以上述概率对各类进行试验时，令$n_{1}, n_{2}, \cdots, n_{k}$表示各类的观察数字。

&emsp;&emsp;我们采用Fisher的另一个建议：如果$n_{i}(i=1,2, \cdots k)$是独立的泊松变量，其平均值是$\Psi_{i}$，那么它们的条件分布为$\theta_{i}=\Psi_{i} / \sum \Psi_{i}$

**$$
(2.1)		\begin{aligned}
p\left(n_{1}, n_{2}, \cdots, n_{k} \mid N\right) &=p\left(n_{1}, n_{2}, \cdots, n_{k}\right) / p(N) \\
&=\frac{e^{-\Sigma \Psi_{i}} \prod\left(\Psi_{i}^{n_{i}} / n_{i} !\right)}{e^{-\Sigma \Psi_{i}}\left(\sum \Psi_{i}\right)^{N} / N !} \\
&=N ! \prod\left(\theta_{i}^{n_{i}} / n_{i} !\right)
\end{aligned}
$$**

&emsp;&emsp;对这一结果的另一种看法是，泊松的概率分布只取决于$\theta=\sum \Psi_{i}$，而给定$N$的条件多叉分布，它只取决于$\theta_{1}, \theta_{2}, \cdots, \theta_{k}$。因此，在先验的这些条件下，$\theta_{i}$的后验分布将只取决于似然的二项式部分。也就是说，这个后验分布可以通过泊松装置得到。

&emsp;&emsp;对于$\Psi_{i}$的一个适当的先验分布，可以通过假设它们的对数是独立的，并且均匀分布在整个实线上得到。从$\Psi_{1}, \Psi_{2}, \cdots, \Psi_{k}$到$\theta, \theta_{1}, \theta_{2}, \cdots, \theta_{k-1}$的变形的雅各布式，很容易发现是$\theta^{k-1}$，因此

**$$
(2.2)		\frac{d \Psi_{1} d \Psi_{2} \cdots d \Psi_{k}}{\Psi_{1} \Psi_{2} \cdots \Psi_{k}}=\theta^{k-1} \frac{d \theta d \theta_{1} \cdots d \theta_{k-1}}{\theta^{k} \theta_{1} \cdots \theta_{k}}=\left(\frac{d \theta}{\theta}\right)\left(\frac{d \theta_{1} \cdots d \theta_{k-1}}{\theta_{1} \cdots \theta_{k}}\right)
$$**

&emsp;&emsp;然后，通过将（2.2）中的最后一个因素乘以（2.1）中的似然，得到$\theta_{i}$的后验分布，其结果与以下数据成正比

**$$
(2.3)		\Pi \theta_{i}^{n_{i}-1}
$$**

&emsp;&emsp;通过考虑泊松分布寻求（2.3）的近似值。给定$n_{i}$的$\Psi_{i}$的后验分布正比于

**$$
(2.4)		\prod\left(e^{-\Psi_{i}} \Psi_{i}^{n_{i}-1}\right)
$$**

&emsp;&emsp;因此，$Psi_{i}$是独立的，每一个都是III型或Gamma分布的。但是，正如上面所解释的，如果一个变量有这样的分布，那么除了较小的$n_{i}$值之外，它的对数是近似正态分布的。均值为$\ln n_{i}$，方差为$n_{i}^{-1}$。

&emsp;&emsp;与实验设计和分析中使用的术语一样，具有系数相加为零的量的线性形式将被称为这些量的对比。考虑对$\ln \Psi_{i}$的对比

**$$
\sum a_{i} \ln \Psi_{i}=\sum a_{i} \ln \left(\theta \theta_{i}\right)=\sum a_{i} \ln \theta_{i}
$$**

&emsp;&emsp;因为$\sum a_{i}=0$，因此，对于$\ln \Psi_{i}$的对比与对于$\ln \theta_{i}$的对比是一致的。但$\ln \Psi_{i}$的对比是近似正态分布的，它们的任何一组都是近似联合正态分布的。根据前面给出的论据，$\ln \theta_{i}$的这些分布适用于多项式分布。因此，我们有以下情况
* ### 定理1

&emsp;&emsp;如果随机变量$n_{1}, n_{2}, \cdots, n_{k}$服从参数为$\theta_{1}, \theta_{2}, \cdots, \theta_{k}的多项式分布；同时如果我们的先验分布在区域$\theta_{i} \geqq 0, \sum \theta_{i}=1$内具有密度比例$\left(\prod \theta_{i}\right)^{-1}$，那么，当常数$a_{p i}(p=1,2, \cdots, m ; i=1,2, \cdots, k ; m<k)$满足 $\sum_{i} a_{p i}=0$，则对比度的联合后验分布$\underline{ }_{i} a_{p i} \ln \theta_{i}(p=1,2, \cdots, m)$近似正态，均值为

**$$
(2.5)		\sum_{i} a_{p i} \ln n_{i}
$$**

&emsp;&emsp;和协方差

**$$
(2.6)		\sum_{i} a_{p i} a_{g i} n_{i}^{-1}
$$**

&emsp;&emsp;均值和协方差的表达式来源于$\ln \Psi_{i}$以及它们均值和方差的独立性。
由(2.5)给出的方法可以写成

**$$
(2.7)		\sum_{i} a_{p i} \ln \left(n_{i} / N\right)
$$**

&emsp;&emsp;在二项式的情况下，$k=2$，唯一的对比是$\ln \theta_{1}-\ln \theta_{2}=\ln \{\theta /(1-\theta)\}$的倍数。因此，先前关于二项式的结果是该定理的一个特例。如果把注意力集中在对数上，那么可以使用简单的正态结果：特别是可以使用方差分析的方法。我们将在下文中看到，在分析二项式数据，特别是列联表数据时，许多有关的参数都可以用对数来表示，因此，可以应用正态理论，与二项式的情况一样，通过从（2.5）中的$n_{i}$除去$\frac{1}{2}来改善近似值。

## （三）关于贝叶斯分析的讨论
&emsp;&emsp;在继续讨论定理1的应用之前，有必要澄清公开使用贝叶斯定理和先验分布的统计分析中的几个要点。这种分析的目的是提供参数的后验分布，或者，如果只有某些参数是感兴趣的，则提供这些参数的边际后验分布。如果数据接下来要被用作决策的基础，那么后验分布为计算最佳决策提供了必要的材料。一旦获得后验分布，剩下的唯一问题就是如何呈现它的描述性问题。如果我们遵循经典统计学及其置信区间的概念，很自然地通过给出一个包含相当大的，例如95%的后验概率的区间来总结后验分布。这样做的缺点是只给出了分布的尾部信息，并没有提供例如最可能的值的任何好主意。除了95%的值之外，提供中位数和其他量值会有帮助。在某些情况下，即使是一个区间也可能被认为是一个过于详细的总结，特别是在参数的一个值特别重要的情况下。在这些情况下，我们可能会觉得根据后验分布说这个值是否是一个可能的值就足够了。再次借用经典统计学的一个想法，即置信集是在置信集水平上作为无效假设进行检验时不会被拒绝的值的集合，我们可以通过查看特定值是否位于包含后验分布的比例$(1-\alpha)$的区间内来进行贝叶斯的显著性检验。如果不是，那么可以说这个值在$\alpha$上是显著的：在这个意义上，它不属于具有合理的高后验概率的一组值。

&emsp;&emsp;当然，有许多区间都包含了一定比例的正向分布。我们将选择一个区间（通常是唯一的），这个区间内的任何数值都不比没有这个区间的任何数值的可能性小。这就提供了通常意义上的最短区间。在制定这个意义检验的概念时，没有考虑到任何经济或一般决策理论的考虑。这似乎与经典的显著性检验的精神是一样的，也没有纳入关于特别关注的价值的强烈的先验想法，我们的先验分布在特殊价值的附近是 "平滑 "的。

&emsp;&emsp;将这些想法应用于对数的后验分布。$\phi_{1}, \phi_{2}, \cdots, \phi_{s}$是s个线性独立的对数，其均值为$m_{i}$，协方差为$v_{i j}$，表达式来自（2.5）和（2.6）。$\phi_{i}$的联合密度在椭圆体上是恒定的：

**$$
(3.1)		\sum_{i, j=1}^{s}\left(\phi_{i}-m_{i}\right) v^{i j}\left(\phi_{j}-m_{j}\right)=c
$$**

&emsp;&emsp;其中，$v^{i j}$是分散矩阵的逆元素，$c$是任意正常数。后验概率为：

**$$
(3.2)		\sum_{i, j=1}^{s}\left(\phi_{i}-m_{i}\right) v^{i j}\left(\phi_{j}-m_{j}\right) \leqq \chi_{\alpha}^{2}
$$**

&emsp;&emsp;在$\alpha=0.05$的情况下，（3.2）的后验概率为95%。提出一个对假设的显著性检验$\phi_{i}=\phi_{i}^{(0)}$，当$\phi_{i}^{(0)}=0(i=1,2, \cdots, s)$时，相关的统计量减少为

**$$
(3.3)		\sum_{i, j=1}^{s} m_{i} v^{i j} m_{j}
$$**

&emsp;&emsp;可与$\chi^{2}$进行对比。

&emsp;&emsp;在应用（3.2）和（3.3）时，尽管$m_{i}$和 $v^{i j}$是统计量，$\phi_{i}$是参数，但它们才是随机变量。在通常的论证中，$m_{i}$是随机变量，服从均值为$\phi_{i}$，协方差为$v_{i j}$（如果已知）的正态分布，式（3.2）和（3.3）仍然正确。特别是（3.3）与$\chi^{2}$分布的比较是方差分析的基础，方差是超假设的已知。因此，在用刚才描述的贝叶斯方法分析对数时，我们可以用方差分析的方法。在下面的章节中，我们将继续使用从这一经典领域衍生出来的观点。

&emsp;&emsp;在继续讨论定理1的应用之前，需要根据这些关于贝叶斯分析的评论做两点说明。第一点是关于基于（3.2）和（3.3）的方法在对数的线性变换下的不变性。由于二次方形式是不变的，因此可以用任何一组对数的线性变换来代替任何一组对数，以此简化分析。

&emsp;&emsp;第二点是关于第III类分布的近似值（公式（2.4）），这是这些方法的基础。一个值得关注的近似是基于统计量$\sum(O-E)^{2} / E$的经典的$\chi^{2}$近似，它可以用于贝叶斯分析，其思路与刚才对（3.3）的解释相同。初步的工作表明，在提供后验分布的近似值方面，$\chi^{2}$的效果比log-odds差。但在二项式情况下用$\theta$的均匀先验分布代替上面使用的$\ln \theta$的均匀分布，其效果非常优异，因此$\chi^{2}$可能是后验分布的一个很好的近似。

## （四）二维列联表
&emsp;&emsp;二维列联表通常是通过将一些项目归入$A_{1}, A_{2}, cdots A_{r}$中的$r$种类别之一， 被同时归入$B_{1}, B_{2}, cdots B_{s}$的一个排他性的穷举类中。一个物品被归入$A_{i}$和$B_{j}$两类的概率$\theta_{i j}$，对每个物品来说都是一样的。如果N个项目被连续分类，则$A_{i}$和$B_{j}$类中的项目数量$n_{i j}(i=1,2, \cdots r ; j=1,2, \cdots s)$服从指数为$N$、参数为$\theta_{i j}$的多项式分布。被归类为$A_{i}$的概率为$\sum_{j=1}^{s} \theta_{i j}=\theta_{i.}$，类似的，$\theta_{. j}=\sum_{i=1}^{r} \theta_{i j}$是被归类为$B_{j}$的概率。通常来说，刚才的描述指的是一个没有边际固定（只有总数$N$）的列联表。下文将对有一个或两个边际固定的列联表作一些说明。

&emsp;&emsp;除了通过$\theta_{i j}$之外，对列联表进行参数化往往是很方便的。例如，$A_{i}(i=1,2, \cdots r), \theta_{i .}$，以及给定$A_{i}, \theta_{i j} / \theta_{i .}$的$B_{j}$类的条件概率，都可以进行参数化。在确定它们的先验分布之前，不能直接对它们进行推断，证明如下：

* ### 定理2

&emsp;&emsp;如果$\theta_{i j}(i=1,2, \cdots r; j=1,2, \cdots s)$的先验分布与$\prod_{i j} \theta_{i j}^{-1}$成比例，那么$\theta_{i .}$以及$\phi_{i j}=\theta_{i j} / \theta_{i .}$的先验分布与下式成比例

**$$
(4.1)		\prod_{i} \theta_{i}^{-1} \prod_{i, j} \phi_{i j}^{-1}
$$**

&emsp;&emsp;考虑参数$\theta_{i j}$的变化，其中对于所有$i$和$j$，排除$i=r, j=s$到$\theta_{i .} (i<r)$和 $\phi_{i j}(j<s)$。由于参数之间存在的约束，某些值必须被排除，即$\sum_{i j} \theta_{i j}=\sum_{i} \theta_{i .}=\sum_{j} \phi_{i j}=1$。其中：

**$$
(4.2)		\begin{array}{rlr}
\theta_{i j} & =\phi_{i j} \theta_{i} . & i<r, j<s, \\
\theta_{r j} & =\phi_{r j}\left(1-\sum_{i<r} \theta_{i \cdot}\right) & j<s, \\
\theta_{i 8} & =\left(1-\sum_{j<s} \phi_{i j}\right) \theta_{i} . & i<r .\\
\end{array}
$$**
                                                                        
&emsp;&emsp;从\theta_{i j}$到新参数的转换的雅各布系数很容易计算出来：
                                                                        
**$$
(4.3)		\prod_{i=1}^{r} \theta_{i}^{s-1}
$$**
                                                                        
&emsp;&emsp;于是，$\phi_{i j}$和$\theta_{i .}$的联合密度与$\prod_{i, j} \theta_{i j}^{-1}$和（5.3）成正比：

**$$
\prod_{i=1}^{r} \prod_{j=1}^{\delta}\left(\theta_{i .}\phi_{i j}\right)^{-1} \prod_{i=1}^{r} \theta_{i .}^{s-1}=\prod \theta_{i .}^{-1} \prod \phi_{i j}^{-1}
$$**

&emsp;&emsp;该定理证明，关于$r s$类的先验分布与关于减少的$r$类的先验分布是一致的。一般来说，类可以合并，定理1的近似值适用于减少的参数。列联表的多项式可能性正比于：

**$$
\prod_{i j} \theta_{i j}^{n_{i j}}=\prod_{i} \theta_{i}^{n_{i .}} \prod_{i, j} \phi_{i j}^{n_{i j}}
$$**
                                                                        
&emsp;&emsp;其中，$n_{i .}=\sum_{j} n_{i j}$
                                                                        
&emsp;&emsp;让我们将结果应用于$2 \times 2$表$(r=s=2)$。一个简单的解析是通过$\theta_{1 .}=\theta_{11}+\theta_{12}$（分类为$A_{1}$类的概率），以及$\phi_{11}$和$\phi_{21}$（分别给出$A_{1}$和$A_{2}$时的分类为$B_{1}$的概率）。将这两个定理的结果结合起来，我们可以看到后验分布近似如下：

(a) $\ln \left\{\theta_{1 .}/ \theta_{2 .}\right\}$服从均值为$\ln \left\{n_{1 .}/ n_{2 .}\right\}$方差为$n_{1 .}^{-1}+n_{2 .}^{-1}$的正态分布
(b) $\ln \left\{\phi_{11} / \phi_{12}\right\}$服从均值为$\ln \left\{n_{11} / n_{12}\right\}$方差为$n_{11}^{-1}+n_{12}^{-1}$的正态分布
(c) $\ln \left\{\phi_{21} / \phi_{22}\right)$服从均值为$\ln \left\{n_{21} / n_{22}\right\}$方差为$n_{21}^{-1}+n_{22}^{-1}$的正态分布

&emsp;&emsp;这三个分布是独立的，是由于在每一种情况下，我们都是在处理二项式情况下的三个似然和它们相应的先验因子。在$A$和$B$互换的情况下，可以得到对称的结果。如果与$A$分类相对应的边际是固定的，则推断（a）不可用。
                                                                        
&emsp;&emsp;很多关于$2 \times 2$表的文献都只关注能够反映两个分类变量间关系的参数，甚至更具体的是致力于研究两个分类变量是否独立：也就是说，$\theta_{i j}=\theta_{i} \cdot \theta_{. j}$。两个可行的检验是Fisher的精确检验和$\chi^{2}$近似。对分类变量之间相关程度的测量可以通过多种方式进行：我们寻求一种对数的方式。如果这些分类变量是独立的：$\phi_{11}=p\left(B_{1} \mid A_{1}\right)=\theta_{11} /\left(\theta_{11}+\theta_{12}\right)=$ $\theta_{21} /\left(\theta_{21}+\theta_{22}\right)=p\left(B_{1} \mid A_{2}\right)=\phi_{12}$，也可以记作：
                                                                        
**$$
\theta_{11} / \theta_{12}=\theta_{21} / \theta_{22} \quad \text { or } \quad \phi_{11} / \phi_{12}=\phi_{21} / \phi_{22}
$$**
                                                                        
&emsp;&emsp;或者说，在$A_{1}$和$A_{2}$内，$B$分类的几率是一样的。因此，一个可以考虑的参数是log-odds

**$$
(4.4)		\begin{aligned}
\phi &=\ln \theta_{11}-\ln \theta_{21}-\ln \theta_{12}+\ln \theta_{22} \\
&=\ln \phi_{11}-\ln \phi_{21}-\ln \phi_{12}+\ln \phi_{22}
\end{aligned}
$$**
                                                                        
&emsp;&emsp;根据主定理，或根据上述（b）和（c）的组合，该参数近似于正态分布，其平均值为

**$$
(4.5)		\ln n_{11}-\ln n_{21}-\ln n_{12}+\ln n_{22}
$$**
                                                                        
&emsp;&emsp;方差为
                                                                        
**$$
(4.6)		n_{11}^{-1}+n_{21}^{-1}+n_{12}^{-1}+n_{22}^{-1}
$$**
                                                                        
&emsp;&emsp;独立性的无效假设是 $\phi=0$，可以通过参考以下内容进行检验

**$$
(4.7)		\frac{\left(\ln n_{11}-\ln n_{21}-\ln n_{12}+\ln n_{22}\right)^{2}}{n_{11}^{-1}+n_{21}^{-1}+n_{12}^{-1}+n_{22}^{-1}}
$$**

&emsp;&emsp;通过在取对数之前从分子中的每个$n_{i j}$中减去$\frac{1}{2}$，可能可以改善近似度。这个结果可用于有一个或没有固定边际的表格。如果$\phi$被用作关联度量，使用(4.5)和(4.6)可以很容易地得到它的贝叶斯置信区间。$\phi$的自然参数是$\theta_{1 .}$和$\theta_{.1}$，即边际概率。

&emsp;&emsp;在处理大于2元的表格时，注意到该分析与方差分析之间的某些相似之处会有所帮助。如果表格（没有固定的边际）以$\theta_{1 .}, \theta_{. 1}$和$\phi$进行分析，前两个对应于$A$和$B$分类的主效应，最后一个对应于两个分类的交互作用：事实上，（4.4）的形式正是基于概率对数的交互作用。然而，如果$\phi$被认为是一种交互作用，那么相应的主效应将是

**$$
(4.8)		\ln \theta_{11}-ln \theta_{21}+ln \theta_{12}-ln \theta_{22}
$$**
                                                                        
**$$
(4.9)		\ln \left\{theta_{1 .}/ θ_{2 .}\right\}=\ln \left(\theta_{11}+\theta_{12}\right)-\ln \left(\theta_{21}+\theta_{22}\right)
$$**
                                                                        
&emsp;&emsp;即使使用了（4.8），也应该注意到，它并不独立于$\phi$，因为$\ln \theta_{i j}$的方差并不相等。因此，方差分析的思想在研究列联表时在概念上是有用的，总的变异被分解成不同的部分，这些部分是独立的，不一定或通常是独立的：个别的贡献也不具有方差分析或$\chi^{2}$经常要求的相加的特性。$\theta_{1 .}, \theta_{. 1}$和$\phi$的后验分布的相互依赖性如下。$\theta_{1 .}$与$\phi\left((a)-(c)\right.$以上无关，$\theta_{. 1}$也是如此。但$\theta_{1}$不独立于$\theta_{.1}$，$\phi$也不独立于一对$\left(\theta_{1 .} , \theta_{. 1}\right)$。似乎没有任何参数化在两种分类中是对称的，并给出独立的分布。
                                                                        
&emsp;&emsp;$2 \times 2$列联表的两个边际$n_{i .}$和$n_{. j}$都是固定的，其经典的例子是女士品茶。Bahadur提出了一个条件分布的参数化，给定两个边际进行贝叶斯分析。如果两个分类变量是独立的，很容易计算出$n_{11}$的这个分布：它是超几何的。用$p\left(n_{11}\right)$表示。那么推荐使用下式作为分类变量不独立时的分布：
                                                                        
**$$
(4.10)		p\left(n_{11} \mid \theta\right)=e^{\theta_{11}} p\left(n_{11}\right) / \sum_{n} e^{\theta n} p(n)
$$**
                                                                        
&emsp;&emsp;其中，参数$\theta$衡量两个分类变量之间的关联程度：$\theta=0$说明两变量独立。但对于参数$\theta$的解释有一定的难度。 但这也许是对实验设计的批评，因为在任何情况下，人们都很难从任何分析中推断出这位女士今后对任何一杯茶进行正确分类的概率是多少。Elfving在对该问题的Neyman-Pearson研究中使用了分布（4.10）。 
                                                                        
&emsp;&emsp;$2 \times 2$表的分析扩展到一般的$r \times 2$表是很简单的。关于$\theta_{. 1}$或关于$\theta_{i .}(i=1,2, \cdots r)$的推断可以分别以以上述二项式和多项式的方式进行。两种分类之间没有关联的无效假设是$\theta_{i j} / \theta_{i}=\phi_{i j}$不依赖于$i$。就log-odds而言，这就等于说下式不依赖于$i$，或者所有的log-odds是相等的。
                                                                        
**$$
(4.11)		\ln \theta_{i 1}-\ln \theta_{i 2}
$$**
                                                                        
&emsp;&emsp;此时，这些对数概率是独立的，因此它们的后验密度的对数与下列各项成正比
                                                                        
**$$
(4.12)		\sum\left\{\left(\ln \theta_{i 1}-\ln \theta_{i 2}\right)-\left(\ln n_{i 1}-\ln n_{i 2}\right)\right\}^{2}\left(n_{i 1}^{-1}+n_{i 2}^{-1}\right)^{-1}
$$**
                                                                        
&emsp;&emsp;这可以写成关于平均值的加权平方和和涉及平均值的项，因此可以用前者来检验（4.11）中的所有内容是否相等。如果$x_{i}=\ln n_{i 1}-\ln n_{i 2}$并且$m_{i}=n_{i 1}^{-1}+n_{i 2}^{-1}$，则检验标准为
                                                                        
**$$
(4.13)		\sum\left(x_{i}-x .\right)^{2} m_{i}
$$**
                                                                        
&emsp;&emsp;其中，$x .=5 m_{i} x_{i} / \sum m_{i}$，并服从自由度为$(r-1)$的$\chi^{2}$分布。假设（4.11）相当于说存在常数$a_{i}, b_{j}(i=1,2, \cdots r ; j=1,2)$，使
                                                                        
**$$
(4.14)		\ln \theta_{i j}=a_{i}+b_{j}
$$**
                                                                        
&emsp;&emsp;此外，如果只考虑对比，$\ln \theta_{i j}$是方差为$n_{i j}^{-1}$的独立正态分布。因此，假设(5.14)是在通常的正态分布理论下的二维方差分析中没有交互作用的假设，方差是已知的，且不相等的，参数和观测值的角色是可以互换的，参数才是随机变量。当模型(4.14)被拟合时，该假设可以通过在模型中考虑适当的残差平方和进行检验。  
                                                                        
&emsp;&emsp;当我们考虑$r$和$s$都大于2的一般的$r \times s$表时，就会出现新的困难。无法找到既独立又能反映相关性假设的对数，即两种分类变量是不相干的。满足后一个要求很容易，但如果没有前一个要求，对应于（4.12）的二次方形式就不再是一个简单的平方和了。当$\ln \theta_{i j}$的方差是随机量是独立性无法实现。例如，考虑一个$3 \times 3$表格中的分类变量之间是否具有关联性的检验。四个对数的消失将相当于分类的独立性：
                                                                        
**$$
(4.15)		\begin{aligned}
&\ln \theta_{11}-\ln \theta_{12}-\ln \theta_{21}+\ln \theta_{22} \\
&\ln \theta_{11}-\ln \theta_{13}-\ln \theta_{21}+\ln \theta_{23} \\
&\ln \theta_{21}-\ln \theta_{22}-\ln \theta_{31}+\ln \theta_{32} \\
&\ln \theta_{21}-\ln \theta_{23}-\ln \theta_{31}+\ln \theta_{33}
\end{aligned}
$$**
                                                                        
&emsp;&emsp;但这些都是相关的，例如，第一个和第二个之间的协方差是$n_{11}^{-1}+n_{21}^{-1}$。从这里开始的直接方法是确定（4.15）中的log-odds的离散矩阵A，以及相同对比的样本值：即用$n_{i j}$代替（4.15）中的$\theta_{i j}$所产生的表达式。如果$n$是样本值的列向量，那么相关的二次方形式是$n^{\prime} A^{-1} n$。无论选择什么样的对数，都会产生相同的二次方形式，因为它们必须与（4.15）中使用的对数呈线性关系，该方法在线性变换下是不变的。 
                                                                        
&emsp;&emsp;然而，计算涉及大小为$(s-1) \times$ $(t-1)$的正方形矩阵，直接进行方差分析方法并将标准作为残差的平方和来计算可能更好，就需要判断$(r-1)$或$(s-1)$未知数中的线性方程哪一个更小。因此，所涉及的矩阵大小要小得多，但对它们的操作要复杂得多，单个条目的计算要简单得多。
                                                                        
## （五）三维列联表
&emsp;&emsp;接下来考虑$r \times s \times t$表的问题，其中的样本通过三种方式分类：已经提到的两种方式，以及排他性和穷尽性的$C_{1}, C_{2}, \cdots C_{t}$类别。分别用$n_{i j k}$和$\theta_{i j k}$表示类中的数量和概率。如同在二维表中，将情况重新参数化是很方便的，例如，使用
                                                                        
**$$
(5.1)		\theta_{i . .}=\sum_{j, k} \theta_{i j k}, \quad \phi_{i j}=\sum_{\vec{k}} \theta_{i j k} / \theta_{i} . \ldots, \quad \Psi_{i j k}=\theta_{i j k} / \sum_{k} \theta_{i j k}
$$**
                                                                        
&emsp;&emsp;给定$A_{i}$，$\theta_{i} . .$是$A_{i}$的概率，$\phi_{i j}$是$B_{j}$的概率；给定$A_{i}$和$B_{j}$，$\Psi_{i j k}$是$C_{k}$的概率。$\theta_{i j k}=\theta_{i} . . \phi_{i j} \cdot \Psi_{i j k}$。定理2的重复应用表明：如果先验密度与$\prod^{\prime} \theta_{i j k}^{-1}$成正比，那么$\left\{\theta_{i . .}\right\},\left\{\phi_{i j .}\right\}$的后验分布将是独立的。此外，从似然函数的形式来看，$\left\{\phi_{i j}\right\}$的分布将不取决于$n_{i . .}$的分布，并且$\left\{\Psi_{i j k}\right\}$的分布也不取决于$n_{i j .}$的分布。这些结果使我们能够分析其中一个或两个分类变量是非随机的列联表。已经讨论过的方法可以对$\left\{\theta_{i . .}\right\}$和 $\left\{\phi_{i j .}\right\}$进行分析：因此我们对$\left\{\Psi_{i j k}\right\}$进行考虑。
                                                                        
&emsp;&emsp;二维列联表的方法可以扩展到更大的表，基于$\chi^{2}$统计量的经典方法在试图扩展时出现了复杂的问题。一个众所周知的对$2 \times 2 \times 2$表的扩展是Bartlett对三维变量不存在相互作用的定义：
                                                                        
**$$
(5.2)		\frac{\theta_{111} \theta_{221}}{\theta_{211} \theta_{121}}=\frac{\theta_{112} \theta_{222}}{\theta_{212} \theta_{122}}
$$**
                                                                        
&emsp;&emsp;这等同于用$\Psi$代替$\theta$来写，后缀不做改变。如果对(5.2)的两边都取对数，认为
                                                                        
**$$
(5.3)		\begin{array}{r}
\Psi=\left\{\ln \theta_{111}-\ln \theta_{211}-\ln \theta_{121}+\ln \theta_{221}\right\} \\
-\left\{\ln \theta_{112}-\ln \theta_{212}-\ln \theta_{122}+\ln \theta_{222}\right\}
\end{array}
$$**
                                                                        
&emsp;&emsp;参数$\Psi$可以作为$2 \times 2 \times 2$表中三因素互动的定义。这种说法很容易得到证实，因为（5.3）正是$\ln \theta_{i j k}$的方差分析中这种交互作用的形式。因此，通过所述方法，可以用七个参数来研究$2 \times 2 \times 2$表：$\theta_{1} . ., \theta_{\cdot 1}, \theta_{. .1}$。，对应于（4.4）中定义的关联测量的三个双因素互动。(双因素相互作用可以清楚地写成$\phi_{i j .}, \phi_{i . k}$和$\phi_{. j k}$)。  
                                                                        
&emsp;&emsp;与方差分析的联系是很重要的，但除了已经提到的主效应和相互作用外，往往还有其他参数值得关注。例如，上述处理在三个分类中是对称的，但列联表实验的主要目的可能是调查一个分类变量对其他分类变量的依赖性，在这种情况下，显然建议采用非对称性分析。贝叶斯分析通常以一种难以管理的形式提供所有参数的联合后验分布。统计学家要做的是从这个材料中提取可能会引起实验者兴趣的主要特征，或在数据相关的任何决策问题中的价值。因此，在分析的类型上必须有相当大的灵活性，或者说在使用后验分布的缩减上必须有相当大的灵活性。一味地关注方差分析的想法不可能导致许多富有成效的结果。我们现在着手说明在贝叶斯对数框架内可能存在的其他方法，将在$2 \times 2 \times 2$表的框架内进行。  
假设我们感兴趣的是C分类对另外两个分类的依赖性。那么可以认为A和B提供了两个因素，我们希望评估它们对因变量（由C分类代表）的影响。如果A和B是非随机的，那么唯一被定义的概率是
                                                                        
**$$
(5.4)		\begin{array}{ll}
\Psi_{i j 1}=p\left(C_{1} \mid A_{i}, B_{j}\right), & (i, j=1,2)
\end{array}
$$**
                                                                        
&emsp;&emsp;其中，$\Psi_{i j 2}=1-\Psi_{i j 1}$。这四种概率可以用不同的方式进行比较，即使用下式的log-odds：
                                                                        
**$$
(5.5)		\ln p\left(C_{1} \mid A_{i}, B_{j}\right)-\ln p\left(C_{2} \mid A_{i}, B_{j}\right)
$$**
                                                                        
&emsp;&emsp;与式（5.3）进行比较。有一种对于下式是否成立的研究：
                                                                        
**$$
(5.6)		p\left(C_{1} \mid B_{j}, A_{1}\right)=p\left(C_{1} \mid B_{j}, A_{2}\right) \quad(j=1,2)
$$**
                                                                        
&emsp;&emsp;如果成立，那么在给定$B$分类的情况下，$C$分类和$A$分类是独立的。或者说，给定$B_{j}$的情况下，$A_{i}$不提供关于$A$的进一步信息。如果这一点对$B_{1}$和$B_{2}$都成立，那么$A$分类就不提供$B$分类尚未提供的信息。与随机过程理论中所研究的相应属性相类似，将其称为马尔科夫属性是很有用的，其中A、B和C分别指三个时间点，可以方便地认为是过去、现在和未来。  
可以用一对log-odds来检验（5.6）的等式
                                                                        
**$$
(5.7)		\begin{aligned}
\left\{\ln p\left(C_{1} \mid B_{j}, A_{1}\right)\right.&\left.-\ln p\left(C_{2} \mid B_{j}, A_{1}\right)\right\} \\
&-\left\{\ln p\left(C_{1} \mid B_{j}, A_{2}\right)-\ln p\left(C_{2} \mid B_{j}, A_{2}\right)\right\}(j=1,2)
\end{aligned}
$$**
                                                                        
&emsp;&emsp;均值为
                                                                        
**$$
(5.8)		\ln n_{1 j 1}-\ln n_{1 j 2}-\ln n_{2 j 1}+\ln n_{2 j 2}
$$**
                                                                        
&emsp;&emsp;方差为
                                                                        
**$$
(5.9)		n_{1 j 1}^{-1}+n_{1 j 2}^{-1}+n_{2 j 1}^{-1}+n_{2 j 2}^{-1}
$$**
                                                                        
&emsp;&emsp;（5.7)中j=1和2的两个表达式是独立的，因此可以通过将y=1和2的均值的平方之和(5.8)除以方差(5.9)来检验马尔科夫特性，即自由度为2上的$\chi^{2}$。如果后验分布充分集中在零周围，那么人们可能会觉得有理由相信，我们可以在获得马尔科夫特性的基础上继续研究。如果是这样，那么就有可能讨论更简单的条件概率$p\left(C_{1} \mid B_{j}\right)$，否则它就没有定义。要看清上面这句话中的 "足够集中 "的确切含义并不容易，允许偏离马尔科夫属性的程度将取决于使用该属性所产生的分析对其偏离的 "稳健性"。  
                                                                        
&emsp;&emsp;接下来考虑这样一种情况，即人们感兴趣的仍然是$C$对$A$和$B$的依赖性，但其中一个独立的分类，例如$A$是随机的，而另一个$B$不是。那么，除了已经讨论过的条件概率$p\left(C_{k} \mid A_{i}, B_{j}\right)$之外，概率$p\left(C_{k} \mid B_{j}\right)$也是有意义的。在这种情况下，方差分析方法可能会产生误导，因为自然主效应和交互作用在列联表分析中的定义与通常的线性假设情况相当不同。$B$对$C$的主效应是以$A$的平均值来定义的：事实上，
                                                                        
**$$
(5.10)		p\left(C_{k} \mid B_{j}\right)=p\left(C_{k} \mid A_{1}, B_{j}\right) p\left(A_{1} \mid B_{j}\right)+p\left(C_{k} \mid A_{2}, B_{j}\right) p\left(A_{2} \mid B_{j}\right)
$$**
                                                                        
&emsp;&emsp;由于$p\left(A_{1} \mid B_{j}\right) \neq$ $p\left(A_{2} \mid B_{j}\right)$，一般来说，这是每个$A$分类的概率的加权平均值。因此，因素$A$和$B$是 "混杂的"，主要效应很难解释。
* ### 案例分析
下面通过一个例子来解释这个问题，数据如下：
                                                                        
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/5-11.png)
                                                                        
&emsp;&emsp;独立变量C分类为Alive或Dead。随机独立的$A$分类是按性别划分的，而非随机的独立$B$分类是按治疗方式划分的。让我们忽略抽样变化，假设比例（在总共52人中）为概率$\left\{\theta_{i j k}\right\}$。 Male 的$2 \times 2$表显示，通过计算log-odds之差得到的Treated和Dead之间的关联是-5/6。 Female 的$2 \times 2$表也得到同样的数值。因此，以log-odsd的差来衡量治疗对两个性别的好处是一样的。这等于说三因素的相互作用（5.3）为零。但是，如果将男性和女性的两个表格合并起来得出不涉及性别的结果，我们就会得到
                                                                        
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/5-12.png)
                                                                        
&emsp;&emsp;并且，治疗与死亡之间的关联为零：或者说这两种分类是独立的。由于治疗和性别的分布是混在一起的，所以可以通过考虑忽略从属分类为Alive或 Dead 而产生的$2 \times 2$表看出，其结果是
                                                                        
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/5-13.png)

&emsp;&emsp;可以看出，死亡率较高的女性比男性更经常接受治疗。因此，要了解治疗和性别的单独影响是不容易的。在通常的方差分析情况下，有可能在某种程度上分离影响，但这种方法在列联表中是不适用的，因为（5.12）中的关联不是（5.11）中分离的性别表中的线性函数。在(5.12)中，我们使用了$\ln \theta_{-j k}$，而不是像线性假设情况下的$\ln \theta_{1 j k}$和$\ln \theta_{2 j k}$的线性形式（再次比较方程（4.8）和（4.9））。如果每个性别接受治疗的比例在40/52时是相同的，那么对应于（5.12）的表格就会有与（5.12）中单独的$2 \times 2$表差不多的关联度。现在唯一的办法是单独对性别进行讨论。 
                                                                        
&emsp;&emsp;扩展到一般的$r \times s \times t$表是很简单的，所有三个因素的相互作用可以用参数来定义：
                                                                        
**$$
\begin{aligned}
\left\{\ln \theta_{i j k}-\ln \theta_{r j k}-\right.&\left.\ln \theta_{i s k}+\ln \theta_{r s k}\right\} \\
&-\left\{\ln \theta_{i j t}-\ln \theta_{r j t}-\ln \theta_{i s t}+\ln \theta_{r s t}\right\}
\end{aligned}
$$**
                                                                        
&emsp;&emsp;其中，$i<r, j<s, k<t$。如果这些都是零，那么就表示不存在相互作用。(5.4)的消失可以用通常的方法来检验，不过必须注意，(5.14)中$i$、$j$和$k$的不同值的对数值不是独立的，需要找到分散矩阵。  

&emsp;&emsp;给定$B$的情况下，$A$和$C$分类是独立的，马尔科夫属性是基于log-odds

**$$
\begin{aligned}
\left\{\ln p\left(C_{k} \mid B_{j}, A_{i}\right)-\ln p\left(C_{t} \mid B_{j}, A_{i}\right)\right\} \\
\quad-\left\{\ln p\left(C_{k} \mid B_{j}, A_{r}\right)-\ln p\left(C_{t} \mid B_{j}, A_{r}\right)\right\}
\end{aligned}
$$**

&emsp;&emsp;其中，$i<r, k<t$。对于每个$j$来说，有$(r - l)(i - 1)$个不独立的对数，但二次形式产生了一个自由度为$(r - 1) 0 - 1)$的%2$\chi^{2}$统计量。因此，总体的自由度是$(r-1) s(t-1)$。

## 拓展
&emsp;&emsp;本文中使用的先验分布是与矿成正比的特殊分布。但正如第一节所解释的那样，只要先验分布在其包含的形成量上与从与要分析的相同类型的列联表中获得的数据相当，并具有特殊先验，就可以得到结果。假设和真实的两个表可以进行合并，并能够根据合并后的表和特殊先验进行相关分析。

&emsp;&emsp;如果能将分析扩展到更普遍的先验条件，那将是非常理想的。例如，可能存在一个$2 \times 2$的列联表，并且对于$\theta_{i}$的先验知识很少，但对 $\phi_{i j}$的先验知识足够多，这样的知识与在$2 \times 2$表中较少的边缘知识以及较多的内部知识来选择一组边缘总数相差不大。另一个例子是存在$A_{1}, A_{2}$和$A_{3}$的三分类情况时，人们在不清楚$\theta_{2}$和$\theta_{3}$的情况下，认为$\theta_{1}$接近0.20时是可信的。

&emsp;&emsp;另一种需要考虑的先验类型是对$\ln \theta_{i}$之间的关联性进行考虑的先验。本文的所有分析都取决于它们的独立性。例如，在一个$2 \times 2$的表格中，人们可能有先验知识，即两个概率$p\left(A_{1} \mid B_{j}\right)(j=1,2)$的值很接近，那么将其纳入分析中是比较好的方式，比如在社会调查中可能存在两个分类，其中B1对应于 "在家"，B2对应于 "不在家"，在这种情况下对考虑关联性的先验知识的考虑是非常重要的。相关先验的第二个例子是，多叉分布是从分组频率分布（比如柱状图）中产生的，而底层密度的平滑性在相邻的组之间产生了关联性。 
&emsp;&emsp;希望在未来的文章中能够通过log-odds和多变量正态分布对这种情况进行处理。

## 参考文献
[1] BARTLETT, M. S. (1935). Contingency table interactions. J. Roy. Statist. Soc. Suppl. 2 248-252.

[2] BARTLETT, M. S. and KendalL, D. G. (1946). The statistical analysis of varianceheterogeneity and the logarithmic transformation. . Roy. Statist. Soc. Suppl. 

[3] BircH, M. W. (1963). Maximum likelihood in three-way contingency tables. . Roy. Statist. Soc. Ser. B 25 220-233.

[4] DARROCH, J. N. (1962). Interactions in multi-factor contingency tables. . Roy. Statist. Soc. Ser. B 24 251-263.

[5] EDwARDs, A. W. F. (1963). The measure of association for  tables. . Roy. Statist. Soc. Ser. A  109-113.

[6] Good, I. J. (1957). Saddle-point methods for the multinomial distributions. Ann. Math. Statist. 28 861-881.

[7] Good, I. J. (1956). On the estimation of small frequencies in contingency tables. . Roy. Statist. Soc. Ser. B 18 113-124.

[8] Good, I. J. (1963). Maximum entropy for hypothesis formulation, especially for multidimensional contingency tables. Ann. Math. Statist. 34 911-934.

[9] Goodman, L. A. (1963). On Plackett's test for contingency table interactions. J. Roy. Statist. Soc. Ser. B 25 179-188.

[10] JEFFREYs, H. (1961). Theory of Probability. Oxford: Clarendon Press.

[11] KULLBACK, S. (1959). Information Theory and Statistics. Wiley, New York.

[12] KULLBACK, S., KUPPERMAN, M. and Kण, Н. H. (1962). Tests for contingency tables and Markov chains. Technometrics .

[13] LANCASTER, H. O. (1951). Complex contingency tables treated by the partition of . J. Roy. Statist. Soc. Ser. B 13 242-249.

[14] LINDLEY, D. V. (1961). The robustness of interval estimates. Bull. Inst. Internat. Stat. 38 209-220.

[15] LINDLEY, D. V. (1964). Introduction to Probability and Statistics. Cambridge Univ. Press.

[16] PLackETT, R. L. (1962). A note on interactions in contingency tables. . Roy. Statist. Soc. Ser. B  162-166.

[17] Ror, S. N. and KastenBatm, M. A. (1956). On the hypothesis of no 'interaction' in a multi-way contingency table. Ann. Math. Statist. 

[18] ScHerfé, H. (1959). Analysis of Variance. Wiley, New York.

[19] Simpson, E. H. (1951). The interpretation of interaction in contingency tables. . Roy. Statist. Soc. Ser. B 13 238-241.

[20] WoolF, B. (1955). On estimating the relation between blood group and disease. Ann. Human Genetics 19 251-253.
