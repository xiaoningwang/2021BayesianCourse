14 Hierarchical Linear Model
----
### 郑从云 2019302130046
## （1）简介
分层线性模型（Hierarchical Linear Model、简称 HLM，也被称为mixed-effect model，random-effect models，nested data models或者multilevel linear models）是一种复杂的统计模型。在计量经济学文献中也常常被称为Random-coefficient regression models(Rosenberg, 1973; Longford, 1993)。在某些统计学文献种也被称为Covariance components models(Dempster, Rubin, & Tsutakawa, 1981; Longford, 1987)。现在广泛被使用的名称Hierarchical Linear Model最早出现于1972年Lindley and Smith的论文以及1973年Smith的论文。

分层线性模型 (HLM) 是普通最小二乘法 (OLS) 回归的一种复杂形式，用于在预测变量处于不同层次级别时分析结果变量的方差；例如教室里的学生根据他们共同的老师和共同的教室分享差异。在 HLM 开发之前，通常使用固定参数简单线性回归技术评估分层数据；然而，由于忽略了共享方差，这些技术不足以进行此类分析。 1980 年代早期引入了一种算法来促进不平衡数据的协方差分量估计。 这一发展使得 HLM 广泛应用于多级数据分析（算法的发展参见 Dempster, Laird, & Rubin, 1977）。

HLM 在许多领域都很流行，并且经常用于教育、健康、社会工作和商业部门。 由于这种统计方法的发展同时发生在多个领域，因此它以多种名称为人所知，包括多级、混合级、混合线性、混合效应、随机效应、随机系数（回归）和 （复杂）协方差分量建模（Raudenbush & Bryk，2002）。 这些标签都描述了与 HLM 相同的高级回归技术。 HLM 同时调查分组数据的层次级别内部和之间的关系，从而使其比其他现有分析更有效地解释不同级别变量之间的差异。
## （2）区别
传统线性回归分析，要求因变量呈正态分布，方差齐次、线性、对立等基本假设。且不同层次的数据，不能用于同一模型之中。
HLM 解决的难题：能够正确处理个体效应和组效应之间的关系，社会科学研究必须注意到环境对于个体的影响。从观察效应的角度说，既包括个体效应，也包括组效应。简单的说，这是对回归的回归。
## （3）算法原理
由于预测因子处于多个水平，最低水平的单位或受试者的可交换性假设被打破。经典回归的最简单扩展是为数据中的每个较高层次单元（即教育示例中的班级或抽样示例中的阶层或集群）引入一组指标变量作为预测因子。但这通常会显著增加模型中的参数数量，只有通过进一步的建模（以人口分布的形式）才能对这些参数进行合理的估计。后者本身可以采用简单的可交换的或独立的和相同分布的形式，但也可以合理地考虑在该第二水平上的进一步回归模型，以允许在该级别定义的预测因子。原则上，以这种方式处理的变化水平数量没有限制。贝叶斯方法为处理未知参数的估计提供了现成的指导，尽管计算的复杂性可能相当大，特别是当一个人离开共轭正态规范的领域时。在本章中，我们将简要介绍分层线性模型的广泛主题，强调处理正常模型时使用的一般原则。

估计回归系数的最大收益通常来自于在模型中指定结构。例如，在选举预测问题中，将国家和地区指标变量与定量预测分开进行聚类和建模至关重要。一般来说，当在回归中使用多个预测变量时，应设置它们，以便它们可以分层结构，以便贝叶斯分析可以在汇集有关它们的信息方面做最有效的工作。
## （4）算法模型
我们首先考虑分层回归模型，其中回归系数组是可交换的，并用正态总体分布建模。每个这样的组被称为一批随机效应器，这些效应器具有不同的系数。
* 简单变系数模型

在随机效应或变系数模型的最简单形式中，向量β中包含的所有回归系数都是可交换的，它们的总体分布可以表示为$$ \beta \sim \mathrm{N}\left(1 \alpha, \sigma_{\beta}^{2} I\right) $$其中，$ \alpha $和$ \alpha_\beta $是未知的标量参数，1是1的j×1向量，1=（1，…，1）T。我们使用此向量矩阵表示法，以便于推广系数β的回归模型

* 组内相关性

在刚才描述的变系数模型和组内相关性之间存在着直接的联系。假设数据1，y分为两组，具有多元正态分布：y∼N（α1，∑y），对于$\operatorname {var}(y _{i} ) = n ^ { 2 }$；对于同一批次，$\cot(y_{ i1 }, y_{i2}) = \rho \eta^{2}$，否则为0。（我们使用符号1表示1的×1向量）如果ρ≥0，这相当于model $y∼N（X\beta，\sigma^{2}I）$，其中X为指标变量的X×J矩阵，如果单位i在j中，则$X_{ij}$=1，否则为0，且β具有变系数总体分布。当$\eta^{2} =  \sigma^{2}+ \sigma_{\beta}^{2}$且$\rho= \sigma_{\beta}^{2}/(\sigma^{2}+\sigma_{\beta}^{2})$时，模型的等效性出现，这可以通过推导y的边缘分布（平均值为β）看出。更一般地说，线性回归中的正组内相关性可以通过使用其系数具有总体分布的指示变量来扩大回归，从而纳入变系数模型。

* 混合效应模型

简单变化系数或随机效应模型的一个重要变化是“混合效应模型”，其中β的前J1个分量被分配独立的不适当的先验分布，其余的J2=J-J1个分量是可交换的
具有共同的均值α和标准差σβ。 第一个J1组分是隐式建模为具有无限先验方差的可交换，有时称为固定效应。

* 几组变系数

一般来说，让β的J个分量被分成K个系数簇，簇k具有总体均值$\alpha_k$和标准差$\alpha_{\beta k}$。 通过将系数集群之一的方差设置为∞，可以获得混合效应模型。

* 可交换性

变系数模型的基本特征是分析单位的可交换性是通过对代表总体分组的指标变量进行调节来实现的。 不同的系数允许每个子组具有不同的平均结果水平，并且将这些参数平均为边际分布以得出在同一子组中的单元上观察到的结果之间的相关性（就像在上述简单的组内相关模型中一样）。
## （5）应用案例
假设一个研究人员想对教育数据进行分层线性建模。分层线性建模通常用于监测因变量(如考试分数)与一个或多个独立变量(如学生的背景、他以前的学习成绩等)之间的关系的确定。在分层线性建模中，违反了经典回归理论的假设，即任何一个人的观察结果与任何其他个体的观察结果都没有系统地相关。这个假设被违背了，因为在经典回归理论中应用这个假设会产生有偏差的估计。

分层线性建模又称为多层次建模方法。它允许从事教育数据研究的研究人员系统地询问政策如何影响学生考试成绩的问题。分层线性建模的优点是，当政策相关变量被使用时(如班级规模，或特定改革的引入等)，研究者可以公开地检查对学生考试成绩的影响。

研究人员分两个步骤进行分层线性建模:         
在第一步，研究人员必须对系统中的每个学校(对于教育数据)或其他单位进行单独的分析。
下面的例子可以很好地解释第一步。学生的理科成绩由一组学生水平预测变量回归，如学生的背景和代表学生性别的二进制变量。

在分层线性建模的第一步中，方程在数学上表示如下:
$$(Science)_{ij} = β_{0j} + β_{1j} (SBG)_{ij} + β_2j (Male)_{ij} + e_{ij}。$$
 
$β_0$是在控制SBG(学生背景)和性别的基础上，决定所考虑的学校的表现水平的指标。$β_1$和$β_2$表明在考虑的两个不同变量中，学生之间存在的不平等程度。

在第二步中，从分层线性建模的第一步得到的回归参数成为感兴趣的结果变量。
下面的例子可以很好地解释第二步。结果变量是指对政策变量影响程度的估计。β0j 的公式如下:

$$β_{0j} = Y_{00} + Y_{01(class size)j} + Y_{02(Discipline) j} + U_{01}。$$

$Y_{01}$表示由于班级规模的平均缩小而导致的理科考试成绩的预期收益(或损失)。$Y_{02}$代表学校实施的纪律政策的成效。

根据 Goldstein 在1995年和 Raudenbush 和 Bryk 在1986年的研究，分层线性建模的统计和计算技术涉及到一个多层次的模型到一个单一的。这就是进行回归分析的地方。
## （6）代码与实践
我们使用包含了来自不同学校和不同背景的学生的数学成绩的数据，研究10个学校的学生的数学成绩和家庭作业之间的关系。

```python
# 将学生的数学成绩与家庭作业进行对比，并结合非汇总的 OLS 回归拟合
from statsmodels.formula.api import ols
unpooled_model = ols('math ~ homework', data).fit()
unpooled_est = unpooled_model.params
```
```python
m = unpooled_est['homework']
c = unpooled_est['Intercept']

plt.scatter(data['homework'], data['math'])
plt.xlabel('homework')
plt.ylabel('math')

abline(m, c, linestyle='--', color=red)
```

![](https://files.mdnice.com/user/21245/59ef7d84-e9e0-46ea-ae9c-79af2c85dfab.png)


```python
# 接着绘制了适合每个学校的汇合回归线和非汇合回归适合作为参考。为简单起见，我们使用 OLS 回归。
pooled_est = {} 

def pooled_model(df, ax, grp_id):
    pooled_model = ols('math ~ homework', df).fit()
    pooled_params = pooled_model.params
    
    mp = pooled_params['homework']
    cp = pooled_params['Intercept']
    
    pooled_est[grp_id] = (mp, cp)
    
    plot_data(df, ax, grp_id)
    abline(m, c, ax, linestyle='--', color=red, label='unpooled fit')
    abline(mp, cp, ax, linestyle='--', color=orange, label='pooled fit')
    
facetgrid(pooled_model)
```
![](https://files.mdnice.com/user/21245/c4d8306b-14e3-4ce8-be25-3e275873cbcb.png)

图表显示了不同群体之间关系的变化。我们还注意到，估计值受到一些组中作业完成率高的少数数据点(可能的异常值)的高度影响。

使用 PyMC3 构建我们的贝叶斯层次模型。我们将在组级参数上构造超优先级，以允许模型在组之间共享学生的个人属性。对于这个模型，我们将使用一个随机斜率 β 和截距 α。这意味着它们将随着每个组而变化，而不是以固定的斜率和截距截取整个数据。概率模型的图形表示如下。

![](https://files.mdnice.com/user/21245/cd1fcc04-6d37-408d-b4cf-cce0c5b66e2b.png)

虽然我们在这里通过观察样本的一般分布来选择先验概率，但使用无信息先验会导致类似的结果。下面的代码片段定义了所使用的 PyMC3 模型。
```python
with pm.Model() as model:
    mu_a = pm.Normal('mu_a', mu=40, sigma=50)
    sigma_a = pm.HalfNormal('sigma_a', 50)
    
    mu_b = pm.Normal('mu_b', mu=0, sigma=10)
    sigma_b = pm.HalfNormal('sigma_b', 5)
    
    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_schools)
    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_schools)
    
    eps = pm.HalfCauchy('eps', 5)
    
    y_hat = a[school] + b[school] * homework
    
    y_like = pm.Normal('y_like', mu=y_hat, sigma=eps, observed=math)
```

```python
#从每个组的估计数的后验概率抽样，绘制所有的回归线
def posterior_plot(df, ax, grp_id):
    grp_label = sch_le.transform([grp_id])[0]
    m_p = trace['b'][:,grp_label]
    c_p = trace['a'][:,grp_label]
    
    plot_posterior_regression_lines(m_p, c_p, ax, color='dimgray', alpha=0.3, lw=0.8)
    
    (mp, cp) = pooled_est[grp_id]
    
    plot_data(df, ax, grp_id, zorder=3)
    abline(m, c, ax, linestyle='--', color=red, label='unpooled fit', zorder=4)
    abline(mp, cp, ax, linestyle='--', color=orange, label='pooled fit', zorder=4)
    
facetgrid(posterior_plot)
```
![](https://files.mdnice.com/user/21245/b187ae79-683e-4b2f-b51e-2badaf8dc48e.png)

注意显示负斜率的组周围的一般较高的不确定性。这个模型意味着，我们必须更加小心地对待从模型中得出的对某些群体的决定。注意数据越多，偏差越小，贝叶斯模型就会收敛到OLS模型。

### 参考文献
[1]Gelman, A.; Hill, J. Data Analysis Using Regression and Multilevel/Hierarchical Models. New York: Cambridge University Press. 2007: 235–299. ISBN 978-0-521-68689-1.

[2]Hierarchical Linear Models (S.Raudenbush, A. Bryk) ISBN 076191904X

[3]Woltman H, Feldstain A, MacKay J C, et al. An introduction to hierarchical linear modeling[J]. Tutorials in quantitative methods for psychology, 2012, 8(1): 52-69.
