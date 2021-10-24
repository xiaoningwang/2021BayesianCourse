# 观察者偏差 Observer Bias

### 陆晨芸 2019302130043


## 一、定义 Definition

观察者偏差是在观察性和干预性研究中影响评估的一种检测偏差。它发生在当观察者(或调查者)的信念或期望可以影响研究中收集的数据的研究中。它发生在观察和记录信息的过程，这其中也包括与真实值的系统性差异。

许多医疗保健类地观察都有可能出现系统性的变化。例如，在评估医学图像时，一个观察者可能记录异常，而另一个观察者可能不会记录。不同的观察者可能倾向于将同一个测量尺度做出向上或向下地取整。不同的观察者会在颜色变化测试中对颜色有不同的理解。当主观判断是观察的一部分时，观察者之间存在很大的差异性，其中一些差异可能是系统性的，并导致偏见的产生。对于客观数据的观察，如死亡，观察者偏差的风险要低得多。

记录客观数据的偏差可能是由于在使用测量设备或数据源方面的培训不足，或者是没有检查的不良习惯造成的。在记录主观数据中，观察者的个人倾向有可能成为观察者偏见的基础。观察者可能在一定程度上会意识到他们自己对某项研究的偏见，也有可能在记录研究信息时没有意识到那些影响他们决定的因素。

随机对照试验旨在提供最公平的干预测试。然而，如果数据收集过程的任何部分涉及到了观察，那么研究中的测量就会被观察者偏差影响。


## 二、案例 Example

### （一）红线问题

在马萨诸塞州，“红线”是连接剑桥和波士顿的地铁线路。上下班高峰期，红线列车发车间隔为平均7-8分钟运行一趟。

当乘客到达车站时，可以根据站台上的乘客人数估算下一班车到达的时间。如果只有几个人，就推测刚刚错过了地铁，下一班地铁预计要等约7分钟。如果站台上有较多乘客，就估计地铁会很快到达。但是如果有相当多的乘客，则要怀疑列车未能如期运行, 没有按时发车。

接下来将通过贝叶斯估计来预测乘客的等待时间。

### （二）模型

在分析前，需要定一些建模细节。首先将旅客抵达车站视作泊松过程，这意味着假设乘客可能在任何时间等概率到达，乘客有一个未知的到达率$\lambda$，以每分钟到达的乘客计量。因为在很短且相同的时间段内观察乘客，所以假设$\lambda$为常数。

另一方面，列车的到达过程不是泊松的。高峰时间从终点（灰西鲜站）去波士顿的列车每隔7-8分钟发出，但到Kendall广场的时候，列车间隔在3-12分钟内变化。
计算每个工作日下午4点到下午6点Kendall广场站前后到站列车的时间间隔，持续记录5天，每天记录了15 次列车到达。这些分布的差别如图2-1所示, 标为z。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure1.png" title="图2-1" width="456" height="350" />

*图2-1 根据收集到的数据绘制的列车间隔的PMF,以KDE平滑处理（z为实际分布；zb是由 乘客看到列车间隔的偏差分布）*


根据收集到的数据绘制的列车间隔的PMF,以KDE平滑处理（z为实际分布；zb是由乘客看到列车间隔的偏差分布）。

这是下午4点到下午6点在站台记录的列车间隔时间的分布。但是如果随机到达站台（不管列车时刻表），会看到一个与此不同的分布，随机到达的乘客所看到的列车间隔的平均值，比实际的平均值要高一些。这是因为乘客到达的时间间隔更可能是一个较大的区间。考虑一个简单的例子： 假设列车间隔是5分钟或者10分钟（相等的概率）。在这种情况下，列车之间的平 时间是7.5分钟。但乘客更可能在10分钟的时段内到达而不是在5分钟内，事实上前者是后者的两倍。如果对到站旅客进行调查会发现，其中2/3在10分钟的时段内到达，5分钟时段内到达的只有1/3。所以到站乘客观察到的列车间隔平均值是8.33分钟。

这种观察者偏差在许多情况下出现。学生们认为班级比实际的要大是因为他们经常上大课，飞机上的乘客认为飞机比实际更满是因为他们常常乘坐满员的航班。

在每种情况下，实际分布中的值都按照比例被过采样了。例如，在红线上，差距就是两倍大。所以，有了列车间隔的实际分布，就可以计算得到乘客看到的列车间隔分布。BiasPmf进行这个计算：

```python
    def BiasPmf(pmf):
        new_pmf = pmf.Copy()
    
        for x, p in pmf.Items():
          new_pmf.Mult(x, x)
 
        new_pmf.Normalize()
        return new_pmf
```

pmf是实际的分布；new_pmf是偏分布。在循环中，将每个值的概率x乘以观测到的似然度，其正比于x，然后对结果归一化。

### （三）等待时间

等待时间设为y，是乘客到达时刻和下一趟列车到达时刻之间的时间。经过时间设为x，是乘客到达时刻和上一趟列车到达时刻之间的时间。使得zb = x +y。给定zb的分布，可以计算出y的分布。首先先从一个简单的情况开始，然后再一般化。假设如前面的例子，zb为5分钟的概率是1/3, 10分钟的概率就是2/3。如果乘客在5分钟间隔内随机到达，y均匀分布于0至5分钟内。如果乘客在10分钟的间隔到达，y均匀分布于0到10分钟内。所以整体分布是根据每一个间隔的概率加权了的均匀分布的混合分布。

下面的函数将计算Zb的分布和y的分布：

```python
    def PmfOfWaitTime(pmf_zb):
        metapmf = thinkbayes.Pmf()
        for gap, prob in pmf_zb.Items():
            uniform = MakeUniformPmf(0, gap)
            metapmf.Set(uniform, prob)
          
        pmf_y = thinkbayes.MakeMixture(metapmf)
        return pmf_y
```

PmfOfWaitTime通过映射每个均匀分布和其概率来构建一个元Pmf。使用 “混合分布”中的MakeMixture,计算混合分布。

PmfOfWaitTime 还使用了 MakeUniformPmf,定义为：

```python
    def MakeUniformPmf(low, high):
        pmf = thinkbayes.Pmf()
        for x in MakeRange(low=low, high=high):
            pmf.Set(x, 1)
        pmf.Normalize()
        return pmf
```

low和high决定了均匀分布的范围(含两端)。最后，MakeUniformPmf使用了 MakeRange,此处定义为：

```python
    def MakeRange(low, high, skip=10):
        return range(low, high+skip, skip)
```

MakeRange定义了一组等待时间(以秒表示)的可能值。默认情况下，它将范围划分为10秒的时间间隔。

为了封装这些分布的计算过程，创建一个类WaitTimeCalculator：

```python
    class WaitTimeCalculator(object):
    
        def __init__(self, pmf_z):
            self.pmf_z = pmf_z
            self.pmf_zb = BiasPmf(pmf)
            
            self.pmf_y = self.PmfOfWaitTime(self.pmf_zb)
            self.pmf_x = self.pmf_y
```

参数pmf_z是z的非偏差分布。pmf_zb是乘客看到的列车间隔的偏差分布。pmf_y 是等待时间的分布。pmf_x是经过的时间的分布，它和等待时间分布是一样的。想知 道为什么？记得对于一个zp的一个特定值，y的分布是从0到zp均匀的，再考虑到 x = zp - y,因此x的分布也是从0到zp均匀的。

图2-2显示了z、zb和y的分布——基于Red Line网站上收集的数据。
为了解释这些分布，将从Pmfs切换到Cdfs。可知，z的平均值为7.8分钟。zb的平均值为8.8分钟，高出z约13%。 y均值为4.4分钟，是zb均值的一半。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure2.png" title="图2-2" width="456" height="350" />

*图2-2 z, zb,乘客等待时间y的CDF*

### （四）预测等待时间
假设给定z的实际分布，已知乘客到达率$\lambda$是每分钟2名乘客。

在这种情况下可以：

- 1.用z的分布来计算zp的先验分布，乘客所看到的列车间隔分布。

- 2.然后，可以使用乘客数量来估计x的分布，即上一趟火车离开后经过的时间。

- 3.最后，使用关系y = zp - x可得y的分布。

第一步是创建一个WaitTimeCalculator,封装zp, x和y的分布——在考虑乘客的数目之前。

```python
    wtc = WaitTimeCalculator(pmf_z)
```

pmf_z是给定的间隔时间的分布。
接下来的步骤是创建一个ElapsedTimeEstimator，它封装了x的后验分布和y的预测分布。

```python
    ete = ElapsedTimeEstimator (wtc,
                                lam=2.0/60,
                                num_passengers=15)
```

参数是WaitTimeCalculator,乘客到达率lam (表示为乘客人数/秒)和站台上看到的乘客数量(假设是15)。
ElapsedTimeEstimator 的定义：

```python
    def __init__(self, wtc, lam, num_passengers):
        self.prior_x = Elapsed(wtc.pmf_x)
        
        self.post_x = self.prior_x.Copy()
        self.post_x.Update((lam, num_passengers))
        
        self.pmf_y = PredictWaitTime(wtc.pmf_zb, self.post_x)
```

prior_x和posterior_x是经过时间的先验和后验分布。pmf_y是等待时间的预测分布。

ElapsedTimeEstimator 使用 Elapsed 和 PredictWaitTime, 定义如下。

Elapsed是表示x的假想分布的Suite对象。x的先验分布直接由WaitTimeCalculator 得到。然后使用这些数据，包括到达率，lam和站台上乘客的数量计算后验分布。

下面是Elapsed的定义：

```python
    class Elapsed(thinkbayes.Suite):
    
        def Likelihood(self, data, hypo):
            x = hypo
            lam, k = data
            like = thinkbayes.EvalPoissonPmf(lam * x, k)
            return like
```

Likelihood接受一个假设和数据，并计算该假设下数据的似然度。在这个例子里面hypo是上一趟列车后经过的时间，data是一个包括lam和乘客数量元组。
数据的似然度是给定到达率lam下，x时间内k次列车抵达的概率。利用一个泊松分布的PMF来计算它。
最后，PredictWaitTime的定义是：

```python
    def PredictWaitTime(pmf_zb, pmf_x):
        pmf_y = pmf_zb - pmf_x
        RemoveNegatives(pmf_y)    
        return pmf_y
```

pmf_zb是列车间隔的分布情况；pmf_x是经过时间的分布(根据对乘客数量的观察得到)。由于y = zb - x，可以计算：

```python
    pmf_y = pmf_zb - pmf_x
```

减法运算符调用Pmf._sub_,其中列举了所有zb和x对，计算其差，将结果加总到pmf_y。

由此产生的Pmf包括一些显然不可能的负值。例如，如果乘客是在5分钟的间隔期间到达的，乘客的等待时间不可能超过5分钟。RemoveNegatives会移除这些不可能的值并重新归一化。

```python
    def RemoveNegatives(pmf):
        for val in pmf.Values():
            if val < 0:
                pmf.Remove(val)
        pmf.Normalize()
```

图2-3显示了结果。x的先验分布和y一样。x的后验分布表明，看到站台上的15名乘客后，考虑到自上一趟车过后的时间大概是5-10分钟，所以预计下一班列车会在5分钟内到达，置信度为80%。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure3.png" title="图2-3" width="456" height="350" />

*图2-3 x的先验分布和后验分布，以及预测的y值*

### （五）估计到达率
到目前为止的分析基于已知(1)列车间隔的分布(2)乘客到达率的假设。现在,开始处理第二个假设。

假设一个人刚搬到波士顿，他不了解红线地铁的乘客到达率。利用几天上下班时间，就可以做至少是可量化的猜测。只要再花一点心思，他甚至可以定量的估计$\lambda$。每一天他到达站台时，他应该注意时间和到达乘客的数量(如果站台太大，他可以选择一个样本区域)。然后记录自己的等待时间，以及在他等待期间新到站的乘客数量。

5天后，可能得到这样的数据：

| k1 |  y  | k2 |
| -- | --- | -- |
| 17 | 4.6 |  9 |
| 22 | 1.0 |  0 |
| 23 | 1.4 |  4 |
| 18 | 5.4 | 12 |
|  4 | 5.8 | 11 |

其中k1是当他到达时，正在等候的乘客数;y是他的等待时间;k2为等待期间到达的乘客数量。

一个多星期的记录中，他等待时间是18分钟，看到36名乘客到达，因此可以估计，到达率是每分钟2名乘客。就实验来说，这一估计足够了，但为了完整起见，应该计算人的后验分布，然后演示怎么样在后面的分析中利用该分布。

ArrivalRate是个代表人假设的Suite对象。Likelihood接收假设和数据，计算出假设下的数据似然度。

在例子里面，假设是人的取值。数据是y、k数据对，其中y是一个等待时间，k是到达的乘客人数。

```python
    class ArrivalRate(thinkbayes.Suite):
    
        def Likelihood(self, data, hypo):
            lam = hypo
            y, k = data
            like = thinkbayes.EvalPoissonPmf(lam * y, k)
            return like
 ```

ArrivalRateEstimator封装估算$\lambda$的过程。参数passenger_data，是一个包括k1, y, k2元素的元组，具体数据如前文”预测等待时间“所示。

```python
    class ArrivalRateEstimator(object):
    
        def __init__(self, passenger_data):
            low, high = 0, 5
            n = 51
            hypos = numpy.linspace(low, high, n) / 60
            
            self.prior_lam = ArrivalRate(hypos)
            self.post_lam = self.prior_lam.Copy()
            for k1, y, k2 in passenger_data:
            self.post_lam.Update((y, k2))
```

__init__构建假设，这是lam假设值的序列，然后生成先验分布prior_lam。for 循环以数据更新前验概率，产生后验分布post_lam。

图2-4给出了先验和后验分布。正如预期的那样，均值和中位值都在观察得到的值附近，每分钟2名乘客。但不确定后验分布的范围是否是由于$\lambda$基于小样本的原因。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure4.png" title="图2-4" width="456" height="350" />

*图2-4 基于5天乘客数据的lam的前验和后验分布*

### （六）消除不确定性
无论何时，分析中总有一些输入量带来的不确定性，可以通过下面这个步骤将这一因素考虑进来：

- 1.实现基于不确定参数的确定值分析(在本例中是$\lambda$)。

- 2.计算不确定参数的分布。

- 3.对参数的每个值进行分析，并生成一组预测分布。

- 4.使用参数分布所对应的权值计算出预测分布的混合分布。

步骤1和步骤2已经完成。创建一个类WaitMixtureEstimator处理步骤3和步骤4。

```python
    class WaitMixtureEstimator(object):
    
        def __init__(self, wtc, are, num_passengers=15):
            self.metapmf = thinkbayes.Pmf()
            
            for lam, prob in sorted(are.post_lam.Items()):
                ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
                self.metapmf.Set(ete.pmf_y, prob)
                
            self.mixture = thinkbayes.MakeMixture(self.metapmf)
```

wtc是包含zb分布的WaitTimeCalculator实例。are则是包含了lam分布的 ArrivalTimeEstimator实例。第一行创建了一个元Pmf来映射y的可能分布和其概率。对于lam的每一个值，用ElapsedTimeEstimator计算y的相应分布，并将其存储在元Pmf。然后用MakeMixture来计算混合分布。

图2-5显示了结果。背景中的阴影线表示了 y对应于lam每个值的分布，细线表示似然度。粗线是这些分布的混合分布。

在这种情况下，可以用lam的单点估计得到一个非常类似的结果。因此就实用而言，将估计的不确定性包含进来不是必需的。

在一般情况下，如果系统响应是非线性的，那么包括可变性就很重要了。此时，输入的微小变化都会引起输出的较大变化，而本例中，lam的后验变化很小，对于小的扰动，系统的响应近似线性。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure5.png" title="图2-5" width="456" height="350" />

*图2-5 对应了 lam所有可能值的y的预测分布*

### （七）决策分析
建设如果预测乘客等待时间会超过15分钟，则建议乘客乘坐出租车离开。

在这种情况下计算“y超过15分钟”作为num_passengers的函数的概率。在num_passengers的区间上运行“预测等待时间”里的分析方法。但该分析对长时间延误的频次敏感，而由于长时间延误罕见，因此很难估计其时间延误发生频次。

由于只有一周的数据，观察到的最长延误是15分钟，故无法准确估计长时间延误的频次，但可以使用以前的观察来进行至少是粗略的估计。

有一名乘客在一年中看到过由于信号问题、停电、其他车站的警察行动造成的3个长时间延误，所以估计大约每年有3次长时间延误。

但这种看法是偏颇的，应该更倾向于观察长时间延误是因为它们影响了大批乘客。所以，应该把个人的意见作为zb的样本，而不是z的。

通过在一年时间中乘坐红线的220次观察到的间隔时间gap_times 产生了220个列车间隔的样本，并计算它们的Pmf：

```python
    n = 220
    cdf_z = thinkbayes.MakeCdfFromList(gap_times)
    sample_z = cdf_z.Sample(n)
    pmf_z = thinkbayes.MakePmfFromList(sample_z)
```

接下来，偏置pmf_z得到zb的分布情况，抽取样本，然后添加了30分钟、40分钟和50分钟的三次延误(以秒表示)：

```python
    cdf_zp = BiasPmf(pmf_z).MakeCdf()
    sample_zb = cdf_zp.Sample(n) + [1800, 2400, 3000]
```

Cdf.Sample比Pmf.Sample更高效，因而一般会更快地将Pmf转换成Cdf。

接下来，以zb的样本用KDE来估计Pdf，然后将Pdf转换为Pmf：

```python
    pdf_zb = thinkbayes.EstimatedPdf(sample_zb)
    xs = MakeRange(low= 60)
    pmf_zb = pdf_zb.MakePmf(xs)
```

最后，反偏置zb的分布来获得z的分布，用z创建WaitTimeCalculator：

```python
    pmf_z = UnbiasPmf(pmf_zb)
    wtc = WaitTimeCalculator(pmf_z)
```

现在准备进行计算一个长时间等待的概率。

```python
    def ProbLongWait(num_passengers, minutes):
        ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
        cdf_y = ete.pmf_y.MakeCdf()
        prob = 1 - cdf_y.Prob(minutes * 60)
```

根据平台上的乘客人数，ProbLongWait用ElapsedTimeEstimator提取等待时间的分布，并计算等待时间超过minutes的概率。

图2-6显示了结果。当乘客的数目小于20，推断系统运行正常，此时长时间延迟的概率很小。如果有30名乘客，估计自上趟火车已经过了15分钟，这比正常延迟时间长，因此推断出了某些问题，并预期会有更长的延迟。

如果能接受有10%的概率会错过南站列车，又当有不到30名乘客的时候，新到站的乘客应留下来继续等待。但如果发现乘客更多的话，应选择乘坐出租车赶往目的地。

或者，进一步分析可以量化错过南站。

<img src="https://github.com/Lucylcylu/2021BayesianCourse/blob/main/figure/Figure6.png" title="图2-6" width="456" height="350" />

*图2-6 以站台上乘客人数为变量的等待时间超过15分钟的概率函数*


## 三、影响 Impact

Hróbjartsson 和他的同事们通过比较结果评估员不知道干预措施的研究和结果评估员知道干预措施的研究，提出了三份评估观察者偏差影响程度的系统性综述。他们对三种类型的随机对照试验进行了调查，分别是: 二元结果试验、涉及主观测量量表的随机对照试验和时间-事件试验。

### （一）二元结果试验

#### 1、目的

通过试验来评价非盲目的结果评估对二元结果的随机临床试验中估计治疗效果的影响。

#### 2、实验方法

对同一二元结果的盲法和非盲法评估的试验进行系统回顾。计算每项试验的风险比的比率--非盲法评估的风险比相对于盲法评估的相应风险比。风险比的比率<1表示非盲评估者比盲评估者产生了更乐观的效应量。利用反方差随机效应成元分析汇集了每个比率，并用元回归探讨了风险比的比率的变化原因。同时分析盲评者和非盲评者之间的一致率，并计算为中和任何偏见而需要重新分类的患者数量。

#### 3、实验结果

主要分析中共纳入了21项试验（涉及4391名患者）；8项试验提供了个别患者的数据。大多数试验的结果是主观的——比如，在对病人功能的定性评估中，风险比的比率从0.02到14.4不等。风险比的集合比率为0.64（95%置信区间 0.43至0.96），表明非盲目的风险比平均被夸大了36%。同时发现低风险比与结果主观性（P=0.27）；非盲评者对试验的总体参与（P=0.60）；或非盲患者的结果漏洞（P=0.52）的得分之间没有明显的关联。在有数据可查的12项试验中，盲评者和非盲评者在78%的评估中达成一致（四分位数范围64-90%）。与非盲评估者相关的治疗效果被夸大是由每个试验中平均3%的被评估患者的错误分类引起的（1-7%）。

#### 4、结论

平均而言，主观二元结果中的非盲评估者在随机临床试验中产生了严重有偏效应估计，将风险比夸大了36%。这种偏差与盲法和非盲法结果评估者之间的高一致率相一致，也是由少数患者的错误分类所导致的。

### （二）涉及主观测量量表的随机对照试验

#### 1、目的

尽管有偏倚的风险，但临床试验通常是在没有盲目的结果评估者的情况下进行的。拟通过试验来评估非盲目的结果评估对随机临床试验中涉及主观测量量表的估计效果的影响。

#### 2、实验方法

试验对同一测量量表结果的盲法和非盲法评估的随机临床试验进行了系统回顾。在PubMed、EMBASE、PsycINFO、CINAHL、Cochrane中央控制试验登记册、HighWire Press和Google Scholar中搜索相关研究，并计算每个试验的效应量的差异（即非盲评估和盲评估之间的标准化平均差异）。如果效应量的差异小于0，说明非盲评者产生了更乐观的效果估计。实验使用反方差随机效应成元分析汇集了效应量的差异，并使用元回归来确定差异的潜在原因。

#### 3、实验结果

试验审查中包括共24项过去的试验。主要的成元分析包括16项具有主观结果的试验（涉及2854名患者）。当面对非盲评者时，估计的治疗效果被证明更有效（效应量的集合差异-0.23\[95%置信区间 -0.40至-0.06]）。相对而言，非盲评者将集合效应量夸大了68%（95%置信区间 14%至230%）。异质性是中等的（I(2)=46%，P=0.02），并且无法用元回归来解释。

#### 4、结论

试验为具有主观测量量表结果的随机临床试验中的观察者偏差提供了经验性证据。在这类试验中，如果评估人员没有盲目性，就会有高风险导致实质性偏倚。

### （三）时间-事件试验

#### 1、目的

通过试验来评估非盲目的结果评估者对中估计治疗效果的影响。

#### 2、实验方法

对同一时间-事件结果的盲法评者和非盲法评者的随机临床试验进行系统回顾，并比较基于非盲评估和盲评估的风险比。风险比（RHR）<1表示非盲评估者产生了更乐观的效果估计。实验用反方差随机效应元分析对RHRs进行了汇总。

#### 3、实验结果

试验共进行了18项试验。11项有主观结果的试验（涉及1969名患者）RHR为0.88（0.69至1.12），（I2=44%，P=0.06），但由于质的异质性，无条件的汇集是有问题的。四项非典型巨细胞病毒视网膜炎试验比较了实验性口服给药和对照性静脉给药，结果是偏向于对照性干预，RHR为1.33（0.98至1.82）。七项巨细胞病毒性视网膜炎、胫骨骨折和多发性硬化症的试验将实验性干预与标准对照干预（如安慰剂、无治疗或积极对照）进行比较，结果偏向于实验性干预，RHR为0.73（0.57至0.93），表明非盲法危险比平均被夸大了27%（7%至43%）。

#### 4、结论

在具有主观时间-事件结果的随机试验中，缺乏盲评者会导致观察者偏差的高风险。非盲评者通常倾向于实验性干预，将危险比平均夸大了约27%；但在特殊情况下，非盲评者倾向于控制性干预，在相反的方向上诱发了相当程度的观察者偏差。

## 四、预防措施 Preventive steps

一个关键的方法是确保结果评估人员不知道研究参与者的暴露状况。这可以适用于随机对照试验，即个人被分配到一个特定的干预措施，也可以适用于观察性研究，即跟踪不同暴露环境下的研究参与者的进展。实现盲法可能意味着将暴露数据的获取与结果数据分开; 在盲法试验中，整个研究过程中分配应保持未知（除非出于安全原因必须披露）。

方法还可以包括对观察员进行如何记录调查结果的充分培训，在记录开始之前查明任何潜在的冲突，并明确界定收集数据的方法、工具和时限。

另一个预防方面包括培训研究观察员，使其意识到自己的偏见和习惯，以提高准确性。有一项关于血压的研究着眼于减少观察者偏见的训练程序，以及它们能持续多久。研究表明，护士在报告血压时有偏见，要么少报，要么多报；培训确实减少了护士之间的差异，但差异仍然存在，而且也没有因不同的时间点上培训而改变。

虽然观察者的偏见可以减少，但是观察者的偏见很可能会一直存在，研究人员在分析和评估数据时应该意识到这一点。







-------------------------------------------------------------------------------------
### 参考文献 Reference

1.Beevers G et al. ABC of hypertension. Blood pressure measurement. Part I-sphygmomanometry: factors common to all techniques. BMJ 2001;322:981–5

2.Bruce NG et al. Observer bias in blood pressure studies. J Hypertens.1988;6(5):375-80

3.Hróbjartsson A et al. Observer bias in randomised clinical trials with binary outcomes: systematic review of trials with both blinded and non-blinded outcome assessors. BMJ 2012;344:e1119

4.Hróbjartsson A et al. Observer bias in randomized clinical trials with measurement scale outcomes: a systematic review of trials with both blinded and nonblinded assessors. CMAJ 2013;185:E201–11

5.Hróbjartsson A et al. Observer bias in randomized clinical trials with time-to-event outcomes: systematic review of trials with both blinded and non-blinded outcome assessors. Int J Epidemiol 2014;43:937–48

6.Porta M et al. A dictionary of epidemiology. 6th edition. New York: Oxford University Press: 2014

7.Stewart MJ et al. Measurement of blood pressure in the technological age. Br Med Bull. 1994 Apr;50(2):420-42

8.Allen B. Downey. Bayes Stastics in Python: Think Bayes. O’Reilly Media, Inc., 1005 Gravenstein Highway North, Sebastopol, CA 95472.

