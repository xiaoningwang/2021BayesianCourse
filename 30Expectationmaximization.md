<font size =4>

# **EM算法**  
  
## 一、EM算法简介  
  
### 1.简介  
  
EM 算法，全称 Expectation Maximization Algorithm，即期望最大算法。期望最大算法是一种迭代算法，用于含有隐变量（Hidden Variable）的概率参数模型的最大似然估计或极大后验概率估计。  
EM算法是最常见的隐变量估计方法，在机器学习中有极为广泛的用途，例如常被用来学习高斯混合模型（Gaussian mixture model，简称GMM）的参数；隐式马尔科夫算法（HMM）、LDA主题模型的变分推断等等。

### 2.思想  
  
EM 算法的核心思想非常简单，分为两步：Expection-Step 和 Maximization-Step。E-Step也称期望步或E步，主要通过观察数据和现有模型来估计参数，然后用这个估计的参数值来计算似然函数的期望值；而 M-Step，即极大步或M步， 是寻找似然函数最大化时对应的参数。由于算法会保证在每次迭代之后似然函数都会增加，所以函数最终会收敛。

## 二、从极大似然估计到EM算法
  
这里通过一个例子来说明极大似然估计，再引出EM算法。  
假设需要调查某学校学生的身高分布。由于总体数量较大，不可能直接统计，因此首先在学校的学生总体中进行抽样，假设随机抽到了200个学生，设为样本组X。利用这200个学生的身高估计全校学生身高的均值$\mu$和方差$\sigma^{2}$。

### 1.极大似然估计
  
使用极大似然估计前，需假设这个学校的学生身高服从正态分布$N\left(\mu, \sigma^{2}\right)$。这个分布的方差和均值未知。也就是说，要假设抽到每个学生的概率函数为$p(x \mid \theta)$，$p(x \mid \theta)$~$N\left(\mu, \sigma^{2}\right)$，其中参数$\theta=[\mu, \sigma]^{T}$未知。下面对参数$\theta$进行估算。  


#### 1.1.估算参数
  
由于每个样本随机抽取，且相互独立，因此抽到A学生的概率为$p\left(x_{A} \mid \theta\right)$，抽到B学生的概率为$p\left(x_{B} \mid \theta\right)$，抽到X样本组的概率为  
$$
  
L(\theta)=L\left(x_{1}, x_{2}, \cdots, x_{n} ; \theta\right)=\prod_{i=1}^{n} p\left(x_{i} \mid \theta\right), \quad \theta \in \Theta
  
$$  
这个概率反映了在概率密度函数的参数是$\theta$时获得该组样本X的概率。上述L是关于$\theta$的函数，这个函数反映的是在不同的参数$\theta$取值下，取得当前这个样本集的可能性，因此称为参数$\theta$相对于样本 X 的似然函数（likelihood function），记为$L(\theta)$。  
将L取对数，则变为对数似然函数。
$$
  
  H(\theta)=\ln L(\theta)=\ln \prod_{i=1}^{n} p\left(x_{i} \mid \theta\right)=\sum_{i=1}^{n} \ln p\left(x_{i} \mid \theta\right)
  
$$  
对数似然函数更便于后续的计算。  
  
得到上述$L(\theta)$后，求解使似然函数最大的参数$\theta$，也即要求出使某样本X出现的概率最大的参数$\theta$。  


#### 1.2.极大似然总结
  
多数情况下，我们是根据已知条件来推算结果，而极大似然估计是已知结果，寻求使该结果出现的可能性最大的条件，以此作为估计值。  
极大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是已知某个随机样本满足某种概率分布，但是其中具体的参数不清楚，通过若干次试验，观察其结果，利用结果推出参数的大概值。  
极大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率极大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值。  


### 2.从极大似然到EM算法
  
#### 2.1.参数估计
  
EM算法和极大似然估计的前提是一样的，都要假设数据总体的分布，如果不知道数据分布，是无法使用EM算法的。上述问题使用极大似然估计时假设样本来自正态分布总体，但实际问题解决中无法判断总体的数据分布，就无法使用EM算法。  
这里假设200个学生样本中有男生和女生两个群体，但无法确定这两个群体的身高数据分布。首先假设男生和女生的初始分布参数（初始值。然后计算出每个人更可能属于第一个还是第二个正态分布中，这一步称为Expectation；然后问题转变为极大似然问题，用极大似然估计求出两个分布的参数，这一步称为Maximization，此时参数发生改变，每个样本个体所属群体也发生了改变，因此需要再对E步进行调整，再对M步进行调整。如此循环，直至参数不再发生大的变动。  

#### 2.2.EM算法简述
  
上述问题中学生属于男生还是女生这一问题称为隐含参数，女生和男生的身高分布参数称为模型参数。  
EM 算法解决这个的思路是使用启发式的迭代方法，既然无法直接求出模型分布参数，那么可以先猜想隐含参数（EM 算法的 E 步），接着基于观察数据和猜测的隐含参数一起来极大化对数似然，求解模型参数（EM算法的M步)。由于之前的隐含参数是猜测的，所以此时得到的模型参数一般还不是理想结果。基于当前得到的模型参数，继续猜测隐含参数（EM算法的 E 步），然后继续极大化对数似然，求解我们的模型参数（EM算法的M步)。以此类推，不断的迭代下去，直到模型分布参数基本无变化，算法收敛，就找到了合适的模型参数。


###  3.利用EM算法计算贝叶斯边际后验模式
  
在有许多参数的问题中对于联合分布进行常规的近似无法得出理想结果，但对参数的子集进行边际后验估计结果却相对准确。假设参数为$\theta(\gamma,\varphi)$,首先要估计边际后验概率密度$p(\gamma \mid \varphi)$。将$p(\gamma \mid \varphi)$近似为正态分布、t分布或者混合分布之后，其条件分布$\mathrm{p}(\gamma \mid \varphi, \mathrm{\gamma})$将被近似为参数随$\varphi$变化的正态分布（或t分布、混合分布）。  
由EM算法的特性可知，EM算法作为一种迭代方法，在一些常见常见模型中用来寻找边际后验密度$p(\gamma \mid \varphi)$，是非常有用的。对于这些模型来说，直接最大化$p(\gamma \mid \varphi)$比较困难，但用$\mathrm{p}(\gamma \mid \varphi, \mathrm{\gamma})$和$\mathrm{p}(\varphi \mid \varphi, \mathrm{\gamma})$来处理则相对简单得多。  
如果把$\varphi$看作是问题中的参数，而把$\gamma$看作缺失值，那么EM算法就把一个存在已久的处理缺失值的想法变成了正式的算法：先猜测一个参数值，然后，（1）用给定猜测参数的期望值替换缺失值，（2）假设缺失数据等于其估计值，（3）假设新的参数估计正确，重新估计缺失值，（4）重新估计参数。如此循环下去直至收敛。事实上，EM算法比这四个步骤更高效，因为每个缺失值都不是单独估计的；相反，那些需要估计模型参数的缺失数据的函数是联合估计的。  
而且，EM算法的适用广泛，不止能应用在计算边际后验中，对于其他模式也同样适用。

## 三、（广义）EM算法推导
  
在本章中，EM算法计算出边际后验分布概率密度函数$p(\gamma \mid \varphi)$，并对参数γ求平均。可知，EM算法的每一次迭代都会增加对数后验密度的值，直到收敛。  
首先从一个较为简单的公式入手：  
$$
  
\log p(\phi \mid y)=\log p(\gamma, \phi \mid y)-\log p(\gamma \mid \phi, y)
  
$$  
对上述公式两边求期望，将$\gamma$视为随机变量，其分布为$\mathrm{p}(\gamma \mid \phi^{\text {old}}, \mathrm{\gamma})$,$\phi^{\text {old }}$是当前的猜测。上述方程左边不取决于$\gamma$，则对$\gamma$求平均值可以得到  
$$
  
\log p(\phi \mid y)=\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))-\mathrm{E}_{\mathrm{old}}(\log p(\gamma \mid \phi, y))    
  
$$  
假设上述等式为(3.1)。  
其中公式为分布$\mathrm{p}(\gamma \mid \phi^{\text {old }}, \mathrm{y})$下的$\gamma$的平均值。上述公式右边的最后一项，即$\mathrm{E}_{\text {old }}(\log p(\gamma \mid \phi, y))$，在$\phi=\phi^{\text {old }}$时取最大值。另外一项，即预期对数联合后验密度，$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$，在计算中将会反复使用。  
$$
  
\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))=\int(\log p(\gamma, \phi \mid y)) p\left(\gamma \mid \phi^{\mathrm{old}}, y\right) d \gamma
  
$$  
上述表达式在EM文献中被称为$Q(\phi \mid \phi^{\text {old }})$，被视为预期的完全对数似然。  
现在假设有任意一个新的$\phi$值，对于这个$\phi$有
$$
  
\mathrm{E}_{\text {old }}\left(\log p\left(\gamma, \phi^{\text {new }} \mid y\right)\right)>\mathrm{E}_{\text {old }}\left(\log p\left(\gamma, \phi^{\text {old }} \mid y\right)\right)

$$
如果我们用$\phi^{\text {new}}$替换$\phi^{\text {old}}$，那么式(3.1)中等号右侧第一项$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$就会增加，而此时公式第二项却不会增加，因此总值必然增加，则$\log p(\phi^{\text {new}} \mid y)>\log p(\phi^{\text {old}} \mid y)$。这个思想促成了广义EM（GEM）算法：在每次迭代中，确定$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$，把它作为$\phi$的函数来考虑，并将$\phi$更新为一个使这个函数增加的新值。EM算法是一种特殊情况，在这种情况下，选择$\phi$的新值是为了求$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$的最大值，而不仅仅是使这个值增加。EM和GEM算法都具有在每次迭代中增加边际后验密度$p(\phi \mid y)$的特性。  
因为边际后验密度$p(\phi \mid y)$在EM算法的每一步都会增加，而且$Q$函数在每一步都是最大的，所以除了一些特殊情况，EM都会收敛到后验密度的局部模式。(因为GEM算法在每一步都没有最大化，所以它不一定收敛到局部模式。）EM算法收敛到局部模式的速度取决于联合密度$p(\gamma, \phi \mid y)$中关于$phi$的 "信息 "在边际密度$p(\phi \mid y)$中丢失的比例。如果缺失信息的比例很大，收敛的速度就会很慢。 


## 四、EM算法的实现

### 1.EM算法描述
EM算法在算法上可以描述如下。  
（1）首先粗略地给出一个参数估计,$\phi^{0}$。    
（2）对于t = 1, 2, ... :  
&emsp;&emsp;(a)E步。确定预期对数后验密度函数，$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))=\int p\left(\gamma \mid \phi^{\text {old }}, y\right) \log p(\gamma, \phi \mid y) d \gamma$。  
其中期望值为$\gamma$的条件后验分布的平均值，给定当前估计值，$\phi^{old}=\phi^{t-1}$。  
&emsp;&emsp;(b) M步。令$\phi^{t}$是使$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$取得最大值的$\phi$值。而对于GEM算法，只要求$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$的值增加即可，而不一定要求最大值。  
由上述可知，边际后验密度$p(\varphi \mid y)$在EM算法的每一步都会增加，因此，除去一些特殊情况外，该算法会都收敛到后验密度的局部模式。  
   
* 寻找多模式。  
用EM寻找多模式的一个简单方法是在整个参数空间内的许多点同时开始迭代。如果已经得到了几个结果，就可以用正态近似法粗略地比较它们的相对质量，如上节所述。  
* 调试。  
在运行EM算法时，一个有效的调试方法是在每次迭代时计算边际后验密度的对数$\log p(\phi^{t} \mid y)$,确认其是否单调递增。对所有计算后验边际密度比较容易的问题都建议使用这种调试方法。  

### 2.EM算法应用举例
假设有一个均值和方差未知，部分共轭先验分布的正态分布。  
  
假设我们在一个天平上对一个物体称重n次，称重结果$y_1,......y_n$互相独立且服从$N(\mu,\sigma^2)$，µ是物体的真实重量。为简单起见，假设μ的先验分布为$\mathrm{N}\left(\mu_{0}, \tau_{0}^{2}\right)$（$\mu_{0}$和$\tau_{0}$已知），且对数σ的标准非信息均匀先验分布。这些形成了部分共轭的联合先验分布。因为该模型不是完全共轭的，所以$(\mu,\theta)$的联合后验分布没有标准形式，$\mu$的边际后验密度也没有闭式表达。然而，我们可以使用EM算法找到$\mu$的边际后验模式，对$\sigma$进行平均；也就是说，$(\mu,\sigma)$对应于一般概念中的$(\varphi,\gamma)$。  
  
* 联合对数后验密度。  
联合后验密度的对数是
$$
  
\log p(\mu, \sigma \mid y)=-\frac{1}{2 \tau_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}-(n+1) \log \sigma-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{n}\left(y_{i}-\mu\right)^{2}+constant

$$
假设上式为(4.1)。式(4.1)忽略了不随$\mu$和$\sigma^{2}$变化而改变的项。  
  
* E步  
E步中需确定式4.1的期望值，对$\sigma$求平均值，并以当前的猜想和$\mathrm{\mu}_{\mathrm{old}}$及$y$为条件得到：
$$
  
\begin{aligned} \mathrm{E}_{\text {old }} \log p(\mu, \sigma \mid y)=&-\frac{1}{2 \tau_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}-(n+1) \mathrm{E}_{\text {old }}(\log \sigma) \\ &-\frac{1}{2} \mathrm{E}_{\text {old }}\left(\frac{1}{\sigma^{2}}\right) \sum_{i=1}^{n}\left(y_{i}-\mu\right)^{2}+\text { constant} \end{aligned}

$$
设上式为(4.2)。  
现在估计$\mathrm{\mu}_{\mathrm{old}}(log \sigma)$和$\mathrm{E}_{\mathrm{old}}\left(\frac{1}{\sigma^{2}}\right)$。实际上，只需评估后一个表达式即可，因为前一个表达式与式(4.2)中的$\mu$没有关系，因此不会对M步产生影响。  
$\mathrm{E}_{\mathrm{old}}\left(\frac{1}{\sigma^{2}}\right)$可以通过以下方法来估计：$\mu$已知，$\sigma^2$的后验分布是已知均值和未知方差的正态分布，方差和$\chi^2$成反比。
$$
  
\sigma^{2} \mid \mu, y \sim \operatorname{Inv}-\chi^{2}\left(n, \frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu\right)^{2}\right)

$$
那么，$\frac{1}{\sigma^{2}}$的条件后验分布是一个缩放的$\chi^2$ ，并且
$$
  
\mathrm{E}_{\text {old }}\left(\frac{1}{\sigma^{2}}\right)=\mathrm{E}\left(\frac{1}{\sigma^{2}} \mid \mu^{\text {old }}, y\right)=\left(\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu^{\text {old }}\right)^{2}\right)^{-1}

$$
那么就可以将式(4.2)表示为
$$
  
\mathrm{E}_{\mathrm{old}} \log p(\mu, \sigma \mid y)=-\frac{1}{2 \tau_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}-\frac{1}{2}\left(\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu^{\mathrm{old}}\right)^{2}\right)^{-1} \sum_{i=1}^{n}\left(y_{i}-\mu\right)^{2}+const

$$
设上式为(4.3)。    
* M步。  
在M步中，需找到使上述表达式取最大值的$\mu$。因为式(4.3)具有正态对数后验密度的形式，且有先验分布$\mu \sim \mathrm{N}\left(\mu_{0}, \tau_{0}^{2}\right)$和$n$个数据点$y_i$，每个数据点的方差为$\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu^{\text {old }}\right)^{2}$。M步是通过等效后验密度的模式实现的，其公式为
$$

\mu^{\text {new }}=\frac{\frac{1}{\tau_{0}^{2}} \mu_{0}+\frac{n}{\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu^{\mathrm{old}}\right)^{2}} \bar{y}}{\frac{1}{\tau_{0}^{2}}+\frac{n}{\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\mu^{\mathrm{old}}\right)^{2}}}

$$
如果对这个计算进行迭代，那么$\mu$会收敛到$p(\mu \mid y)$的边际模式。

## 五、EM算法拓展
  
基本EM算法的变体和扩展增加了该算法可以应用的问题范围，而且有些版本的收敛速度比基本EM算法快得多。此外，EM算法及其扩展可以通过计算得到的二阶导数矩阵来补充，用来作为模式下渐进方差的估计。以下是EM算法的一些拓展。  
  
### 1.ECM算法

ECM算法是EM算法的一个变体，其中M步被一组条件最大值，或者CM步所取代。这里假设$\phi^t$。E步没有变化：计算预期对数后验密度，在给定当前迭代的$\gamma$的条件后验分布上求平均值。M步骤被一组S条件最大值所取代。在第S个条件最大值中，找到$\phi^{t+s / S}$的值，使所有$\phi$的预期对数后验密度最大，这样$g_{s}(\phi)=g_{s}\left(\phi^{t+(s-1) / S}\right)$,$g_{s}(·)$称为约束函数。最后一个CM步的输出，$\phi^{t+s / S}=\phi^{t+1}$，是ECM算法的下一个迭代。一组约束函数$g_{s}(·),s=1,...,S$，必须满足某些条件，以保证收敛到静止点。最常见的约束函数的选择是参数的第s个子集的指标函数。参数向量$\phi$被划分为S个互不相干的穷举子集$(\phi_1, ..., \phi_s)$，在第S个条件最大化步骤中，除$\phi_s$中的参数外，所有参数都被约束为等于其当前值,在$j \neq k$时，$\phi_{j}^{t+s / S}=\phi_{j}^{t+(s-1) / S}$。基于参数划分的ECM算法是广义EM算法的一个例子。此外，如果CM的每一步都是通过设置一阶导数等于零来实现最大化，那么ECM与EM的共同特性是它将收敛于φ的边际后验分布的局部模式。 由于对数后验密度$p(\phi \mid y)$随着ECM算法的每次迭代而增加的单调递增性仍可用于调试。  
如上一段所述，ECM在每个E步之后都会执行几个CM步。多周期ECM算法在一次迭代中都会执行额外的E-step。例如，可以在每次条件最大化之前执行一个额外的E步。与ECM算法相比，多循环ECM算法的每次迭代都需要更多的计算，但有时可以用较少的总迭代次数达到近似收敛。  
  
### 2.ECME算法

ECME算法是ECM的一个扩展，它用实际对数后验密度$log p(\phi \mid y)$的条件最大化步骤取代了预期对数联合密度$\mathrm{E}_{\mathrm{old}}(\log p(\gamma, \phi \mid y))$的条件最大化步骤。缩写中的最后一个E指的是选择最大化实际对数后验密度或预期对数后验密度。ECME的迭代也会在每次迭代中增加对数后验密度。此外，如果每次条件最大化都将一阶导数设为零，ECME将收敛到一个局部模式。  
ECME对提高收敛速度非常有效，因为实际的边际后验密度是在某些步骤上增加的，而不是在其他参数分布的当前估计上平均的全部后验密度。当较快收敛的数值方法（如牛顿法）直接应用于某些CM步骤的边际后验密度时，收敛速度的提高会非常明显。例如，如果一个CM步骤需要进行一维搜索以求得预期的对数联合后验密度最大值，那么同样的方法可以直接应用于要求得的边际后验密度的对数。   

###  3.AECM算法

通过对在每个条件最大化步骤中对$\gamma$进行不同的交替定义，ECME算法可以被进一步泛化。$\phi$代表缺失数据，且在不同的最大化步骤中有不同的方式来完成数据时，这种泛化是最直接的。在某些问题上，交替进行可以使收敛速度大大加快。AECM算法与EM算法有共同的特性，即随着每一步的后验密度的增加而收敛到一个局部模式。

## 六、补充ECM（SECM）算法

EM算法很有吸引力，因为它通常很容易实现，并且具有稳定和可靠的收敛特性。基本算法及其变体可以被加强，以产生模式下的渐进方差矩阵的估计，形成边际后验密度的近似值。补充EM（SEM）算法和补充ECM（SECM）算法用来自对数联合后验密度和重复EM或ECM步骤的信息来获得参数$\phi$的近似渐近方差矩阵。  
为了描述SEM算法，首先为EM算法隐含参数的映射引入符号$M(\phi)$，即$\phi^{t+1}=M(\phi^t)$。渐进方差矩阵V是
$$
  
V=V_{\text {joint }}+V_{\text {joint }} D_{M}\left(I-D_{M}\right)^{-1}

$$
其中$D_M$是在边际模式$\widehat{\phi }$评估的EM图的雅各布矩阵，$V_{joint}$是基于平均于$\gamma$的联合后验密度的对数的渐进方差矩阵，有
$$
  
V_{\mathrm{joint}}=\left[\left.\mathrm{E}\left(-\frac{d^{2} \log p(\phi, \gamma \mid y)}{d \theta^{2}} \mid \phi, y\right)\right|_{\phi=\hat{\phi}}\right]^{-1}

$$
通常情况下，$V_{joint}$可以通过分析计算，因此只需要$D_M$的值即可。根据以下算法，在每个边际模式下，使用E步和M步对矩阵$D_M$进行数值计算。  
（1）进行EM算法到收敛，得到边际模式，$\widehat{\phi}$。（如果EM的多次运行导致不同的模式，则对每个模式分别应用以下步骤。）  
（2）为SEM算法计算选择一个起始值$\phi^0$，使$\phi^0$不等于任何$\widehat{\phi}$。其中一种可能是使用与原始EM计算相同的起始值。
（3）重复以下步骤，得到一连串的矩阵$R^t$ ,$t = 1, 2, 3, ...，$其中每个元素$r_{i j}^{t}$都收敛到$D_M$的适当元素中。下面描述生成$R^t$的步骤，假设给定当前EM迭代的$\phi^t$。  
&emsp;&emsp;(a)进行通常的E步和M步，输入$\phi^t$，得到$\phi^{t+1}$。  
&emsp;&emsp;(b)对于φ的每个元素，例如$\phi_i$ :  
&emsp;&emsp;&emsp;i.定义$\phi^t(i)$等于$\widehat{\phi}$，但第i个元素除外，它被其当前值$\phi^t_i$取代。  
&emsp;&emsp;&emsp;ii.运行一个E步和一个M步，将$\phi^t(i)$作为参数向量$\phi$的输入值。将这些E步和M步的结果称为$\phi^{t+1}(i)$。Rt的第i行计算,对于任一个j，得到
$$
  
r_{i j}^{t}=\frac{\phi_{j}^{t+1}(i)-\hat{\phi}_{j}}{\phi_{i}^{t}-\hat{\phi}_{i}}

$$ 
当一个元素$r_{ij}$的值不再变化时，它就代表了$D_M$中相应元素的估计值。当某一行的所有元素都收敛时，就不必再对这一行重复最后一步操作。如果$\phi$的某些元素与$\gamma$无关，那么EM将立即收敛到该成分的模式，$D_M$的相应元素等于零。在这种情况下，SEM可以很容易地修改，从而获得方差矩阵。同样的方法可以用来补充ECM算法的渐近方差矩阵的估计。SECM算法是基于以下结果：
$$
  
V=V_{\text {joint }}+V_{\text {joint }}\left(D_{M}^{\mathrm{ECM}}-D_{M}^{\mathrm{CM}}\right)\left(I-D_{M}^{\mathrm{ECM}}\right)^{-1}

$$
$D^{ECM}_M$的定义和计算方式与上述讨论中的$D_M$类似，只是用ECM代替了EM，而且$D^{ECM}_M$是直接应用于$log p(\phi \mid y)$的条件最大化的收敛率。后面的矩阵只取决于$V_{joint}$和$\nabla_{s}=\nabla g_{s}(\hat{\phi}), s=1, \ldots, S$, $s = 1,...,S$，是约束函数$g_s$在$\widehat{\phi}$的梯度。
$$

D_{M}^{\mathrm{CM}}=\prod_{s=1}^{S}\left[\nabla_{s}\left(\nabla_{s}^{T} V_{\mathrm{joint}} \nabla_{s}\right)^{-1} \nabla_{s}^{T} V_{\mathrm{joint}}\right]

$$
这些梯度向量对于直接固定$\phi$的成分的约束来说是很容易计算的。一般来说，SECM似乎需要分析工作来计算$V_{joint}$和$\nabla_{s}$, $s = 1,...,S$，除了应用$D^{ECM}_M$的数值计算外，这些计算中的一部分可以用ECM迭代的结果来进行。 

  
  
  
  
------------------------------------------
  
  
  
参考文献：[1]Bayesian Data Analysis(Third Edition),C-Gelman Andrew,International Standard Book Number-13: 978-1-4398-9820-8 (eBook - PDF).
  
  
  
