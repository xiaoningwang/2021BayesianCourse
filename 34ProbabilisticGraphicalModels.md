# 34.概率图模型

## Probabilistic Graphical Models

王一依 2019302130032

### 一、基础知识

假设我们的观测对象是高维随机变量$(x_1,x_2,...,x_p)$，对于这样的一个高维随机变量而言，在实际操作过程中，我们往往需要知道的是它的边缘概率$p(x_i)$，以及条件概率$p(x_j|x_i)$。

首先，我们来看下对于高维随机变量（下面以二维为例），两个基本运算法则：

加法法则：$p(x_1)=\int p(x_1,x_2)dx_2$

乘法法则：$p(x_1,x_2)=p(x_1|x_2)p(x_2)=p(x_2|x_1)p(x_1)$

根据两个基本法则可以推出两个常用的法则：

链式法则：$p(x_1,x_2,...,x_p)=p(x_1)\prod_{a}^{b}p(x_i|x_1,x_2,...x_{i-1})$

贝叶斯法则：$p(x_2|x_1)=\frac{p(x_2,x_1)}{\int p(x_2,x_1)x_2}=\frac{p(x_1|x_2)p(x_2)}{\int p(x_2,x_1)x_2}=\frac{p(x_1|x_2)p(x_2)}{\int p(x_1|x_2)p(x_2)x_2}$

### 二、概率图模型的基本概念

**概率图模型（Probabilistic Graphical Model，PGM）**，简称图模型（Graphical Model，GM），是指一种用图结构来描述多元随机变量之间**条件独立性**的概率模型，从而给研究高维空间的概率模型带来了很大的便捷性。

#### 概率图模型的三个基本问题

##### 1.表示问题：对于一个概率模型，如何通过图结构来描述变量之间的依赖关系。

##### 2.学习问题：图模型的学习包括图结构的学习和参数的学习。

##### 3.推断问题：在已知部分变量时，计算其他变量的条件概率分布

![a](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_1.png)

### 三、概率图模型的表示方式：有向图/无向图

概率图模型的表示是指用图结构来描述变量之间的依赖关系，研究如何利用概率网络中的独立性来简化联合概率分布的方法表示。常见的概率图模型可以分为两类：**有向图模型和无向图模型**。

有向图模型使用**有向非循环图（Directed Acyclic Graph，DAG）**来描述变量之间的关系。如果两个节点之间有连边，表示对应的两个变量为因果关系，即不存在其他变量使得这两个节点对应的变量条件独立。

有向图又称**贝叶斯网络**，从模型的条件独立性依赖的个数上看，可以分为单一模型和混合模型，单一模型条件依赖的集合就是单个元素，而混合模型条件依赖的集合则是多个元素。如果给模型加上时间概念，则可以有马尔科夫链和高斯过程等。从空间上，如果随机变量是连续的，则有如高斯贝叶斯网络这样的模型。混合模型加上时间序列，则有隐马尔科夫模型、卡尔曼滤波、粒子滤波等模型。

无向图模型使用无向图（Undirected Graph）来描述变量之间的关系。每条边代表两个变量之间有概率依赖关系，但是并不一定是因果关系。

下图给出了两个代表性图模型（有向图和无向图）的示例，分别表示了四个变量${X_1,X_2,X_3,X_4}$之间的依赖关系。图中带阴影的节点表示可观测到的变量，不带阴影的节点表示隐变量，连边表示两变量间的条件依赖关系。

![b](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_2.png)

##### 1.有向图模型

有向图模型（Directed Graphical Model），也称为贝叶斯网络（Bayesian Network）或信念网络（Belief Network，BN），是一类用有向图来描述随机向量概率分布的模型。常见的有向图模型：很多经典的机器学习模型可以使用有向图模型来描述，比如**朴素贝叶斯分类器**、**隐马尔可夫模型**、**深度信念网络等**。

贝叶斯网络：对于一个$K$维随机向量$X$和一个有 $K$个节点的有向非循环图$G$,$G$中的每个节点都对应一个随机变量，每个连接$e_{ij}$表示两个随机变量 $X_i$和$X_j$之间具有非独立的因果关系。令$X_{\pi k}$表示变量$X_k$的所有父节点变量集和，$P(X_k|X_{\pi k})$表示每个随机变量的局部条件概率分布（Local Conditional Probability Distribution）。如果$X$的联合概率分布可以分解为每个随机变量 $X_k$的局部条件概率的连乘形式，即
$$
p(x)=\prod_{k=1}^{K}p(x_k|x_{\pi k})
$$
那么$(G,K)$构成了一个贝叶斯网络。

贝叶斯网络模型的概率分解，在贝叶斯网络中，变量间存在如下四种关系：

（1）间接因果关系，即图（a）：当$X_2$已知时，$X_1$和$X_3$为条件独立，即$X_1\bot X_3|X_2$。

（2）间接果因关系，即图（b）：当$X_2$已知时，$X_1$和$X_3$为条件独立，即$X_1\bot X_3|X_2$。

（3）共因关系，即图（c）：当$X_2$未知时，$X_1$和$X_3$是不独立的；当$X_2$已知时，$X_1$和$X_3$条件独立，即$X_1\bot X_3|X_2$。

（4）共果关系，即图（d）：当 $X_2$未知时，$X_1$和$X_3$是独立的；当 $X_2$已知时，$X_1$和$X_3$不独立。

![c](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_3.png)

**实例：经典的机器学习模型——朴素贝叶斯分类器**

朴素贝叶斯（Naive Bayes，NB）分类器是一类简单的概率分类器，在强（朴素）独立性假设的条件下运用贝叶斯公式来计算每个类别的条件概率。

给定一个有$M$维特征的样本$x$和类别$y$，类别$y$的条件概率为
$$
p(y|x;\theta)=\frac{p(x_1,...,x_M|y;\theta)p(y;\theta)}{p(x_1,...,x_M)}\\\propto p(x_1,...,x_M|y;\theta)p(y;\theta)
$$
其中$\theta$为概率分布的参数。

在朴素贝叶斯分类器中，假设在给定$Y$的情况下，$X_M$之间是条件独立的。下图给出了朴素贝叶斯分类器的图模型表示。

![d](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_4.png)

条件概率分布$p(y|x)$可以分解为
$$
p(y|x;\theta)\propto p(y|\theta_c)\prod_{m=1}^{M}p(x_m|y;\theta_m)
$$
其中$\theta_c$是$y$的先验分布概率的参数，$\theta_m$是条件概率分布$p(x_m|y;\theta_m)$的参数。若$x_m$为连续值，$p(x_m|y;\theta_m)$可以用高斯分布建模；若$x_m$为连续值，$p(x_m|y;\theta_m)$可以用多项分布建模。

虽然朴素贝叶斯分类器的条件独立性假设太强，但是在实际应用中，朴素贝叶斯分类器在很多任务上也能得到很好的结果，并且模型简单，可以有效防止过拟合。

##### 2.无向图模型

无向图模型，也称为马尔可夫随机场（Markov Random Field，MRF）或马尔可夫网络（Markov Network），是一类用无向图来描述一组具有局部马尔可夫性质的随机向量$X$的联合概率分布的模型。常见的无向图模型有：**最大熵模型**、**条件随机场**、**玻尔兹曼机**、**受限玻尔兹曼机等**。

马尔可夫随机场：对于一个随机向量 $X=[X_1,X_2,...,X_k]^T$和一个有$K$个节点的无向图$G(\nu,\varepsilon)$ （可以存在循环），图 $G$中的节点$k$表示随机变量$X_k$，$1\leq k\leq K$。如果$(G,X)$满足局部马尔可夫性质，即一个变量 $X_k$在给定它的邻居的情况下独立于所有其他变量， $p(x_k|x_{\k})=p(x_k|x_{\aleph(k)})$其中$\aleph(k)$为变量$X_k$的邻居集合，$\k$为除$X_k$外其他变量的集合，那么$(G,X)$就构成了一个马尔可夫场。

##### 3.有向图和无向图之间的转换

有向图和无向图可以相互转换，但将无向图转为有向图通常比较困难。在实际应用中，将有向图转为无向图更加重要，这样可以利用无向图上的精确推断算法，比如联合树算法。

无向图模型可以表示有向图模型无法表示的一些依赖关系，比如循环依赖； 但它不能表示有向图模型能够表示的某些关系，比如因果关系。

以下图（a）中的有向图为例，其联合概率可以分解为
$$
p(x)=p(x_1)p(x_2)p(x_3)p(x_4|x_1,x_2,x_3)
$$
其中$p(x_4|x_1,x_2,x_3)$和四个变量都相关。如果要转化为无向图，需要将这四个变量都归属于一个团中，因此，需要将$x_4$的三个父节点之间都加上连边，如下图（b）所示。这个过程称为道德化。转换后的无向图称为道德图。在道德化的过程中，原来有向图的一些独立性会丢失。

![e](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_5.png)

### 四、学习：图结构的学习和参数的学习

图模型的学习可以分为两部分：一是网络结构学习，即寻找最优的网络结构;二是网络参数估计，即已知网络结构，估计每个条件概率分布的参数。

网络结构学习比较困难，一般是由领域专家来构建。所以只讨论在给定网络结构条件下的参数估计问题。图模型的参数估计问题又分为不包含隐变量时的参数估计问题和包含隐变量时的参数估计问题。

不含隐变量的参数估计：如果图模型中不包含隐变量，即所有变量都是可观测的，那么网络参数一般可以直接通过最大似然来进行估计。

含隐变量的参数估计：如果图模型中包含隐变量，即有部分变量是不可观测的，就需要用EM算法进行参数估计。

### 五、推断：精确推断/近似推断

精确推断包括（变量消除法，信念传播算法），近似推断包括（环路信念传播，变分推断，采样法）

在已知部分变量时算其它变量的后验概率分布。在图模型中，推断（Inference）是指在观测到部分变量$e=\left\{e_1,e_2,...e_M\right\}$时，计算其他变量的某个子集 $q=\left\{q_1,q_2,...q_M\right\}$的条件概率$p(q|e)$。假设一个图模型中，除了变量$e$、$q$外，其余变量表示为$z$。根据贝叶斯公式有
$$
p(q｜e)=\frac{p(q,e)}{p(e)}\\=\frac{\sum_{z}p(q,e,z)}{\sum_{q,z}p(q,e,z)}
$$
因此，图模型的推断问题的关键为求任意一个变量子集的边际概率分布问题。 在图模型中，常用的推断算法可以分为精确推断算法和近似推断算法两类。

精确推断算法是指可以计算出条件概率$p(q|e)$的精确解的算法。在实际应用中，精确推断一般用于结构比较简单的推断问题。当图模型结构比较复杂时，精确推断的计算量会比较大。此外，如果图模型中的变量时连续的，并且其积分函数没有闭式解，那么也无法使用精确推断。因此，在很多情况下也常常采用近似的方法来进行推断。

### 六、贝叶斯网络利用R的简单计算

下图是一个简单的贝叶斯网络的例子。$C$表示通常的感冒，$R$表示流鼻涕，$H$表示头疼，$B$表示接触过有禽流感的家禽，$A$是禽流感，$N$是一般抗生素无效。下面代码构造并生成此例子的贝叶斯网络图。

``` library(bnlearn)
library(Rgraphviz)
bl.av<-model2network('[C][B][A|B][R|C:A][H|C:A][N|A]')
graphviz.plot(bl.av)
```

![Rplot](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_6.png)

定义本例中贝叶斯网络的概率。

``` ny<-c("no","yes")
C<-array(dimnames = list(C=ny),dim=2,c(0.90,0.10))
B<-array(dimnames = list(B=ny),dim=2,c(0.999,0.001))
A<-array(dimnames = list(A=ny,B=ny),dim=c(2,2),c(1,0,0.99,0.01))
R<-array(dimnames = list(R=ny,A=ny,C=ny),dim=c(2,2,2),c(0.95,0.05,0.5,0.5,0.1,0.9,0.02,0.98))
H<-array(dimnames = list(H=ny,A=ny,C=ny),dim=c(2,2,2),c(0.94,0.06,0.03,0.97,0.4,0.6,0.01,0.99))
N<-array(dimnames = list(N=ny,A=ny),dim=c(2,2),c(0.7,0.3,0.001,0.999))
cpts<-list(C=C,B=B,A=A,R=R,H=H,N=N)
bn.av.fit=custom.fit(bl.av,cpts)
bn.av.fit
```

输出三个节点的概率和条件概率。

``` Conditional probability table:
 
     B
A       no  yes
  no  1.00 0.99
  yes 0.00 0.01

  Parameters of node B (multinomial distribution)

Conditional probability table:
 B
   no   yes 
0.999 0.001 

  Parameters of node C (multinomial distribution)

Conditional probability table:
 C
 no yes 
0.9 0.1 

  Parameters of node H (multinomial distribution)

Conditional probability table:
 
, , C = no

     A
H       no  yes
  no  0.94 0.03
  yes 0.06 0.97

, , C = yes

     A
H       no  yes
  no  0.40 0.01
  yes 0.60 0.99


  Parameters of node N (multinomial distribution)

Conditional probability table:
 
     A
N        no   yes
  no  0.700 0.001
  yes 0.300 0.999

  Parameters of node R (multinomial distribution)

Conditional probability table:
 
, , C = no

     A
R       no  yes
  no  0.95 0.50
  yes 0.05 0.50

, , C = yes

     A
R       no  yes
  no  0.10 0.02
  yes 0.90 0.98
```

生成两个节点的条件概率图如下。

![1](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_7.png)

![3](https://github.com/WangYiyi0326/2021BayesianCourse/blob/main/figure/wyy_8.png)


