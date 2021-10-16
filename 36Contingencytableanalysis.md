# 列联表分析Contingency table analysis

## 引言Introduction
列联表（contingency table），是对于两个或两个以上的属性变量（定类尺度或定序尺度）进行交叉分析时常用的工具，可以通过观察频数、百分比、行边缘分布、列边缘分布等列联表中的数据指标进行基础的描述性统计分析，也可以通过独立性检验对属性变量间的相关性进行检验，并可以应用于众多领域的调查研究，如医学中研究乳腺癌与首次分娩时年龄是否超过30 岁之间有无关联<sup>[1]</sup>、社会学中研究受教育程度与人的宗教信仰程度之间有无关联<sup>[2]</sup>、营养学中研究高盐摄入量与心血管病之间有无关联<sup>[1]</sup>、市场调查中研究消费者喜欢的啤酒类型与饮酒者性别之间的关联性等<sup>[3]</sup>。

## （一）二维列联表的一般形式
### 频数表

假设按两个特性对事物进行研究, 特性 A 有 n 类, 特性 B 有 p 类, 属于 A<sub>i</sub> 和 B<sub>j</sub>, 的个体数目（频数）为 n<sub>ij</sub> (i=1,2,⋯,n;j=1,2,⋯,p), 如下表：
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/%E9%A2%91%E6%95%B0%E8%A1%A8.png)

注：上表中， n 是所有频数的和, <img src="https://latex.codecogs.com/svg.image?n_{i.}=n_{i1}&plus;n_{i2}&plus;\cdots&space;&plus;n_{ip},&space;n_{.j}=n_{1j}&plus;n_{2j}&plus;\cdots&space;&plus;n_{nj}" title="n_{i.}=n_{i1}+n_{i2}+\cdots +n_{ip}, n_{.j}=n_{1j}+n_{2j}+\cdots +n_{nj}" />。
### 频率表

与频数表相似的是频率意义上的列联表，能够更为方便直接地观察各频数之间的关系, 将列联表中每一个元素都除以对应元素的总和n，得到频率表，如下表：
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/%E9%A2%91%E7%8E%87%E8%A1%A8.png)

上表中，令

<img src="https://latex.codecogs.com/svg.image?P=\begin{bmatrix}&space;&p_{11}&space;&space;&p_{12}&space;&space;&\cdots&space;&space;&p_{1p}&space;\\&space;&p_{21}&space;&space;&p_{22}&space;&space;&\cdots&space;&space;&p_{2p}&space;\\&space;&\vdots&space;&space;&space;&\vdots&space;&space;&&space;&space;&\vdots\\&space;&p_{n1}&space;&space;&p_{n2}&space;&space;&\cdots&space;&space;&p_{np}\\\end{bmatrix}" title="P=\begin{bmatrix} &p_{11} &p_{12} &\cdots &p_{1p} \\ &p_{21} &p_{22} &\cdots &p_{2p} \\ &\vdots &\vdots & &\vdots\\ &p_{n1} &p_{n2} &\cdots &p_{np}\\\end{bmatrix}" />

<img src="https://latex.codecogs.com/svg.image?{P_{I}}^{'}=(p_{1.},p_{2.},\cdots&space;,p_{n.})" title="{P_{I}}^{'}=(p_{1.},p_{2.},\cdots ,p_{n.})" />

<img src="https://latex.codecogs.com/svg.image?{P_{J}}^{'}=(p_{.1},p_{.2},\cdots&space;,p_{.P})" title="{P_{J}}^{'}=(p_{.1},p_{.2},\cdots ,p_{.P})" />

<img src="https://latex.codecogs.com/svg.image?1^{'}=(1,1,\cdots&space;,1)" title="1^{'}=(1,1,\cdots ,1)" />

可知，p<sub>ij</sub>是特性A的第i状态与特性B第j状态出现的概率；p<sub>.j</sub>和p<sub>i.</sub>则表示边缘概率。
## （二）独立性检验（test of independence）
为研究列联表中的变量是否具有相关性，可以运用卡方检验进行变量间的独立性检验。若变量间是独立的，则说明他们没有相关性，反之则说明具有一定的相关关系。对列联表的独立性检验可以帮助我们对多变量的交叉分布特征进行探究，为后续的研究提供有力的支撑。

* 独立性检验的步骤如下：

1）	建立原假设H0和备择假设H1

H0：变量A与变量B相互独立

H1：变量A与变量B不独立

2）	建立卡方统计量：<img src="https://latex.codecogs.com/svg.image?\chi&space;^{2}=\sum_{i=1}^{n}\sum_{j=1}^{p}\frac{[n_{ij}-\hat{E}(n_{ij})]^{2}}{\hat{E}(n_{ij})}=n\sum_{i=1}^{n}\sum_{j=1}^{p}\frac{(p_{ij}-p_{i.}p_{.j})^{2}}{p_{i.}p_{.j}}" title="\chi ^{2}=\sum_{i=1}^{n}\sum_{j=1}^{p}\frac{[n_{ij}-\hat{E}(n_{ij})]^{2}}{\hat{E}(n_{ij})}=n\sum_{i=1}^{n}\sum_{j=1}^{p}\frac{(p_{ij}-p_{i.}p_{.j})^{2}}{p_{i.}p_{.j}}" />

3）	查询卡方分布临界值表，对比卡方统计量。若落在拒绝域<img src="https://latex.codecogs.com/svg.image?\chi&space;^{2}>\chi&space;^{2}_{\alpha&space;}[(n-1)(p-1)]" title="\chi ^{2}>\chi ^{2}_{\alpha }[(n-1)(p-1)]" />内，则说明两变量间不独立，具有一定的相关关系，若没有落在拒绝域内，则说明两变量相互独立，不存在相关关系。
当接受原假设H0，即变量A与变量B独立时，卡方统计量渐近服从自由度为<img src="https://latex.codecogs.com/svg.image?(n-1)(p-1)" title="(n-1)(p-1)" />的卡方分布<sup>[4]</sup>。

## （三）列联表中相关程度测量
在通过卡方检验对列联表中的变量进行独立性检验后，若检验结果拒绝原假设H0，即两个变量间存在一定的相关关系，那么接下来可以通过以下三个指标对变量间的相关程度大小进行度量。三个指标分别为<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数、列联相关系数、以及V相关系数，下面将逐个说明。
### 3.1 <img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数
#### 1）计算公式： <img src="https://latex.codecogs.com/svg.image?\varphi=\sqrt{\frac{\chi&space;^{2}}{n}}&space;" title="\varphi=\sqrt{\frac{\chi ^{2}}{n}} " />  

其中，<img src="https://latex.codecogs.com/svg.image?\chi&space;^{2}&space;" title="\chi ^{2} " />由上一节中的公式所得，n为样本容量。

#### 2）说明（以四格表为例）

<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数相关系数通常用于刻画四格表中两个分类变量间的相关程度，一个简单的四格表为例，进一步进行说明：
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/%E5%9B%9B%E6%A0%BC%E8%A1%A8.png)

表中，a、b、c、d为对应的条件频数，可知，当两个分类变量相互独立时，有ad=bc，因此，可以利用ad-bc的差值能够反映变量间相关程度的思想，构造<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数，表达式如下：
<img src="https://latex.codecogs.com/svg.image?\varphi=\sqrt{&space;\frac{\chi&space;^{2}}{n}}=\frac{ad-bc}{\sqrt{(a&plus;b)(c&plus;d)(a&plus;c)(b&plus;d)}}" title="\varphi=\sqrt{ \frac{\chi ^{2}}{n}}=\frac{ad-bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}" />

当 ad = bc 时，<img src="https://latex.codecogs.com/svg.image?\varphi=0&space;" title="\varphi=0" />， 表示两变量相互独立；当 b = c = 0 时，<img src="https://latex.codecogs.com/svg.image?\varphi=1&space;" title="\varphi=1 " />，即两变量完全相关；当 a = c = 0 时，<img src="https://latex.codecogs.com/svg.image?\varphi=-1&space;" title="\varphi=-1 " />，即两变量也是完全相关。对于四格表而言，<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数的取值范围在（0 ，1）内，绝对值越大，说明两变量间的相关程度越高。由于列联表中，变量的位置可以任意变换，因此符号没有实际意义。可以通过<img src="https://latex.codecogs.com/svg.image?\left|\varphi&space;\right|=1" title="\left|\varphi \right|=1" />认为此时两变量是完全相关的。具体两种情况如下图：
![image](https://github.com/huqiwen1023/2021BayesianCourse/blob/main/figure/%E5%AE%8C%E5%85%A8%E7%9B%B8%E5%85%B3%E7%9A%84%E5%9B%9B%E6%A0%BC%E8%A1%A8.png)

#### 3）评价

<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数适用于四格表，当列联表中的行数或者列数大于2时，取值也会随之增大，且没有上限。因此对于这种情况，可以采用后面的列联相关系数进行相关程度的度量。

### 3.2 列联相关系数
#### 1）计算公式：<img src="https://latex.codecogs.com/svg.image?c=\sqrt{\frac{\chi&space;^{2}}{\chi&space;^{2}&plus;n}}" title="c=\sqrt{\frac{\chi ^{2}}{\chi ^{2}+n}}" />

其中，<img src="https://latex.codecogs.com/svg.image?\chi&space;^{2}&space;" title="\chi ^{2} " />由上一节中的公式所得，n为样本容量。

#### 2）说明
列联相关系数又称列联系数，简称c系数。当两个变量相互独立时，c = 0；当两个变量不独立时，c的取值范围在（0 ，1）内，并严格小于1，c的最大取值随列联表中的行数或者列数的增大而增大，可以根据行数和列数进行计算。
例如，对于四格表，c = 0.7071 ； 对于3 * 3列联表，c = 0.8165 ；对于4 * 4列联表，c = 0.87。

#### 3）评价
列联相关系数只能够在列联表的行数和列数都相同时才能够进行对比，不同规模的列联表无法直接比较列联相关系数。其优势在于计算的简便，并且对总体分布没有任何要求，因此适应性较强。

### 3.3 V相关系数
#### 1）计算公式：<img src="https://latex.codecogs.com/svg.image?V=\sqrt{\frac{\chi&space;^{2}}{n\times&space;min{(R-1),(C-1)}}" title="V=\sqrt{\frac{\chi ^{2}}{n\times min{(R-1),(C-1)}}" />

其中，<img src="https://latex.codecogs.com/svg.image?\chi&space;^{2}&space;" title="\chi ^{2} " />由上一节中的公式所得，n为样本容量，R为列联表行数，C为列联表列数。

#### 2）说明
Gramer针对<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数取值没有上限、V相关系数取值小于1，提出了V相关系数。V的取值范围在（0 ，1）内，且当两个变量相互独立时，V = 0；当两个变量完全相关时，V = 1，而当列联表为四格表时，V=<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />。
例如，对于四格表，c = 0.7071 ； 对于3 * 3列联表，c = 0.8165 ；对于4 * 4列联表，c = 0.87。

#### 3）评价
V相关系数建立在<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数和V相关系数的基础之上，更好地综合了变量间相关程度的测度方法。


## （四）列联表的应用
列联表分析作为一种基础性的数据分析方法，列联表分析是非参数检验方法中应用最广泛的方法之一, 它在经济、社会、医学、教育等学领域定性分析中应用的较为广泛, 它是定性资料进行定量分析的基础，下面以市场调查和医学领域为例进行说明。
### 4.1 列联表在市场调查中的应用
由于市场调查一般采用问卷的形式，所收集的数据多为属性数据、分类数据，列联分析对于此类问题有着良好的表现，可以通过独立性检验对变量间是否具有关联性做出判断，并利用对于列联表中变量相关系数的度量进一步进行探究。比如，研究旅游者类型与受教育程度之间的关系<sup>[5]</sup>，旅游者类型分为追求信息者和不追求信息者两类，受教育程度则分为中学肄业、中学毕业、大学肄业、大学毕业、研究生毕业五类，建立这两个变量间的列联表，通过卡方检验可以判断出旅游者类型与受教育程度具有一定的相关性，受教育程度越高，对旅游时的信息收集和追求也会更高，则能够为后续针对性地开展旅游业项目等提供一定的数据参考。

### 4.1 列联表在医学中的应用
在医学中, 对某些疾病成因的探究越来越重要, 鉴于其样本是离散多元的, 列联表在医学, 生物学中应用更加广泛。疾病及其成因的研究属于对分类变量进行的分析研究,对疾病和疾病成因进行分级, 建立列联表的基础上, 对疾病的成因进行分析, 便于对疾病的预防与治疗, 这样列联表在医学中的应用极为广泛。例如，基于列联表对糖尿病与酗酒进行相关性分析<sup>[6]</sup>，通过卡方检验、<img src="https://latex.codecogs.com/svg.image?\varphi&space;" title="\varphi " />相关系数、列联相关系数三种方法进行数据分析，得出了糖尿病与酗酒之间很大概率上存在相关性的结论，帮助我们从孤立的数据本身分析问题的本质, 及时的发现问题和解决问题。
## （五）参考文献

[1]孙尚拱．生物统计学基础．北京：科学出版社，2005．

[2]张淑梅，王睿，曾莉.属性数据分析引论.北京：高等教育出版社，2008.

[3]丹尼斯J.斯威尼，托马斯A.威廉斯，戴维R.安德森.商务与经济统计（精要版）.北京：机械工业出版社，2012.

[4]杨振海，程维虎，张军舰．拟合优度检验．北京：科学出版社，2011.

[5]杨宇. 列联表分析在市场调查中的应用[J]. 管理观察, 2009(13):207-208.

[6]赵鹏辉,崔蕊.列联表检验在疾病成因中的应用[J].大庆师范学院学报,2013,33(03):33-38.
