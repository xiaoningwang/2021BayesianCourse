# 一、朴素贝叶斯模型


## （一）简介


NBM，是Naive Bayesian Model的缩写，即朴素贝叶斯模型。朴素贝叶斯模型，是一种基于贝叶斯定理与特征条件独立假设的分类方法，在贝叶斯算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。朴素贝叶斯模型易于构建，没有复杂的迭代参数估计，可以很容易编码，并且能够非常快地预测，这使得它对于大的数据集特别有用，尽管简单但通常表现出色，并且由于它通常优于更复杂的分类方法而被广泛使用，与决策树模型（Decision Tree Model）同为目前使用最广泛的分类模型之一。
  
  
## （二）前提假设


朴素贝叶斯分类器基于一个简单假定：假定给定目标值时属性之间相互条件独立-朴素贝叶斯假定(Naive Bayes Assumption)，也叫类条件独立性假定（Class Conditional Independence),即特征向量中一个特征的取值并不影响其他特征的取值。



  <img src="https://latex.codecogs.com/svg.image?p(x_{i}|y)=p(x_{i}|y,\forall&space;x_{j})(i\neq&space;j)" title="p(x_{i}|y)=p(x_{i}|y,\forall x_{j})(i\neq j)" />



x<sub>i</sub>是某个样本向量的第i项特征

## （三）算法原理

朴素贝叶斯基于“特征之间是独立的”这一朴素假设，计算在给定数据x的条件下属于类c<sub>k</sub>的概率，即后验概率p(c<sub>k</sub>|x),并且求使后验概率最大的类c<sub>k</sub>。
根据贝叶斯定理，后验分布（给定数据x的条件下属于类c<sub>k</sub>的概率）


<img src="https://latex.codecogs.com/svg.image?p(c_{k}|x)=\frac{p(c_{k})p(x|c_{k})}{p(x)}\\&space;\indent&space;\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\propto&space;p(c_{k})p(x|c_{k})=p(c_{k})p(x_{1},x_{2},\cdots&space;,x_{n}|c_{k})\\\indent\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,=p(c_{k})p(x_{1}|c_{k})p(x_{2}|c_{k})\cdots&space;p(x_{n}|c_{k})=p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" title="p(c_{k}|x)=\frac{p(c_{k})p(x|c_{k})}{p(x)}\\ \indent \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\propto p(c_{k})p(x|c_{k})=p(c_{k})p(x_{1},x_{2},\cdots ,x_{n}|c_{k})\\\indent\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,=p(c_{k})p(x_{1}|c_{k})p(x_{2}|c_{k})\cdots p(x_{n}|c_{k})=p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" />


有了上面的式子及假定的p(x<sub>i</sub>|c<sub>k</sub>)的条件分布,给定数据x<sub>1</sub>,x<sub>2</sub>,···,x<sub>n</sub>之后，我们就可以寻求使<img src="https://latex.codecogs.com/svg.image?p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" title="p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" />达到最大的类c<sub>k</sub>。


## （四） 算法流程


朴素贝叶斯算法流程如下：

* 准备工作阶段:


  首先确定特征属性，设x={a<sub>1</sub> ,a<sub>2</sub>,...,a<sub>m</sub>}为待分类项，其中a为x的一个特征属性,类别集合为C={c<sub>1</sub> ,c<sub>2</sub>,...,c<sub>n</sub> }并获取训练样本；

* 分类器训练阶段:

  
  对每个类别计算类先验概率p(c<sub>k</sub>)，对每个特征属性计算所有划分的条件概率密度p(x<sub>1</sub>,x<sub>2</sub>,···,x<sub>n</sub>|c<sub>k</sub>)

* 应用阶段:


  对每个类别计算后验概率p(c<sub>k</sub>)p(x<sub>i</sub>|c<sub>k</sub>)，选取后验概率的最大值所对应的类作为X的分类结果。

## （五）常见模型及应用

朴素贝叶斯方法通常在给定类别（比如c<sub>k</sub>)之后假定了它们的条件分布p(x<sub>i</sub>|c<sub>k</sub>)的类型，比如正态分布、多项分布或Bernoulli分布等。


* 高斯朴素贝叶斯（Gaussian Naive Bayes）：

  适用于连续变量，并且其各个特征 𝑥<sub>i</sub> 在各个类c<sub>k</sub>下是服从正态分布的，那么在计算P(x<sub>i</sub>|y)的时候可以直接使用高斯分布的概率密度公式,即：
  
&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<img src="https://latex.codecogs.com/svg.image?P(x_{i}|y)=\frac{1}{\sqrt{2\pi&space;}\sigma_{y}&space;}e^{-\frac{(x-\mu_{y}&space;)^{2}}{2\sigma_{y}&space;^{2}}}" title="P(x_{i}|y)=\frac{1}{\sqrt{2\pi }\sigma_{y} }e^{-\frac{(x-\mu_{y} )^{2}}{2\sigma_{y} ^{2}}}" />

&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;𝜇<sub>𝑦</sub>：在类别为𝑦的样本中，特征𝑥<sub>𝑖</sub>的均值
  
&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;𝜎<sub>𝑦</sub>：在类别为𝑦的样本中，特征𝑥<sub>𝑖</sub>的标准差

因此只需要计算出各个类别中此特征项划分的各个均值和标准差，就能使用高斯朴素贝叶斯对样本类别进行预测。

* 多项式朴素贝叶斯（Multinomial Naive Bayes）：

  多项朴素贝叶斯实现了对多项式分布数据的朴素贝叶斯算法，可用于属性分类的问题。比如一个评论是正面、负面还是中性等；一个文档是属于体育、科技、民生、新闻等，可以计算出一篇文档为某些类别的概率，最大概率的类型就是该文档的类别，预测变量多为某种特征的频数。
  多项式模型在计算先验概率P(c<sub>k</sub>)和条件概率P(x<sub>i</sub>|c<sub>k</sub>)时，会做一些平滑处理，具体公式为：
  
 &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.image?p(c_{k})=\frac{N_{c_{k}}&plus;\alpha&space;}{N&plus;k&space;\alpha&space;}" title="p(c_{k})=\frac{N_{c_{k}}+\alpha }{N+k \alpha }" />
 
 &nbsp; &nbsp; &nbsp;&nbsp;N:总的样本个数 k:总的类别个数 N<sub>c<sub>k</sub></sub>:类别为y<sub>k</sub>的样本个数 α:平滑值
 
 &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.image?p(x_{i}|c_{k})=\frac{N_{c_{k}},x_{i}&plus;\alpha&space;}{N_{c_{k}}&plus;n\alpha&space;}" title="p(x_{i}|c_{k})=\frac{N_{c_{k}},x_{i}+\alpha }{N_{c_{k}}+n\alpha }" />
 
&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; N<sub>c<sub>k</sub></sub>:类别为y<sub>k</sub>的样本个数  n:特征的维数  N<sub>y<sub>k</sub>,xi</sub>:类别为y<sub>k</sub>的样本中，第i维特征的值是xi的样本个数    α:平滑值

&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;  当α=1时，称作Laplace平滑，当0<α<1时，称作Lidstone平滑，α=0时,称作不做平滑，通常α取值为1.


* 伯努利朴素贝叶斯（Bernoulli Naive Bayes）：

  使用伯努利贝叶斯方法的预测变量通常是二分变量或布尔变量，符合0-1分布。在计算p(x<sub>i</sub>|y）的时候可以直接使用伯努利分布的概率公式：
  
&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; <img src="https://latex.codecogs.com/svg.image?p(x_{i}|y)=p(1|y)x_{i}&plus;(1-p(1|y))(1-x_{i})" title="p(x_{i}|y)=p(1|y)x_{i}+(1-p(1|y))(1-x_{i})" />

&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; 1表示成功，其成功的概率为p;0表示失败，失败的概率为1-p


## （六）案例及Python实践

* 下面为卷尾花数据集构建高斯朴素贝叶斯分类模型，进行模型训练以及对训练模型进行预测的Python代码如下所示：

  from sklearn import datasets

  from sklearn import naive_bayes

  from sklearn.metrics import classification_report

  from sklearn.model_selection import train_test_split

  bayes = naive_bayes.GaussianNB()

  iris = datasets.load_iris()

  X = iris.data

  Y = iris.target

  x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)

  bayes.fit(x_train,y_train)

  print("贝叶斯模型训练集的准确率：%.3f"%bayes.score(x_train,y_train))

  print("贝叶斯模型测试集的准确率：%.3f"%bayes.score(x_test,y_test))

  target_names = ['setosa','versicolor','virginica']

  y_hat = bayes.predict(x_test)

  print(classification_report(y_test,y_hat,target_names = target_names))


* 运行及结果如图所示

![image](https://github.com/luxinyu-xg/2021BayesianCourse/blob/main/figure/%E5%9B%BE%E7%89%871.png)

### 参考文献
[1]石川，王啸，胡琳梅.数据科学导论。清华大学出版社，2021

[2]吴喜之，贝叶斯数据分析，中国人民大学出版社，2020

