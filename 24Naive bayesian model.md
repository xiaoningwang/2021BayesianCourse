# 一、朴素贝叶斯模型


## （一）简介

NBM，是Naive Bayesian Model的缩写，即朴素贝叶斯模型。朴素贝叶斯模型，是一种基于贝叶斯定理与特征条件独立假设的分类方法，在贝叶斯算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。朴素贝叶斯模型易于构建，没有复杂的迭代参数估计，可以很容易编码，并且能够非常快地预测，这使得它对于大的数据集特别有用，尽管简单但通常表现出色，并且由于它通常优于更复杂的分类方法而被广泛使用，与决策树模型（Decision Tree Model）同为目前使用最广泛的分类模型之一。
  
## （二）前提假设

朴素贝叶斯分类器基于一个简单假定：假定给定目标值时属性之间相互条件独立-朴素贝叶斯假定(Naive Bayes Assumption)，也叫类条件独立性假定（Class Conditional Independence),即特征向量中一个特征的取值并不影响其他特征的取值。

$$

  <img src="https://latex.codecogs.com/svg.image?p(x_{i}|y)=p(x_{i}|y,\forall&space;x_{j})(i\neq&space;j)" title="p(x_{i}|y)=p(x_{i}|y,\forall x_{j})(i\neq j)" />
$$

x_i是某个样本向量的第i项特征

## （三）算法原理
朴素贝叶斯基于“特征之间是独立的”这一朴素假设，计算在给定数据x的条件下属于类c~k~的概率，即后验概率p(c~k~),并且求使后验概率最大的类c~k~。
根据贝叶斯定理，后验分布（给定数据x的条件下属于类c~k~的概率）

$$

<img src="https://latex.codecogs.com/svg.image?p(c_{k}|x)=\frac{p(c_{k})p(x|c_{k})}{p(x)}\\&space;\indent&space;\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\propto&space;p(c_{k})p(x|c_{k})=p(c_{k})p(x_{1},x_{2},\cdots&space;,x_{n}|c_{k})\\\indent\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,=p(c_{k})p(x_{1}|c_{k})p(x_{2}|c_{k})\cdots&space;p(x_{n}|c_{k})=p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" title="p(c_{k}|x)=\frac{p(c_{k})p(x|c_{k})}{p(x)}\\ \indent \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\propto p(c_{k})p(x|c_{k})=p(c_{k})p(x_{1},x_{2},\cdots ,x_{n}|c_{k})\\\indent\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,=p(c_{k})p(x_{1}|c_{k})p(x_{2}|c_{k})\cdots p(x_{n}|c_{k})=p(c_{k})\coprod_{i=1}^{n}p(x_{i}|c_{k})" />
$$
