## Predicting-Boston-Housing-Prices
Based on Boston data, this project uses a variety of forecasting methods to predict the future house prices of the area.
## 1、问题综述

在这个项目中，你将利用马萨诸塞州波士顿郊区的房屋信息数据训练和测试多个**价格预测模型**，并对模型的性能和预测能力进行测试。项目涉及4个模型如下所示：

-[线性回归预测模型](https://baike.baidu.com/item/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E9%A2%84%E6%B5%8B%E6%B3%95/12609970?fr=aladdin) -[支持向量机预测模型](https://baike.baidu.com/item/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/9683835?fr=aladdin) -[K-Means预测模型](https://baike.baidu.com/item/K%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/15779627?fromtitle=kmeans&fromid=10932719&fr=aladdin) -[决策树预测模型](https://baike.baidu.com/item/%E5%86%B3%E7%AD%96%E6%A0%91/10377049?fr=aladdin)

## 2、数据收集与处理
### 2.1 数据导入
 本项目所使用的波士顿郊区房屋信息数据集被广泛应用于价格预测模型的训练和检验，`python`程序语言中`sklearn`包内置该数据，故我们在使用该数据集的时候，只需要调用具体的`function`即可导入
```
# 从sklearn.datasets包中导入波士顿房价数据读取器模块，如果是第一次使用该包的话，运行pip install sklearn导入
from sklearn.datasets import load_boston

#将读取的房价数据储存在变量boston中，作为整个程序的数据基础
boston = load_boston()
```
该数据集由14项字段组成，其中前13项为对指定房屋的**数值型特征描述**，最后一项为**目标房价**。为了使得大家更明白该数据集的含义，这里对每个英文字段的具体含义进行展示:

![](./images/ziduanming.png)

### 2.2 数据分割
数据分割是指将所有的样本数据按照一定比例随机分为2个组别。其中一组作为训练样本，用来训练模型，让模型能够认识到该组数据中存在的一种规律或规则；另一组作为测试样本，用真实的样本数据验证训练出来的模型是否是准确，为后续的预测模型的应用提供正确性描述。
```
# 从sklearn.cross_validation导入数据分割器
from sklearn.model_selection import train_test_split

#导入 numpy并重命名为np
import numpy as np

import matplotlib.pyplot as plt
#将数据集中的数据转化为python可以更好计算的数组形式
X=boston.data #13项数值型特征
y=boston.target #1项目标房价

#随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

#由于画图时plt.title默认显示是英文，所以当我们需要设置标体为中文时，需要先设置系统环境,使得输出的标题为中文；
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 绘制目标房价频率分布直方图
#导入python绘图专用包，seaborn以及matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# 分析回归目标值的差异。
#定义绘图画布的大小
plt.figure(figsize=(5,4))
#分别绘制整个数据集、训练数据集、测试数据集的房价频率分布直方图
ax = sns.kdeplot(y,color='r',label='Total',lw=3)
ax2 = sns.kdeplot(y_train,label='Train',lw=3)
ax3 = sns.kdeplot(y_test,label='Test',lw=3)
#绘制横纵坐标轴
plt.xlabel('房价',fontdict={'size':12})
plt.ylabel('频率',fontdict={'size':12})
plt.title('波士顿房价频率分布曲线图',fontdict={'size':15})
plt.show()
```
![](./images/波士顿房价样本数据.png)

从上图可以直观得到三个结论，首先是boston房价均值分布在22左右，同时用于训练的数据集和用于测试的数据集在数据特征上和原始数据集保持一致，但是房价的上限和下限之间存在过大的差距。
### 2.3 数据标准化
为了消除不同量级下13个特征指标对目标房价的影响，以及降低房价本身的影响，需要对特征值以及目标值进行标准化处理。
```
# 从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理,得到标准化后的训练集和测试集。
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)
```

## 3、模型应用与分析

### 3.1 线性回归分析
#### 3.1.1 线性回归预测
```
# 从sklearn.linear_model导入LinearRegression。
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器LinearRegression。
lr = LinearRegression()

# 使用训练数据进行参数估计。
lr.fit(X_train, y_train)

# 对测试数据进行回归预测。
lr_y_predict = lr.predict(X_test)
```
#### 3.1.2 线性回归结果分析
通过绘制测试集和预测结果的散点分布图，不难看出，二者之间不重合的点较多，说明二者在预测的结果上存在一定的差距，预测准确性直观上不太高。
```
xx=[]
for i in range(len(y_test)):
    xx.append(i)
type1=plt.scatter(xx,y_test,color='r',label='Test')
type2=plt.scatter(xx,lr_y_predict,color='orange',label='Predict')
plt.xlabel('序号',fontdict={'size':12})
plt.ylabel('房价',fontdict={'size':12})
plt.title('线性回归波士顿房价预测结果对比分析',fontdict={'size':15})
plt.legend((type1,type2),("Test","Predict"))
plt.show()
```
![](./images/线性回归预测结果散点图.png)

为了更直观的看出来预测结果和测试数据集之间的差异，通过统计各个区段之间的房价形成房价频率分布直方图。图图中不难看出，二者仅在有限几个小区段内重叠，这说明预测房价和测试结果之间存在差异较大。
```
#通过绘制频率分布曲线探究房价和测试结果之间存在的差异
ax5 = sns.kdeplot(y_test,label='Test',color='r',lw=3)
ax4 = sns.kdeplot(lr_y_predict,label='predict',color='orange',lw=3)
plt.xlabel('序号',fontdict={'size':12})
plt.ylabel('房价',fontdict={'size':12})
plt.title('波士顿预测房价频率分布曲线图',fontdict={'size':15})
plt.show()
```
![](./images/线性回归预测结果曲线图.png)

为充分量化预测值和测试值之间的差距，评估模型精度，这里导入线性回归模型自带的评估模块，输出评估结果为0.6757955，总的来说模型精度不太高。
```
print('模型性能得分为：',lr.score(X_test,y_test))
```
`线性回归预测模型性能得分为：0.68`

### 3.2 支持向量机分析

### 3.3 K-Means回归分析

### 3.4 决策树分析

### 3.5 模型性能对比分析

