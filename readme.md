# 利用LDF与QDF对Mnist数据集进行分类
环境：python3.6
## 数据准备
数据集下载地址：http://yann.lecun.com/exdb/mnist/

从官网下载Mnist数据集放入项目目录mnistData中

## 训练
打开train.py，如果是第一次运行，需要将
rewrite置为True，读取原始的二进制文件并保存。
利用selectClass设置选择的分类类别，本程序支持二分类与多分类问题
利用numOfPC设置pca降维的维数，
运行train.py，完成模型的训练
运行main.py，如果是第一次运行，需要将
rewrite置为True，读取原始的二进制Mnist文件并保存。
设置完成后，运行，即可保存训练完成的模型。

## 测试
运行完train.py后，训练时设置的参数将会保存至modleData的pickle文件中
直接运行test.py即可读取配置并得到各方法以及各类的分类准确率。

## 相关文件及函数说明：
### util.py
工具支持包
displayImg：数据可视化，将numpy数据转化为图像并展示

loadRawToNpy：数据预处理，将二进制的训练集文件进行读取并转化为Npy文件，方便训练读取

dataSelect：筛选需要分类的类别样本

accuracyDisplay：将各类别的分类准确率进行打印输出

### train.py
数据训练程序

ldfMethodsTrain：正则化的ldf训练函数

ldfPcaMethodsTrain：pca降维的ldf训练函数

qdfMethodsTrain：正则化的qdf训练函数，但实验中发现，其协方差的行列式在正则化后趋于0，导致其对数趋于无穷，因此该方法仅作展示思路，并不运行。

qdfPcaMethodsTrain：pca降维的qdf训练函数

### test.py
数据测试程序

ldfMethodsTest：正则化的ldf测试函数

ldfPcaMethodsTest：pca降维的ldf测试函数

qdfMethodsTest：正则化的qdf测试函数，但实验中发现，其协方差的行列式在正则化后趋于0，导致其对数趋于无穷，因此该方法仅作展示思路，并不运行。

qdfPcaMethodsTest：pca降维的qdf测试函数

### mnistData
Mnist数据集存放文件夹

### modelData
训练完成的模型存放文件夹