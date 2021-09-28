# 利用LDF与QDF对Mnist数据集进行分类
环境：python3.6

数据集下载地址：http://yann.lecun.com/exdb/mnist/

从官网下载Mnist数据集放入项目目录中
运行main.py，如果是第一次运行，需要将
rewrite置为True，读取原始的二进制文件并保存。
numOfPC展示的是pca降维的维数，
selectClass是需要分类的类别，本程序支持二分类与多分类问题，
设置完成后，直接运行，即可得到分类结果与正确率。

相关函数说明：

display_img：数据可视化，将numpy数据转化为图像并展示

loadRawToNpy：数据预处理，将二进制的训练集文件进行读取并转化为Npy文件，方便训练读取

dataSelect：筛选需要分类的类别样本

ldf_methods：正则化的ldf分类函数

ldf_pca_methods：pca降维的ldf分类函数

qdf_methods：正则化的qdf分类函数，但实验中发现，其协方差的行列式在正则化后趋于0，导致其对数趋于无穷，因此该方法仅作展示思路，并不运行。

ldf_pca_methods：pca降维的qdf分类函数