# coding=utf-8
import os
import math
import joblib
import numpy as np

from sklearn.decomposition import PCA
from util import loadRawToNpy, dataSelect


def ldfMethodsTrain(modelSaveDir='./modelData', numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    LDF多分类
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("./mnistData/train_images.npy")
    train_labels = np.load("./mnistData/train_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
    # 2.计算协方差矩阵与其逆矩阵
    I = np.eye(numOfPC)
    BETA = 0.01
    # 使用shrinkage解决奇异矩阵问题
    covMat = np.dot(np.cov(train_images), (1-BETA))+np.dot(I, BETA)
    # 求逆
    covMatInv = np.linalg.inv(covMat)
    # 3.计算每一类的均值
    # 初始化index matrix，用于存储每一类的索引值
    indexMat = [[] for i in range(classNum)]
    # 初始化存储μi的矩阵
    mean = np.zeros((numOfPC, classNum))

    for search_class in range(classNum):
        indexMat[search_class] = np.argwhere(train_labels == selectClass[search_class])[:, 1]
        sum=0
        for class_sample in range(len(indexMat[search_class])):
            sum = sum + train_images[:, indexMat[search_class][class_sample]]

        mean[:, search_class] = sum / len(indexMat[search_class])
    # 4.计算相关参数
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        w[:,class_index] = np.dot(covMatInv, mean[:,class_index])
        w0[class_index] = (-0.5)*(mean[:,class_index].T@covMatInv@mean[:,class_index])
    # 5.存储文件权重模型
    # 如果路径不存在则创建
    if not os.path.exists(modelSaveDir):
        os.mkdir(modelSaveDir)
    modelData = {}
    modelData['classNum'] = classNum
    modelData['w'] = w
    modelData['w0'] = w0
    modelData['selectClass'] = selectClass
    joblib.dump(modelData, os.path.join(modelSaveDir, "ldfMethods.pickle"))
    print('ldf methods trained finished.')
    return


def ldfPcaMethodsTrain(modelSaveDir='./modelData', numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    LDF多分类与PCA降维
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("./mnistData/train_images.npy")
    train_labels = np.load("./mnistData/train_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        estimator = PCA(n_components=numOfPC)
        # sklearn的PCA降维以行作为样本，因此需要转置
        train_images = estimator.fit_transform(train_images.T).T  # 将训练集的特征根据构建的PCA对象进行降维
    # 3.计算协方差矩阵与其逆矩阵
    I = np.eye(numOfPC)
    BETA = 0.01
    # 使用shrinkage解决奇异矩阵问题
    covMat = np.dot(np.cov(train_images), (1-BETA))+np.dot(I, BETA)
    # 求逆
    covMatInv = np.linalg.inv(covMat)
    # 4.计算每一类的均值
    # 初始化index matrix，用于存储每一类的索引值
    indexMat = [[] for i in range(classNum)]
    # 初始化存储μi的矩阵
    mean = np.zeros((numOfPC, classNum))

    for search_class in range(classNum):
        indexMat[search_class] = np.argwhere(train_labels == selectClass[search_class])[:, 1]
        sum=0
        for class_sample in range(len(indexMat[search_class])):
            sum = sum + train_images[:, indexMat[search_class][class_sample]]

        mean[:, search_class] = sum / len(indexMat[search_class])
    # 5.计算相关参数
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))                   # 初始化w0参数向量
    for class_index in range(classNum):
        w[:,class_index] = np.dot(covMatInv, mean[:,class_index])
        w0[class_index] = (-0.5)*(mean[:,class_index].T@covMatInv@mean[:,class_index])
    # 6.存储文件权重模型
    # 如果路径不存在则创建
    if not os.path.exists(modelSaveDir):
        os.mkdir(modelSaveDir)
    modelData = {}

    modelData['w'] = w
    modelData['w0'] = w0
    modelData['numOfPC'] = numOfPC
    modelData['classNum'] = classNum
    modelData['estimator'] = estimator
    modelData['selectClass'] = selectClass

    joblib.dump(modelData, os.path.join(modelSaveDir, "ldfPcaMethods.pickle"))
    print('ldf PCA methods trained finished.')
    return


def qdfMethodsTrain(modelSaveDir='./modelData', numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    QDF分类
    :return:
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("./mnistData/train_images.npy")
    train_labels = np.load("./mnistData/train_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
    # 2.读取分类信息，并计算每一类的协方差矩阵与其逆矩阵，以及均值
    # 每一类的协方差矩阵不一样，所以需要先读取分类再进行计算
    # 初始化index matrix，用于存储每一类的索引值
    indexMat = [[] for i in range(classNum)]
    # 存放每一类的特征数据
    classesMat = [[] for i in range(classNum)]
    # 存放每一类的协方差矩阵数据
    covMat = [[] for i in range(classNum)]
    #存放每一类的协方差矩阵的逆矩阵数据
    covMatInv = [[] for i in range(classNum)]
    # 初始化存储μi的矩阵
    mean = np.zeros((numOfPC, classNum))
    I = np.eye(numOfPC)
    BETA = 0.1
    for search_class in range(classNum):
        indexMat[search_class] = np.argwhere(train_labels == selectClass[search_class])[:, 1]

        classesMat[search_class] = train_images[:, indexMat[search_class]]
        # 按行取均值
        mean[:, search_class] = np.mean(classesMat[search_class], axis=1)
        covMat[search_class] = np.dot(np.cov(classesMat[search_class]), (1 - BETA)) + np.dot(I, BETA)
        covMatInv[search_class] = np.linalg.inv(covMat[search_class])

    # 3.计算相关参数
    W = [[] for i in range(classNum)]            # 初始化W参数矩阵
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        W[class_index] = -0.5 * covMatInv[class_index]
        w[:,class_index] = np.dot(covMatInv[class_index], mean[:, class_index])
        w0[class_index] = (-0.5)*(mean[:, class_index].T@covMatInv[class_index]@mean[:, class_index])-\
                          0.5*math.log(np.linalg.det(covMat[class_index]))
    # 4.存储文件权重模型
    # 如果路径不存在则创建
    if not os.path.exists(modelSaveDir):
        os.mkdir(modelSaveDir)
    modelData = {}

    modelData['w'] = w
    modelData['W'] = W
    modelData['w0'] = w0
    modelData['classNum'] = classNum
    modelData['selectClass'] = selectClass

    joblib.dump(modelData, os.path.join(modelSaveDir, "qdfMethods.pickle"))
    print('qdf methods trained finished.')
    return

def qdfPcaMethodsTrain(modelSaveDir='./modelData', numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    QDF分类
    :return:
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("./mnistData/train_images.npy")
    train_labels = np.load("./mnistData/train_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        estimator = PCA(n_components=numOfPC)
        # sklearn的PCA降维以行作为样本，因此需要转置
        train_images = estimator.fit_transform(train_images.T).T  # 将训练集的特征根据构建的PCA对象进行降维
    # 2.读取分类信息，并计算每一类的协方差矩阵与其逆矩阵，以及均值
    # 每一类的协方差矩阵不一样，所以需要先读取分类再进行计算
    # 初始化index matrix，用于存储每一类的索引值
    indexMat = [[] for i in range(classNum)]
    # 存放每一类的特征数据
    classesMat = [[] for i in range(classNum)]
    # 存放每一类的协方差矩阵数据
    covMat = [[] for i in range(classNum)]
    #存放每一类的协方差矩阵的逆矩阵数据
    covMatInv = [[] for i in range(classNum)]
    # 初始化存储μi的矩阵
    mean = np.zeros((numOfPC, classNum))
    # shrinkage和伪逆方法，这种情况下ln(det(covMat{i}))太小，不适合
    # I = np.eye(numOfPC)
    # BETA = 0.01
    for search_class in range(classNum):
        indexMat[search_class] = np.argwhere(train_labels == selectClass[search_class])[:, 1]

        classesMat[search_class] = train_images[:, indexMat[search_class]]
        # 按行取均值
        mean[:, search_class] = np.mean(classesMat[search_class], axis=1)
        # covMat[search_class] = np.dot(np.cov(classesMat[search_class]), (1-BETA))+np.dot(I, BETA)
        covMat[search_class] = np.cov(classesMat[search_class])
        covMatInv[search_class] = np.linalg.inv(covMat[search_class])

    # 3.计算相关参数
    W = [[] for i in range(classNum)]            # 初始化W参数矩阵
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        W[class_index] = -0.5 * covMatInv[class_index]
        w[:,class_index] = np.dot(covMatInv[class_index], mean[:, class_index])
        w0[class_index] = (-0.5)*(mean[:, class_index].T@covMatInv[class_index]@mean[:, class_index])-\
                          0.5*math.log(np.linalg.det(covMat[class_index]))
    # 4.存储文件权重模型
    # 如果路径不存在则创建
    if not os.path.exists(modelSaveDir):
        os.mkdir(modelSaveDir)
    modelData = {}

    modelData['w'] = w
    modelData['W'] = W
    modelData['w0'] = w0
    modelData['numOfPC'] = numOfPC
    modelData['classNum'] = classNum
    modelData['estimator'] = estimator
    modelData['selectClass'] = selectClass

    joblib.dump(modelData, os.path.join(modelSaveDir, "qdfPcaMethods.pickle"))
    print('qdf PCA methods trained finished.')
    return

if __name__ == '__main__':
    """ 训练 """
    # 根目录
    ROOT_DIR = os.path.abspath("")
    # 保存model的文件夹
    MODEL_DIR = os.path.join(ROOT_DIR, "model")
    # 训练集数据存放地址
    train_image_file = './mnistData/train-images.idx3-ubyte'
    train_label_file = './mnistData/train-labels.idx1-ubyte'
    # 第一次使用，需改为True
    rewrite = False
    # 降维维数
    numOfPC = 100
    # 分类数量
    # 多分类
    # selectClass = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 二分类
    selectClass = np.array([5, 8])
    # 将训练集转换为npy文件便于后续读写
    if rewrite:
        loadRawToNpy(train_image_file, train_label_file)
    ldfMethodsTrain(selectClass=selectClass)
    ldfPcaMethodsTrain(numOfPC=numOfPC, selectClass=selectClass)
    # 简单的qdf会出错:math domain error,这是由于伪逆接近于0造成的，因此采用了降维操作
    # qdfMethodsTrain(selectClass=selectClass)
    qdfPcaMethodsTrain(numOfPC=numOfPC, selectClass=selectClass)

    print('ok')