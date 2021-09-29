# coding=utf-8
'''将二进制格式的MNIST数据集转成.jpg图片格式并保存，图片标签包含在图片名中'''

import os
import joblib
import numpy as np
from util import loadRawToNpy, dataSelect, accuracyDisplay


def ldfMethodsTest(modelDir):
    """ LDF分类测试 """
    # 1.读取数据
    test_images = np.load("./mnistData/test_images.npy")
    test_labels = np.load("./mnistData/test_labels.npy")
    # 2.读取模型
    modelLDF = joblib.load(modelDir)
    classNum = modelLDF['classNum']
    w = modelLDF['w']
    w0 = modelLDF['w0']
    selectClass = modelLDF['selectClass']
    # 筛选数据
    if classNum == 10:
        pass
    else:
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 3.分类，计算判断正确的个数
    accuracyCount = np.zeros(shape=(10,), dtype=np.int32)
    sampleCount = np.zeros(shape=(10,), dtype=np.int32)
    numTest = test_images.shape[1]
    for test_num in range(numTest):
        # 记录当前的样本类别
        sampleCount[test_labels[:, test_num]] = sampleCount[test_labels[:, test_num]] + 1
        test_sample = test_images[:, test_num]
        g = [float('-inf') for i in selectClass]
        for test_class in range(classNum):
            g[test_class] = np.dot(w[:, test_class].T, test_sample)+w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            accuracyCount[test_labels[:, test_num]] = accuracyCount[test_labels[:, test_num]] + 1
    print('\nLDF methods accuracy', np.sum(accuracyCount)/np.sum(sampleCount))
    accuracyDisplay(sampleCount, accuracyCount)
    return


def ldfPcaMethodsTest(modelDir):
    """LDF PCA分类测试 """
    # 1.读取数据
    test_images = np.load("./mnistData/test_images.npy")
    test_labels = np.load("./mnistData/test_labels.npy")
    # 2.读取模型
    modelLDF = joblib.load(modelDir)
    w = modelLDF['w']
    w0 = modelLDF['w0']
    numOfPC = modelLDF['numOfPC']
    classNum = modelLDF['classNum']
    estimator = modelLDF['estimator']
    selectClass = modelLDF['selectClass']
    # 筛选
    if classNum == 10:
        pass
    else:
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        test_images = estimator.transform(test_images.T).T        # 将测试集使用相同的方法进行降维
    # 6.分类，计算准确率
    # 计算判断正确的个数
    accuracyCount = np.zeros(shape=(10,), dtype=np.int32)
    sampleCount = np.zeros(shape=(10,), dtype=np.int32)
    numTest = test_images.shape[1]
    for test_num in range(numTest):
        # 记录当前的样本类别
        sampleCount[test_labels[:, test_num]] = sampleCount[test_labels[:, test_num]] + 1
        test_sample = test_images[:, test_num]
        g = [float('-inf') for i in range(classNum)]
        # 利用判别式进行分类
        for test_class in range(classNum):
            g[test_class] = np.dot(w[:, test_class].T, test_sample)+w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            accuracyCount[test_labels[:, test_num]] = accuracyCount[test_labels[:, test_num]] + 1
    print('\nLDF PCA methods accuracy', np.sum(accuracyCount) / np.sum(sampleCount))
    accuracyDisplay(sampleCount, accuracyCount)
    return


def qdfMethodsTest(modelDir):
    """ QDF分类测试 """
    # 1.读取数据
    test_images = np.load("./mnistData/test_images.npy")
    test_labels = np.load("./mnistData/test_labels.npy")
    # 2.读取模型
    modelLDF = joblib.load(modelDir)
    W = modelLDF['W']
    w = modelLDF['w']
    w0 = modelLDF['w0']
    classNum = modelLDF['classNum']
    selectClass = modelLDF['selectClass']
    # 筛选
    if classNum == 10:
        pass
    else:
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 3.分类，计算准确率
    accuracyCount = np.zeros(shape=(10,), dtype=np.int32)
    sampleCount = np.zeros(shape=(10,), dtype=np.int32)
    numTest = test_images.shape[1]
    for test_num in range(numTest):
        # 记录当前的样本类别
        sampleCount[test_labels[:, test_num]] = sampleCount[test_labels[:, test_num]] + 1
        test_sample = test_images[:, test_num]
        g = [float('-inf') for i in range(classNum)]
        for test_class in range(classNum):
            g[test_class] = test_sample.T @ W[test_class] @ test_sample + \
                            np.dot(w[:, test_class].T, test_sample) + \
                            w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            accuracyCount[test_labels[:, test_num]] = accuracyCount[test_labels[:, test_num]] + 1
    print('\nQDF methods accuracy', np.sum(accuracyCount) / np.sum(sampleCount))
    accuracyDisplay(sampleCount, accuracyCount)
    return

def qdfPcaMethodsTest(modelDir):
    """ QDF PCA 分类测试 """
    # 1.读取数据
    test_images = np.load("./mnistData/test_images.npy")
    test_labels = np.load("./mnistData/test_labels.npy")
    # 2.读取模型
    modelLDF = joblib.load(modelDir)
    W = modelLDF['W']
    w = modelLDF['w']
    w0 = modelLDF['w0']
    numOfPC = modelLDF['numOfPC']
    classNum = modelLDF['classNum']
    estimator = modelLDF['estimator']
    selectClass = modelLDF['selectClass']
    # 筛选
    if classNum == 10:
        pass
    else:
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        # 将测试集使用相同的方法进行降维
        test_images = estimator.transform(test_images.T).T
    # 5.分类，计算准确率
    accuracyCount = np.zeros(shape=(10,), dtype=np.int32)
    sampleCount = np.zeros(shape=(10,), dtype=np.int32)
    numTest = test_images.shape[1]
    for test_num in range(numTest):
        # 记录当前的样本类别
        sampleCount[test_labels[:, test_num]] = sampleCount[test_labels[:, test_num]] + 1
        test_sample = test_images[:, test_num]
        g = [float('-inf') for i in range(classNum)]
        for test_class in range(classNum):
            g[test_class] = test_sample.T @ W[test_class] @ test_sample + \
                            np.dot(w[:, test_class].T, test_sample) + \
                            w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            accuracyCount[test_labels[:, test_num]] = accuracyCount[test_labels[:, test_num]] + 1
    print('\nQDF PCA methods accuracy', np.sum(accuracyCount) / np.sum(sampleCount))
    accuracyDisplay(sampleCount, accuracyCount)
    return

if __name__ == '__main__':

    test_image_file = './mnistData/t10k-images.idx3-ubyte'
    test_label_file = './mnistData/t10k-labels.idx1-ubyte'

    # 第一次使用，需改为True
    rewrite = False
    if rewrite:
        loadRawToNpy(test_image_file, test_label_file)
    modelSaveDir = './modelData'
    ldfMethodsTest(os.path.join(modelSaveDir, "ldfMethods.pickle"))
    ldfPcaMethodsTest(os.path.join(modelSaveDir, "ldfPcaMethods.pickle"))
    # 简单的qdf会出错:math domain error,这是由于伪逆接近于0造成的，因此采用了降维操作
    # qdfMethodsTest(os.path.join(modelSaveDir, "qdfMethods.pickle"))
    qdfPcaMethodsTest(os.path.join(modelSaveDir, "qdfPcaMethods.pickle"))

    print('ok')