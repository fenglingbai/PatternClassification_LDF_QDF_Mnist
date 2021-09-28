# coding=utf-8
'''将二进制格式的MNIST数据集转成.jpg图片格式并保存，图片标签包含在图片名中'''

import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def display_img(img_array, img_class, img_id, save_dir, img_save=False):
    """数据可视化"""
    # 变换至0~255
    img_array = img_array - min(img_array)
    img_array = img_array * (255 / max(img_array) )
    img_array = img_array.astype(np.uint8).reshape(28, 28, 1)

    plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.close('all')

    if img_save:
        save_name = os.path.join(save_dir, '{}_{}.jpg'.format(img_id, img_class))
        cv2.imwrite(save_name, img_array)
    return

def loadRawToNpy(image_file, label_file):
    """
    读取原始数据文件，并保存为npy文件
    """
    if 'train' in os.path.basename(image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'

    with open(image_file, 'rb') as f1:
        image_file = f1.read()
    with open(label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    image_all = []
    label_all = []
    for i in range(num_file):
        label_all.append([label_file[i]])
        image_list = [item for item in image_file[i * 784:i * 784 + 784]]
        # 注意归一化并转换为float64，防止精度不够造成的误判
        image_all.append(np.array(image_list, dtype=np.float64)/255)
    # 每一列是一个样本
    label_all = np.array(label_all).T
    image_all = np.array(image_all).T

    np.save(prefix+'_images', image_all)
    np.save(prefix+'_labels', label_all)
    return

def dataSelect(imageData, labelData, selectClass):
    """
    根据selectClass对数据进行筛选
    """
    select_list = []
    for sample_id in range(imageData.shape[1]):
        if labelData[0, sample_id] in selectClass:
            select_list.append(sample_id)
    return imageData[:, select_list], labelData[:, select_list]


def ldf_methods(numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    LDF多分类
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
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

    for serach_class in range(classNum):
        indexMat[serach_class] = np.argwhere(train_labels == selectClass[serach_class])[:, 1]
        sum=0
        for class_sample in range(len(indexMat[serach_class])):
            sum = sum + train_images[:, indexMat[serach_class][class_sample]]

        mean[:, serach_class] = sum/len(indexMat[serach_class])
    # 4.计算相关参数
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        w[:,class_index] = np.dot(covMatInv, mean[:,class_index])
        w0[class_index] = (-0.5)*(mean[:,class_index].T@covMatInv@mean[:,class_index])
    # 5.分类，计算准确率
    count = 0               # 计算判断正确的个数
    numTest = test_images.shape[1]

    for test_num in range(numTest):
        test_sample = test_images[:, test_num][:, np.newaxis]
        g = [float('-inf') for i in selectClass]
        for test_class in range(classNum):
            g[test_class] = np.dot(w[:, test_class].T, test_sample)+w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            count = count+1
    print('LDF methods', count/numTest)
    return


def ldf_pca_methods(numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    LDF多分类与PCA降维
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        estimator = PCA(n_components=numOfPC)
        # sklearn的PCA降维以行作为样本，因此需要转置
        train_images = estimator.fit_transform(train_images.T).T  # 将训练集的特征根据构建的PCA对象进行降维
        test_images = estimator.transform(test_images.T).T        # 将测试集使用相同的方法进行降维
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

    for serach_class in range(classNum):
        indexMat[serach_class] = np.argwhere(train_labels == selectClass[serach_class])[:, 1]
        sum=0
        for class_sample in range(len(indexMat[serach_class])):
            sum = sum + train_images[:, indexMat[serach_class][class_sample]]

        mean[:, serach_class] = sum/len(indexMat[serach_class])
    # 5.计算相关参数
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))                   # 初始化w0参数向量
    for class_index in range(classNum):
        w[:,class_index] = np.dot(covMatInv, mean[:,class_index])
        w0[class_index] = (-0.5)*(mean[:,class_index].T@covMatInv@mean[:,class_index])
    # 6.分类，计算准确率
    count = 0               # 计算判断正确的个数
    numTest = test_images.shape[1]

    for test_num in range(numTest):
        test_sample = test_images[:, test_num][:, np.newaxis]
        g = [float('-inf') for i in range(classNum)]
        for test_class in range(classNum):
            g[test_class] = np.dot(w[:, test_class].T, test_sample)+w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            count = count+1
    print('PCA LDF methods', count/numTest)
    return


def qdf_methods(numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    QDF分类
    :return:
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
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
    for serach_class in range(classNum):
        indexMat[serach_class] = np.argwhere(train_labels == selectClass[serach_class])[:, 1]

        classesMat[serach_class] = train_images[:, indexMat[serach_class]]
        # 按行取均值
        mean[:, serach_class] = np.mean(classesMat[serach_class], axis=1)
        covMat[serach_class] = np.dot(np.cov(classesMat[serach_class]), (1-BETA))+np.dot(I, BETA)
        covMatInv[serach_class] = np.linalg.inv(covMat[serach_class])

    # 3.计算相关参数
    W = [[] for i in range(classNum)]            # 初始化W参数矩阵
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        W[class_index] = -0.5 * covMatInv[class_index]
        w[:,class_index] = np.dot(covMatInv[class_index], mean[:, class_index])
        w0[class_index] = (-0.5)*(mean[:, class_index].T@covMatInv[class_index]@mean[:, class_index])-\
                          0.5*math.log(np.linalg.det(covMat[class_index]))
    # 5.分类，计算准确率
    count = 0               # 计算判断正确的个数
    numTest = test_images.shape[1]

    for test_num in range(numTest):
        test_sample = test_images[:, test_num][:, np.newaxis]
        g = [float('-inf') for i in range(classNum)]
        for test_class in range(classNum):
            g[test_class] = test_sample.T @ W[test_class] @ test_sample + \
                            np.dot(w[:, test_class].T, test_sample) + \
                            w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            count = count+1
    print('QDF methods', count/numTest)
    return

def qdf_pca_methods(numOfPC=784, selectClass=np.array([0,1,2,3,4,5,6,7,8,9])):
    """
    QDF分类
    :return:
    """
    # 1.读取数据
    classNum = len(selectClass)
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")
    # 筛选
    if classNum == 10:
        pass
    else:
        train_images,train_labels =dataSelect(imageData=train_images,
                                              labelData=train_labels,
                                              selectClass=selectClass)
        test_images, test_labels = dataSelect(imageData=test_images,
                                                labelData=test_labels,
                                                selectClass=selectClass)
    # 2.PCA降维
    if numOfPC == 784:
        pass
    else:
        assert numOfPC >= 1 and numOfPC < 784
        estimator = PCA(n_components=numOfPC)
        # sklearn的PCA降维以行作为样本，因此需要转置
        train_images = estimator.fit_transform(train_images.T).T  # 将训练集的特征根据构建的PCA对象进行降维
        test_images = estimator.transform(test_images.T).T        # 将测试集使用相同的方法进行降维
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
    for serach_class in range(classNum):
        indexMat[serach_class] = np.argwhere(train_labels == selectClass[serach_class])[:, 1]

        classesMat[serach_class] = train_images[:, indexMat[serach_class]]
        # 按行取均值
        mean[:, serach_class] = np.mean(classesMat[serach_class], axis=1)
        # covMat[serach_class] = np.dot(np.cov(classesMat[serach_class]), (1-BETA))+np.dot(I, BETA)
        covMat[serach_class] = np.cov(classesMat[serach_class])
        covMatInv[serach_class] = np.linalg.inv(covMat[serach_class])

    # 3.计算相关参数
    W = [[] for i in range(classNum)]            # 初始化W参数矩阵
    w = np.zeros((numOfPC, classNum))            # 初始化w参数矩阵
    w0 = np.zeros((classNum,))  # 初始化w0参数向量
    for class_index in range(classNum):
        W[class_index] = -0.5 * covMatInv[class_index]
        w[:,class_index] = np.dot(covMatInv[class_index], mean[:, class_index])
        w0[class_index] = (-0.5)*(mean[:, class_index].T@covMatInv[class_index]@mean[:, class_index])-\
                          0.5*math.log(np.linalg.det(covMat[class_index]))
    # 5.分类，计算准确率
    count = 0               # 计算判断正确的个数
    numTest = test_images.shape[1]

    for test_num in range(numTest):
        test_sample = test_images[:, test_num][:, np.newaxis]
        g = [float('-inf') for i in range(classNum)]
        for test_class in range(classNum):
            g[test_class] = test_sample.T @ W[test_class] @ test_sample + \
                            np.dot(w[:, test_class].T, test_sample) + \
                            w0[test_class]
        out_label = g.index(max(g))
        if selectClass[out_label] == test_labels[:, test_num]:
            count = count+1
    print('PCA QDF methods', count/numTest)
    return

if __name__ == '__main__':
    train_image_file = './train-images.idx3-ubyte'
    train_label_file = './train-labels.idx1-ubyte'
    test_image_file = 't10k-images.idx3-ubyte'
    test_label_file = './t10k-labels.idx1-ubyte'

    save_train_dir = './train_images/'
    save_test_dir ='./test_images/'
    # 第一次使用，需改为True
    rewrite = False
    numOfPC = 100
    # 分类数量
    # 多分类
    # selectClass = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 二分类
    selectClass = np.array([5, 8])

    if rewrite:
        loadRawToNpy(train_image_file, train_label_file)
        loadRawToNpy(test_image_file, test_label_file)

    ldf_methods(selectClass=selectClass)
    # 简单的qdf会出错:math domain error,这是由于伪逆接近于0造成的，因此采用了降维操作
    # qdf_methods(selectClass=selectClass)
    ldf_pca_methods(numOfPC=numOfPC, selectClass=selectClass)
    qdf_pca_methods(numOfPC=numOfPC, selectClass=selectClass)
    # qdf_pca_methods(numOfPC=100)

    print('ok')