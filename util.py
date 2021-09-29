import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def displayImg(img_array, img_class, img_id, save_dir, img_save=False):
    """数据可视化"""
    # 图像归一化，变换至0~255
    img_array = img_array - min(img_array)
    img_array = img_array * (255 / max(img_array) )
    img_array = img_array.astype(np.uint8).reshape(28, 28, 1)
    # 调用函数可视化
    plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.close('all')
    # 保存图片
    if img_save:
        save_name = os.path.join(save_dir, '{}_{}.jpg'.format(img_id, img_class))
        cv2.imwrite(save_name, img_array)
    return


def loadRawToNpy(image_file, label_file):
    """ 读取原始数据文件，并保存为npy文件 """
    # 判断训练集与测试集
    if 'train' in os.path.basename(image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'
    # 读取文件，传入数据
    with open(image_file, 'rb') as f1:
        image_file = f1.read()
    with open(label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    image_all = []
    label_all = []
    # 读取每一张图片的数据，转化为array
    for i in range(num_file):
        label_all.append([label_file[i]])
        image_list = [item for item in image_file[i * 784:i * 784 + 784]]
        # 注意归一化并转换为float64，防止精度不够造成的误判
        image_all.append(np.array(image_list, dtype=np.float64)/255)
    # 每一列是一个样本
    label_all = np.array(label_all).T
    image_all = np.array(image_all).T
    # 保存转换后的数据
    np.save(prefix+'_images', image_all)
    np.save(prefix+'_labels', label_all)
    return


def dataSelect(imageData, labelData, selectClass):
    """ 根据selectClass对数据进行筛选 """
    select_list = []
    # 筛选出原始数据中类别在selectClass中的样本的索引
    for sample_id in range(imageData.shape[1]):
        if labelData[0, sample_id] in selectClass:
            select_list.append(sample_id)
    return imageData[:, select_list], labelData[:, select_list]


def accuracyDisplay(sampleCount, accuracyCount, classNum=10):
    """ 根据样本的每一类数量与分类结果正确的每一类数量进行正确率的打印输出 """
    assert sampleCount.shape == accuracyCount.shape
    assert sampleCount.shape[0] == classNum
    for classIndex in range(classNum):
        if sampleCount[classIndex] != 0:
            print('Class ', classIndex, ' accuracy: ', accuracyCount[classIndex]/sampleCount[classIndex])
    return
