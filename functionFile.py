from sklearn.cluster import KMeans
import numpy as np
import PIL
import sklearn
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from collections import Counter
import skimage
from skimage import io, color
from skimage import filters


class pixel(object):
    def __init__(self, pos, rgbData):
        self.pos = pos
        self.rgbData = rgbData
        self.id = False


def pixelSplit(im_array):
    """
    实现将一幅彩色图像转化为多个像素类
    每个像素包含两个数据：在图像中的位置；rgb三通道数据
    返回一个pixel对象list
    """
    x, y, z = np.shape(im_array)
    result = []

    for i in range(x):
        for j in range(y):
            tempPixel = pixel(pos=[x, y], rgbData=im_array[x, y, :])
    return result


def getClusterResult(im_array):
    X, Y, Z = im_array.shape
    trainData = []
    for i in range(X):
        for j in range(Y):
            trainData.append(im_array[i, j, :])
    trainData = np.array(trainData)

    clf = KMeans(n_clusters=2, random_state=0).fit(trainData)
    res = clf.predict(trainData).reshape((X, Y))

    return res.reshape((X, Y))


def mainPixelChoose(res_array):
    """
    选取出现次数少的类别作为显著性区域
    """
    counterDict = Counter(res_array.reshape(-1))
    k1, k2 = counterDict.keys()
    num1 = counterDict[k1]
    num2 = counterDict[k2]
    if (num1 > num2):
        res = (res_array == k2).astype(int)
    else:
        res = (res_array == k1).astype(int)

    return res


def show2(res_array):
    inve = mainPixelChoose(res_array)
    res_unit8 = np.uint8(inve) * 255
    return Image.fromarray(res_unit8)


def getIntersection(res, lab_res):
    assert res.shape == lab_res.shape
    return (res & lab_res)


# 设置一种投票模拟器
def voter(*res_array):
    voterNum = len(res_array)  # 投票器个数

    for i in range(voterNum):
        assert res_array[0].shape == res_array[i].shape

    X, Y = res_array[0].shape

    threshold = voterNum // 2  # 投票阈值，大于此值设为0，小于设置为1

    res = np.zeros([X, Y])

    for i in range(voterNum):
        res = res + res_array[i]

    res = (res > threshold).astype(int)
    return res


def modeSmoothing(res_array, filterSize=[3, 3]):
    # 默认3*3滤波器
    # 只针对2分类, 且0, 1取值
    X, Y = res_array.shape

    paddle_x = (filterSize[0] // 2)
    paddle_y = (filterSize[1] // 2)

    #     assiMat = np.zeros([X + 2*paddle_x, Y + 2*paddle_y])
    #     assiMat[1:X+1, 1:Y+1] = res_array
    res = np.zeros([X, Y])
    res = res_array[:, :]
    classes = set(res_array.reshape(-1))
    classNum = len(classes)

    Size = filterSize[0] * filterSize[1]
    for i in range(paddle_x, X - paddle_x):
        for j in range(paddle_y, Y - paddle_y):
            temp = res[i - paddle_x:i + paddle_x, j - paddle_y:j + paddle_y].sum()
            #             print(temp)
            #             if (temp > (Size//2)):
            #                 res[i, j] = 1
            #             else:
            #                 res[i, j] = 0
            res[i, j] = np.median(temp)

    return res
# 设置平滑器，3x3滤波器，选择众数，5次迭代
# 继而尝试svm？


def picOutputing(fileNum, filePath='D:\\DIP_Practice\\(ASD)Image+GT'):
    fileName = str(fileNum) + '.jpg'
    picName = filePath + '\\' + fileName

    rgb = io.imread(picName)
    lab = color.rgb2lab(rgb)
    hsv = color.rgb2hsv(rgb)

    rgb_res = getClusterResult(rgb)
    lab_res = getClusterResult(lab)
    hsv_res = getClusterResult(hsv)

    res_vote = voter(lab_res, rgb_res, hsv_res)

    resPath = filePath + '\\' + 'res' + str(fileNum) + '.jpg'

    img = show2(res_vote)

    io.imsave(resPath, img)
#     return res_vote