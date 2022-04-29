import cv2 as cv
import numpy as np


# 均值哈希算法
def aHash(imgfile):
    # 缩放为32*32，并转化为灰度图
    imgresize = cv.resize(imgfile, (32, 32), interpolation=cv.INTER_CUBIC)
    imggray = cv.cvtColor(imgresize, cv.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(32):
        for j in range(32):
            s = s + imggray[i, j]
    # 求平均灰度
    avg = s / (32 * 32)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(32):
        for j in range(32):
            if imggray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值哈希算法
def dHash(imgfile):
    # 缩放为32*32，并转化为灰度图
    imgresize = cv.resize(imgfile, (33, 32), interpolation=cv.INTER_CUBIC)
    imggray = cv.cvtColor(imgresize, cv.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(32):
        for j in range(32):
            if imggray[i, j] > imggray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 感知哈希算法
def pHash(imgfile):
    # 缩放为64*64，并转化为灰度图
    imgresize = cv.resize(imgfile, (64, 64), interpolation=cv.INTER_CUBIC)
    imggray = cv.cvtColor(imgresize, cv.COLOR_BGR2GRAY)

    # 创建二维列表
    h, w = imggray.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = imggray  # 填充数据

    # 二维Dct变换
    vis1 = cv.dct(cv.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32, 32)

    # 把二维list变成一维list
    img_list = vis1.flatten()
    # img_list = np.array(vis1.tolist()).flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


# 均值、差值Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return (len(hash1) - n) / len(hash1)


# 感知Hash值对比
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    n = sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
    return 1 - n * 1. / (32 * 32 / 4)


def GrayComparison(imgfile1, imgfile2):
    ahash1 = aHash(imgfile1)
    ahash2 = aHash(imgfile2)
    differ1 = cmpHash(ahash1, ahash2)

    dhash1 = dHash(imgfile1)
    dhash2 = dHash(imgfile2)
    differ2 = cmpHash(dhash1, dhash2)

    phash1 = pHash(imgfile1)
    phash2 = pHash(imgfile2)
    differ3 = hammingDist(phash1, phash2)

    # print("图片一的均值哈希序列：\n", ahash1)
    # print("图片二的均值哈希序列：\n", ahash2)
    # print("二者的均值哈希相似度：\n", differ1)
    # print("图片一的差值哈希序列：\n", dhash1)
    # print("图片一的差值哈希序列：\n", dhash2)
    # print("二者的差值哈希相似度：\n", differ2)
    # print("图片一的感知哈希序列：\n", phash1)
    # print("图片一的感知哈希序列：\n", phash2)
    # print("二者的感知哈希相似度：\n", differ3)

    n = differ1 * 0.3 + differ2 * 0.2 + differ3 * 0.5
    # print('灰度相似度：', n)

    return n


# imgfile1 = cv.imread('tiankong1.jpg')
# imgfile2 = cv.imread('tiankong2.jpg')
# GrayComparison(imgfile1, imgfile2)
