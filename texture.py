#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pprint import pprint
import cv2
import math

# 定义最大灰度级数
gray_level = 16


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    # for i in range(len(ret)):
    #     for j in range(len(ret[i])):
    #         print(ret[i][j], end='\t')
    #     print()

    return ret


def feature_computer(p):
    Con = 0.0  # 角二阶矩
    Eng = 0.0  # 熵
    Asm = 0.0  # 对比度
    Idm = 0.0  # 反差分矩阵
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def getGLCMPara(imgfile):
    try:
        img_shape = imgfile.shape
    except:
        print('imread error')
        return

    imgresize = cv2.resize(imgfile, (32, 32), interpolation=cv2.INTER_CUBIC)

    imggray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(imggray, 1, 0)
    asm_0, con_0, eng_0, idm_0 = feature_computer(glcm_0)
    '''
    glcm_1=getGlcm(img_gray, 0,1)
    asm_1, con_1, eng_1, idm_1 = feature_computer(glcm_1)
    print(asm_1, con_1, eng_1, idm_1)
    glcm_2=getGlcm(img_gray, 1,1)
    asm_2, con_2, eng_2, idm_2 = feature_computer(glcm_2)
    print(asm_2, con_2, eng_2, idm_2)
    glcm_3=getGlcm(img_gray, -1,1)
    asm_3, con_3, eng_3, idm_3 = feature_computer(glcm_3)
    print(asm_3, con_3, eng_3, idm_3)
    '''
    return [asm_0, con_0, eng_0, idm_0]


def compareGLCMPara(GLCMPara1, GLCMPara2):
    tempsum, sum = 0, 0
    for i in range(len(GLCMPara1)):
        tempmax = max(GLCMPara1[i], GLCMPara2[i])
        if tempmax == 0:
            tempmax = 1
        tempsum = pow((GLCMPara1[i] - GLCMPara2[i]), 2) / pow(tempmax, 2)
        sum += tempsum
    return 1 - math.sqrt(sum / len(GLCMPara1))


def TexrtureComparison(imgfile1, imgfile2):
    GLCMpara1 = getGLCMPara(imgfile1)
    GLCMpara2 = getGLCMPara(imgfile2)
    n = compareGLCMPara(GLCMpara1, GLCMpara2)

    # print("纹理相似度", n)

    # print('图片一GLCM:')
    # print(GLCMpara1)
    # print("对比度, 角二阶矩, 熵, 反差分矩阵(图片一):")
    # print('图片二GLCM:')
    # print(GLCMpara2)
    # print("对比度, 角二阶矩, 熵, 反差分矩阵(图片二):")

    return n


# imgfile1 = cv2.imread("kexuelou.jpg")
# imgfile2 = cv2.imread("tushuguan.jpg")
#
# TexrtureComparison(imgfile1, imgfile2)
