#!/usr/bin/python

# Standard imports
import cv2
import numpy as np
import math
from pprint import pprint

def getKeypoints(imgfile):
    imggray = cv2.cvtColor(imgfile, cv2.COLOR_BGR2GRAY)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # 设置阈值
    params.minThreshold = 10
    params.maxThreshold = 200

    # 是否限制 Blob 面积.
    params.filterByArea = True
    params.minArea = 10

    # 是否限制 Blob 圆形度
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Blob 面积与其凸包面积比
    params.filterByConvexity = True
    params.minConvexity = 0.87

    #  Blob  区域转动惯量最小值与最大值比值
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(imggray)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(imggray, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return keypoints


def keypointsComparison(shape1, keypoints1, shape2, keypoints2):
    # 将每幅图的等分成 10 * 10 份（area），统计每份（area）中斑块（blob）的个数

    # print(shape1)
    # print(shape2)
    # print()

    scale1_x = shape1[0] / 10
    scale1_y = shape1[1] / 10
    scale2_x = shape2[0] / 10
    scale2_y = shape2[1] / 10

    area1 = [[0 for i in range(10)] for i in range(10)]  # 生成 10 * 10 的二维全零列表
    area2 = [[0 for i in range(10)] for i in range(10)]  # 生成 10 * 10 的二维全零列表

    for point in keypoints1:
        x = point.pt[1]
        mod_x = x % scale1_x
        index_x = int((x - mod_x) / scale1_x)
        y = point.pt[0]
        mod_y = y % scale1_y
        index_y = int((y - mod_y) / scale1_y)
        area1[index_x][index_y] = area1[index_x][index_y] + 1

    for point in keypoints2:
        x = point.pt[1]
        mod_x = x % scale2_x
        index_x = int((x - mod_x) / scale2_x)
        y = point.pt[0]
        mod_y = y % scale2_y
        index_y = int((y - mod_y) / scale2_y)
        area2[index_x][index_y] = area2[index_x][index_y] + 1

    # pprint(area1)
    # print()
    # pprint(area2)
    # print()

    tempsum, sum = 0, 0
    for i in range(10):
        for j in range(10):
            tempmax = max(area1[i][j], area2[i][j])
            if tempmax == 0:
                tempmax = 1
            tempsum = pow((area1[i][j] - area2[i][j]), 2) / pow(tempmax, 2)
            sum += tempsum
    result = 1 - math.sqrt(sum / 100)
    return result


def BlobComparison(imgfile1, imgfile2):
    keypoints1 = getKeypoints(imgfile1)
    keypoints2 = getKeypoints(imgfile2)
    img_shape1 = imgfile1.shape
    img_shape2 = imgfile2.shape
    n = keypointsComparison(img_shape1, keypoints1, img_shape2, keypoints2)
    # print("斑块相似度：", n)
    return n

'''
imgfile1 = cv2.imread("kexuelou.jpg")
imgfile2 = cv2.imread("tushuguan.jpg")

BlobComparison(imgfile1, imgfile2)
'''