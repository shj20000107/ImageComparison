import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def create_rgb_hist(image):
    """"创建 RGB 三通道直方图（直方图矩阵）"""
    h, w, c = image.shape
    # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
    # 16*16*16的意思为三通道每通道有16个bins
    rgb_hist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
            index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
            # 该处形成的矩阵即为直方图矩阵
            rgb_hist[int(index), 0] += 1

    # plt.ylim([0, 10000])
    # plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)

    return rgb_hist


def hist_compare(hist1, hist2):
    """直方图比较函数"""
    '''# 创建第一幅图的rgb三通道直方图（直方图矩阵）
    hist1 = create_rgb_hist(image1)
    # 创建第二幅图的rgb三通道直方图（直方图矩阵）
    hist2 = create_rgb_hist(image2)'''
    # 进行直方图比较
    match = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    # print("相关性：%s" % match)
    return match


def handle_img(img):
    img = cv.resize(img, (100, 100))
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img[:, :, 2] = cv.equalizeHist(img[:, :, 2])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return img


def ColorComparison(imgfile1, imgfile2):
    img1 = handle_img(imgfile1)
    img2 = handle_img(imgfile2)

    hist1 = create_rgb_hist(img1)
    hist2 = create_rgb_hist(img2)

    plt.subplot(1, 2, 1)
    plt.title("hist1")
    plt.plot(hist1)
    plt.subplot(1, 2, 2)
    plt.title("hist2")
    plt.plot(hist2)

    match = hist_compare(hist1, hist2)
    # print("色度相似度：", match)

    # plt.show()
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return match


# imgfile1 = cv.imread('kexuelou.jpg')
# imgfile2 = cv.imread('tushuguan.jpg')
#
# ColorComparison(imgfile1, imgfile2)

