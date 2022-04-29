# -*- coding: utf-8 -*-
import time
from tkinter import *
from tkinter.filedialog import LoadFileDialog
from tkinter.ttk import Separator

import gray
import color
import texture
import blob

import cv2
import numpy as np
from PIL import Image, ImageTk

# -----------------------鼠标操作相关------------------------------------------
lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
count = 0
pointsMax = 6
img = 0
ROI = 0


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, count, pointsMax, window_name
    global lsPointsChoose, tpPointsChoose  # 存入选择的点
    global pointsCount  # 对鼠标按下的点计数
    global img2, ROI_bymouse_flag
    img2 = img.copy()  # 此行代码保证每次都重新再原图画  避免画多了
    # -----------------------------------------------------------
    #    count=count+1
    #    print("callback_count",count)
    # --------------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pointsCount = pointsCount + 1
        print('第' + str(pointsCount) + '次点击')
        # 感觉这里没有用？2018年8月25日20:06:42
        # 为了保存绘制的区域，画的点稍晚清零
        # if (pointsCount == pointsMax + 1):
        #     pointsCount = 0
        #     tpPointsChoose = []
        point1 = (x, y)
        # 画出点击的点
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        print(point1)
        # 将选取的点保存到list列表里
        lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
        tpPointsChoose.append((x, y))  # 用于画点
        # ----------------------------------------------------------------------
        print('点的个数：', len(tpPointsChoose))
        # 将所有的点都标红处理
        for i in range(len(tpPointsChoose)):
            cv2.circle(img2, tpPointsChoose[i], 2, (255, 0, 0), 2)
        # 将鼠标选的点用直线连起来
        for i in range(len(tpPointsChoose) - 1):
            print('i:', i, end=' --> ')
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            print(tpPointsChoose[i], '————', tpPointsChoose[i + 1])
        # ----------------------------------------------------------------------
        '''
        # ----------点击到pointMax时可以提取去绘图----------------
        if (pointsCount == pointsMax):
            # -----------绘制感兴趣区域-----------
            ROI_byMouse()
            ROI_bymouse_flag = 1

            i = 0
            pointsCount = 0
            tpPointsChoose = []
            lsPointsChoose = []
        '''

        # ----------点击到tpPointsChoose[0]附近时时可以提取去绘图----------------
        if (pointsCount > 2) and (x <= tpPointsChoose[0][0] + 3) and (x >= tpPointsChoose[0][0] - 3) and (
                y <= tpPointsChoose[0][1] + 3) and (y >= tpPointsChoose[0][1] - 3):
            # -----------绘制感兴趣区域-----------
            ROI_byMouse()
            ROI_bymouse_flag = 1
            i = 0
            pointsCount = 0
            tpPointsChoose = []
            lsPointsChoose = []
            cv2.destroyWindow(window_name)
            return

        cv2.imshow(window_name, img2)
    # -------------------------右键按下清除轨迹-----------------------------
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        cv2.imshow(window_name, img2)


def ROI_byMouse():
    global src, ROI, ROI_flag, mask2, lsPointsChoose, img_no
    mask = np.zeros(img.shape, np.uint8)
    print(lsPointsChoose)

    pts = np.array([lsPointsChoose], np.int32)  # pts是多边形的顶点列表（顶点集）
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制

    # --------------画多边形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    ##-------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    '''
    cv2.imwrite('mask.jpg', mask2)
    cv2.imshow('mask', mask2)
    '''
    ROI = cv2.bitwise_and(mask2, img)
    '''
    cv2.imwrite('ROI.jpg', ROI)
    cv2.imshow('ROI', ROI)
    '''
    # 截取规则图像
    rect = cv2.minAreaRect(pts)
    box = np.int0(cv2.boxPoints(rect))
    IMG = ROI[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]
    cv2.imwrite("IMG" + str(img_no) + ".jpg", IMG)
    # cv2.imshow('IMG', IMG)


filename = ""
window_name = ""


def myopen():
    global filename, img, ROI, window_name
    fd = LoadFileDialog(root)  # 创建打开文件对话框
    filename = fd.go()  # 显示打开文件对话框，并获取选择的文件名称
    print(filename)
    img = cv2.imread(filename)
    ROI = img.copy()
    window_name = "Picture" + str(img_no)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


img_no = 0  # 当前所选的图片的编号，取值1或2，默认0
is_choose_1 = False
is_choose_2 = False


def myopen1():
    global img_no, is_choose_1, lbl_img1, img1_new, cmp_msg
    img_no = 1
    is_choose_1 = True
    myopen()
    img_t = Image.open("IMG1.jpg")
    img1_new = ImageTk.PhotoImage(img_t)
    lbl_img1.configure(image=img1_new)
    lbl_img1.image = img1_new
    cmp_msg.set("成功打开图片1")

def myopen2():
    global img_no, is_choose_2, lbl_img2, img2_new, cmp_msg
    img_no = 2
    is_choose_2 = True
    myopen()
    img_t = Image.open("IMG2.jpg")
    img2_new = ImageTk.PhotoImage(img_t)
    lbl_img2.configure(image=img2_new)
    lbl_img2.image = img2_new
    cmp_msg.set("成功打开图片2")

def result():
    global cmp_msg
    if is_choose_1 == 0 and is_choose_2 == 0:
        cmp_msg.set("请先完成图片选择")
        return
    if is_choose_1 == 0:
        cmp_msg.set("请继续选择图片1")
        return
    if is_choose_2 == 0:
        cmp_msg.set("请继续选择图片2")
        return
    imgfile1 = cv2.imread("IMG1.jpg")
    imgfile2 = cv2.imread("IMG2.jpg")

    # 灰度相似度
    sim_gray = gray.GrayComparison(imgfile1, imgfile2)
    # 色度相似度
    sim_color = color.ColorComparison(imgfile1, imgfile2)
    # 纹理相似度
    sim_texture = texture.TexrtureComparison(imgfile1, imgfile2)
    # 斑块相似度
    sim_blob = blob.BlobComparison(imgfile1, imgfile2)

    cmp = "灰度相似度: " + str(round(sim_gray, 2)) + "\n色度相似度: " + str(round(sim_color, 2)) + "\n纹理相似度: " + str(
        round(sim_texture, 2)) + "\n斑块相似度: " + str(round(sim_blob, 2)) + "\n"
    cmp_msg.set(cmp)


# GUI设计
root = Tk()  # 建立根窗口，命名为root
root.title("图像相似度比较软件——18074305沈海健")  # 根窗口root设置标题
root.resizable(False, False)  # 根窗口root不可改变大小

frm_name = Frame(root)
frm_name.pack(side=TOP, anchor=CENTER)
Label(frm_name, text="提示：所选区域越接近矩形相似度越准确").pack(side="left")

Separator(root, orient=HORIZONTAL).pack(fill=X, padx=5)

frm_choose = Frame(root)
frm_choose.pack(side=TOP)
Button(frm_choose, text="打开图片1", command=myopen1).pack(side=LEFT)
Button(frm_choose, text="打开图片2", command=myopen2).pack(side=LEFT)

frm_show = Frame(root)
frm_show.pack(side=TOP, anchor=CENTER)
img_t = Image.open("chunbai.jpg")
img1 = ImageTk.PhotoImage(img_t)
img_t = Image.open("chunbai.jpg")
img2 = ImageTk.PhotoImage(img_t)

lbl_img1 = Label(frm_show, image=img1, text="图片1", compound=TOP)
lbl_img1.pack(side=LEFT)

lbl_img2 = Label(frm_show, image=img2, text="图片2", compound=TOP)
lbl_img2.pack(side=LEFT)

Separator(root, orient=HORIZONTAL).pack(fill=X, padx=5)

frm_result = Frame(root)
frm_result.pack(side=TOP)
Button(frm_result, text="计算相似度", command=result).pack(side=TOP)
cmp_msg = StringVar()  # 相似度比较结果文本内容需要更新，因此用textvariable —— StringVar()
result = Label(frm_result, textvariable=cmp_msg)
result.pack(side=TOP)
# 进入消息循环
root.mainloop()
