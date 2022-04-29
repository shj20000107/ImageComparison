from tkinter import *
from PIL import Image, ImageTk

root = Tk()  # 建立根窗口，命名为root
root.title("图像相似度比较软件")  # 根窗口root设置标题
root.resizable(False, False)  # 根窗口root不可改变大小
# frm_show = Frame(root)
img = Image.open("kexuelou.jpg")
img1 = ImageTk.PhotoImage(img)
img = Image.open("tushuguan.jpg")
img2 = ImageTk.PhotoImage(img)
Label(root, image=img1, text="图片1", compound=TOP).pack(side=LEFT)
Label(root, image=img2, text="图片2", compound=TOP).pack(side=LEFT)
root.mainloop()
