#coding:utf-8

# 可以把图片转换成数组的形式，    ImageToMatrix
# 将数组转换成图片              MatrixToImage

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 将图片转换成数据 [28, 28]
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    #im.show()
    width, height = im.size  # 得到图片的宽和高
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype="float") / 255.0
    new_data = np.reshape(data, (width, height))
    return new_data

def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


filename = "test_num/8_3.png"
data = ImageToMatrix(filename)
print data


# 根据图片数据转换成图片
new_im = MatrixToImage(data)
plt.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
new_im.show()
new_im.save("lena_1.bmp")





