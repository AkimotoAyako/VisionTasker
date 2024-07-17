import cv2
import numpy as np
from element.config.CONFIG_UIED import Config
C = Config()


def read_img(path, resize_height=None, kernel_size=None):

    def resize_by_height(org):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re

    try:
        img = cv2.imread(path)
        if kernel_size is not None:
            img = cv2.medianBlur(img, kernel_size)
        if img is None:
            print("*** Image does not exist ***")
            return None, None
        if resize_height is not None:
            img = resize_by_height(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    except Exception as e:
        print(e)
        print("*** Img Reading Failed ***\n")
        return None, None


def gray_to_gradient(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.copy(img)
    img_f = img_f.astype("float")

    kernel_h = np.array([[0,0,0], [0,-1.,1.], [0,0,0]])
    kernel_v = np.array([[0,0,0], [0,-1.,0], [0,1.,0]])
    dst1 = abs(cv2.filter2D(img_f, -1, kernel_h))
    dst2 = abs(cv2.filter2D(img_f, -1, kernel_v))
    gradient = (dst1 + dst2).astype('uint8')
    return gradient


def reverse_binary(bin, show=False):
    """
    Reverse the input binary image
    """
    r, bin = cv2.threshold(bin, 1, 255, cv2.THRESH_BINARY_INV)
    if show:
        cv2.imshow('binary_rev', bin)
        cv2.waitKey()
    return bin


def binarization(org, grad_min, show=False, write_path=None, wait_key=0):
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)  # 转灰度
    grad = gray_to_gradient(grey)        # get RoI with high gradient 计算图像的梯度（Gradient），获取具有高梯度的区域（可能是边缘区域）
    rec, binary = cv2.threshold(grad, grad_min, 255, cv2.THRESH_BINARY)    # enhance the RoI 通过阈值 grad_min 来将梯度图像分成两个部分，一部分设置为白色（255），另一部分设置为黑色（0） 这一步的目的是增强感兴趣区域
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (3, 3))  # remove noises 形态学运算去除噪点 （连接分散的目标区域，填充小孔）
    if write_path is not None:
        cv2.imwrite(write_path, morph)  # 如果指定了 write_path 参数，将处理后的二值图像保存到指定路径
    if show:
        cv2.imshow('binary', morph)  # 显示
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return morph
