import numpy as np

def RotateClockWise180(img):
    new_img=np.zeros_like(img)
    h,w=img.shape[0],img.shape[1]
    for i in range(h): #上下翻转
        new_img[i]=img[h-i-1]
    return new_img
