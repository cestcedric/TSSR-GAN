import glob
import numpy
import os
from   skimage.measure import compare_ssim
import torch

'''
Adapted from https://github.com/thunil/TecoGAN/blob/master/metrics.py
'''


def im2tensor(image, imtype = numpy.uint8, cent = 1., factor = 255./2.):
    return torch.Tensor((image / factor - cent)[:, :, :, numpy.newaxis].transpose((3, 2, 0, 1)))


def listPNGinDir(dirpath):
    filelist = os.listdir(dirpath)
    filelist = [_ for _ in filelist if _.endswith(".png")] 
    filelist = [_ for _ in filelist if not _.startswith("IB")] 
    filelist = sorted(filelist)
    filelist.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    result = [os.path.join(dirpath,_) for _ in filelist if _.endswith(".png")]
    return result


def _rgb2ycbcr(img, maxVal=255):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    O = numpy.array([[16],
                  [128],
                  [128]])
    T = numpy.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = numpy.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = numpy.dot(t, numpy.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = numpy.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr


def to_uint8(x, vmin, vmax):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    x = x.astype('float32')
    x = (x-vmin)/(vmax-vmin)*255 # 0~255
    return numpy.clip(numpy.round(x), 0, 255)


def psnr(img_true, img_pred):
##### PSNR with color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
    diff =  Y_true - Y_pred
    rmse = numpy.sqrt(numpy.mean(numpy.power(diff,2)))
    return 20*numpy.log10(255./rmse)


def ssim(img_true, img_pred): ##### SSIM ##### 
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
    return compare_ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())


def crop_8x8( img ):
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    
    h = (ori_h//32) * 32
    w = (ori_w//32) * 32
    
    while(h > ori_h - 16):
        h = h - 32
    while(w > ori_w - 16):
        w = w - 32
    
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y+h, x:x+w]
    return crop_img, y, x