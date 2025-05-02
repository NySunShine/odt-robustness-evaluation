# from tomo.data.preprocess import *
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, map_coordinates
import random
from einops import rearrange


class Normalize(object):
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def __call__(self, img):
        img = img.clip(self.min, self.max)
        return ((img - self.min) / (self.max - self.min)).astype(np.float32)


class ZNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return ((img - self.mean) / self.std).astype(np.float32)


def norm(x):
    _min = 1.337  # 5 percentile of training set
    _max = 1.455  # 95 percentile of training set
    img = x.clip(_min)
    return ((img - _min) / (_max - _min)).astype(np.float32)


def znorm(x):
    return ((x - 0.125255) / 0.144682).astype(np.float32)


def identity(x, mag):
    return x


def gamma_correction(x, mag):
    # if x.min() != 0:
    #     min_, max_ = x.min(), x.max()
    #     x_norm = (x - min_) / (max_ - min_)
    #     x_gc = x_norm ** mag
    #     return x_gc * (max_ - min_) + min_
    # else:

    return x ** mag


def invert(x, mag):
    min_, max_ = x.min(), x.max()
    x_norm = (x - min_) / (max_ - min_)
    x_ivt = 1 - x_norm
    return x_ivt * (max_ - min_) + min_


def noisy_label(x, y, mag):
    return x, y ** mag


def posterize(x, mag):
    mag = 10 ** (mag // 5 + 1)
    # if x.min() != 0:
    #     if x.min() == x.max():
    #         print(x)
    #     min_, max_ = x.min(), x.max()
    #     x_norm = (x - min_) / (max_ - min_)
    #     x_pos = (x_norm * mag).astype(np.int).astype(np.float32) / mag
    #     return x_pos * (max_ - min_) + min_
    # else:
    return (x * mag).astype(np.int).astype(np.float32) / mag


def additive_gaussian(x, mag):
    noise = np.random.random(x.shape) * mag
    return (x + noise).clip(0, 1)


def blur(x, mag):
    return gaussian_filter(x, mag).clip(0, 1)


def sharpness(x, mag):
    blur = gaussian_filter(x, 3)
    blur_2 = gaussian_filter(blur, 1)
    sharpened = blur + mag * (blur - blur_2)
    return sharpened.clip(0, 1)


def brightness(x, mag):
    return (x * mag).clip(0, x.max())


def elastic_deform(x_, mag):
    alpha, sigma = mag, 3 - mag

    shape = x_.shape
    random_state = np.random.RandomState(None)

    dx = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )

    _x, _y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    indices = np.reshape(_x + dx, (-1, 1)), np.reshape(_y + dy, (-1, 1))

    ret_x = np.zeros(shape)
    for i in range(shape[0]):
        ret_x[i] = map_coordinates(x_[i], indices, order=1, mode="reflect").reshape(
            shape[1:]
        )
    return ret_x


def axial_rotate(x, mag):
    if mag % 4 == 0:
        ret = x
    elif mag % 4 == 1:
        ret = rotate(x, 90, (1, 2), order=1)
    elif mag % 4 == 2:
        ret = rotate(x, 180, (1, 2), order=1)
    elif mag % 4 == 3:
        ret = rotate(x, 270, (1, 2), order=1)
    return ret.copy()


def resizing_crop_x(x, mag):
    # print(x.shape)
    resized = zoom(x, (1, 1, mag), order=1)
    offset = resized.shape[-1] - x.shape[-1]
    begin = offset // 2
    end = begin + x.shape[-1]
    return resized[:, begin:end]


def resizing_crop_y(x, mag):
    resized = zoom(x, (1, mag, 1), order=1)
    offset = resized.shape[-2] - x.shape[-2]
    begin = offset // 2
    end = begin + x.shape[-2]
    return resized[begin:end, :]


def cut_shuffle(*imgs, mag):
    pass


def flip(x, mag):
    if mag % 4 == 0:
        ret = np.array(x)[..., ::-1]
    elif mag % 4 == 1:
        ret = np.array(x)[:, ::-1]
    elif mag % 4 == 2:
        ret = np.array(x)[:, ::-1, ::-1]
    else:
        ret = x
    return ret.copy()


def transpose(x, mag):
    if mag % 2:
        ret = rearrange(x, "d h w -> d w h")
    else:
        ret = x
    return ret.copy()


def _rotate(x, mag):
    h, w = x.shape
    rot_x = rotate(x, mag, axes=(0, 1), order=1, reshape=False)
    _h, _w = rot_x.shape
    c_h = _h // 2 - h // 2
    c_w = _w // 2 - w // 2
    new_x = rot_x[c_h : c_h + h, c_w : c_w + w]
    return new_x


def augment_list():
    aug_list = [
        (identity, 0.0, 1.0),
        # (gamma_correction, 0.5, 1.5),
        # (invert, 0., 1.),
        # (noisy_label, 0.5, 1.5),
        # (posterize, 0, 30),
        # (additive_gaussian, 0.0, 0.3),
        # (blur, 0.0, 0.3),
        # (sharpness, 85.0, 100.0),
        # (brightness, 0.7, 1.3),
        (elastic_deform, 0.0, 3.0),
        # (axial_rotate, 0.0, 360.0),
        # (resizing_crop_x, 1.0, 1.3),
        # (resizing_crop_y, 1.0, 1.3),
        # (_rotate, -30, 30),
        # (flip, 0.0, 30.0),
    ]

    return aug_list


def cut_out(arr, size=32):
    if size:
        _, h, w = arr.shape
        x_beg = np.random.randint(-w, w)
        y_beg = np.random.randint(-h, h)
        x_end = x_beg + size
        y_end = y_beg + size
        x_beg = max(x_beg, 0)
        y_beg = max(y_beg, 0)
        x_end = min(x_end, w)
        y_end = min(y_end, h)
        arr[:, y_beg:y_end, x_beg:x_end] = 0.5
    return arr


class RandAugment:
    def __init__(self, n, m, size_cutout, value_cutout):
        self.n = n
        self.m = m
        self.size_cutout = size_cutout
        self.aug_list = augment_list()

    def __call__(self, x):
        ops = random.choices(self.aug_list, k=self.n)
        for op, minval, maxval in ops:
            if self.m == "None":
                m = np.random.randint(0, 30)
            else:
                m = self.m
            val = (float(m) / 30) * float(maxval - minval) + minval
            x = op(x, mag=val).copy().astype(np.float32)
        x = cut_out(x, self.size_cutout)
        return x


def random_z_crop3d(x, z_size=64):
    z = x.shape[0]
    z_offset = random.randint(0, z - z_size)

    return x[z_offset : z_offset + z_size]


def center_z_crop3d(i, z_size=64):
    z = i.shape[0]
    z_offset = (z - z_size) // 2

    return i[z_offset : z_offset + z_size]
