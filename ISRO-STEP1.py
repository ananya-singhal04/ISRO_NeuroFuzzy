#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:52:14 2020

@author: tanyagautam
"""

import numpy as np

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def pad(img, kernel_size, pad_with=pad_with):
    w, h = img.shape
    padding = int((kernel_size-1)/2)
    return np.pad(img, padding, pad_with)

def mean(img, kernel_size):
    w, h = img.shape
    out = [[np.mean(img[i: i+kernel_size, k: k+kernel_size]) for i in range(w-kernel_size+1)] for k in range(h-kernel_size+1)]
    out = np.array(out)
    return out

def wiener(img, kernel_size=3):
    img = np.asarray(img)
    img_pad = pad(img, kernel_size)
    
    local_mean = mean(img_pad, kernel_size)
    local_var = mean(img_pad**2, kernel_size)
    noise = np.mean(local_var.ravel())
    
    out = img - local_mean
    out *= (1-noise/(local_var+1e-8))
    out += local_mean
    out = np.where(local_var<noise, local_mean, out)
    return np.uint8(out)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
img = mpimg.imread('/Users/tanyagautam/Desktop/isro2.tif')

plt.imshow(img)
plt.show()

c = wiener(img)
c.shape