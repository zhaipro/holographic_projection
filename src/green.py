# encoding : utf-8
import os
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule


class GreenRenderingCore(object):

    def __init__(self, chromakey_bgr, simi_blend):
        with open('green_rendering_kernel.cu', 'r') as f:
            source = f.read()
        mod = SourceModule(source)
        self._func_rendering = mod.get_function('rendering_kernel')

        self._chromakey_bgr = np.array(chromakey_bgr, 'uint8')
        self._simi_blend = np.array(simi_blend, 'float32')

        self._in_uint8_gpu = None

    def process(self, img_in):
        h, w, _ = img_in.shape
        if self._in_uint8_gpu is None:
            self._in_uint8_gpu = gpuarray.zeros((h, w, 3), dtype=np.uint8)
            self._chromakey_bgr_gpu = gpuarray.to_gpu(self._chromakey_bgr)
            self._simi_blend_gpu = gpuarray.to_gpu(self._simi_blend)
            self._in_diff_gpu = gpuarray.zeros((h, w), dtype=np.float32)

        ## +++++++++++++++++++++++++++
        ## forward
        self._in_uint8_gpu.set(img_in)

        self._func_rendering(
            np.intc(h), np.intc(w),
            self._chromakey_bgr_gpu, self._simi_blend_gpu,
            self._in_uint8_gpu,
            self._in_diff_gpu,
            block=(16, 16, 1),
            grid=(8, 8, 1))                 # ?????

        return self._in_uint8_gpu.get()     # ?????


if __name__ == '__main__':
    import cv2
    chromakey_bgr = [68, 138, 81]
    simi_blend = [0.042, 0.049]

    greenrendering_mine = GreenRenderingCore(chromakey_bgr, simi_blend)

    ## -----------------------
    ## 单张测试
    img = cv2.imread("00000.jpg")
    img_rst = greenrendering_mine.process(img)
    cv2.imwrite('a.png', img_rst)

    diff = greenrendering_mine._in_diff_gpu.get()
    cv2.imwrite('mask.jpg', (((diff - 0.06) / 0.05 > 0.5) * 255).astype('uint8'))

    # img_gpu.copy()
